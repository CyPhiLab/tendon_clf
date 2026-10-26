"""Real Vicon measurement node.

Startup works from any robot pose (spirob_zmq.pose_init): wait until the markers are still, map
them into the MuJoCo frame (static base markers, or a saved frame from spirob_zmq.calibrate_frame),
find each segment's marker triangle by shape, label the markers by fitting the pose, and publish
that pose on INITIAL_STATE for the estimator. After that each frame is tracked
(spirob_zmq.marker_tracking) and published on SITE_MEASUREMENT in the MuJoCo frame, with NaN for
markers not seen in that frame. The unlabelled points of every frame are also published on
VICON_RAW (Vicon frame, before filtering), and INITIAL_STATE carries the Vicon -> MuJoCo frame, so a
recording is enough to rerun startup and tracking offline. Markers listed in unpublished_sites
(on the robot, but not used by the estimator) are labelled and tracked too, and not published.
"""

import time

import mujoco
import numpy as np

from spirob_zmq.common import INITIAL_STATE, ROBOT_STATE, SITE_MEASUREMENT, VICON_RAW, load_model, marker_sites
from spirob_zmq.core import REPO_ROOT, Node, run_node
from spirob_zmq.marker_tracking import MarkerTracker
from spirob_zmq.pose_init import (MarkerModel, load_frame, register_base_markers, stationary_clusters,
                                  transform)
from spirob_zmq.vicon import ViconClient


class RealMeasurementNode(Node):
    def __init__(self, params=None):
        super().__init__('real_measurement_node', params)
        log = self.get_logger()

        # Marker order used by the estimator; the tracked markers start with these
        self.site_names = list(self.declare_parameter('site_names'))
        self.tracked = marker_sites(self)
        self.model = load_model(self)
        key = self.model.key(self.declare_parameter('init_key', 'init')).id
        self.markers = MarkerModel(self.model, self.tracked, self.model.key_qpos[key])

        self.rate_hz = self.declare_parameter('vicon_rate_hz', 100.0)
        self.warmup_frames = self.declare_parameter('warmup_frames', 100)
        self.startup_attempts = self.declare_parameter('startup_attempts', 10)
        # Markers farther than this from the MuJoCo origin are ignored (m)
        self.max_marker_radius = self.declare_parameter('max_marker_radius', 0.45)
        self.max_frame_rms = self.declare_parameter('max_frame_rms', 2e-3)
        self.max_init_rms = self.declare_parameter('max_init_rms', 5e-3)
        # Static markers on the base mount: site names in the model (preferred frame reference)
        self.base_marker_sites = list(self.declare_parameter('base_marker_sites', []) or [])
        self.frame_path = self.declare_parameter('vicon_frame_path',
                                                 str(REPO_ROOT / 'ekf_results' / 'vicon_frame.json'))

        self.vicon = ViconClient(self.declare_parameter('vicon_ip', '172.20.52.190'), log, self.rate_hz)
        self.meas_pub = self.create_publisher(SITE_MEASUREMENT)
        self.init_pub = self.create_publisher(INITIAL_STATE)
        self.publish_raw = self.declare_parameter('publish_raw', True)
        self.raw_pub = self.create_publisher(VICON_RAW)

        stamp, points, result = self._startup()
        self.initial = {'q': result.q, 'rms': result.rms}
        m = self.markers
        self.tracker = MarkerTracker(self.tracked, points[[result.labels[n] for n in self.tracked]], stamp,
                                     gate=self.declare_parameter('track_gate', 0.015),
                                     reacquire_gate=self.declare_parameter('reacquire_gate', 0.025),
                                     triangle_tol=self.declare_parameter('triangle_tol', 0.0025),
                                     speed_gate=self.declare_parameter('speed_gate', 0.5),
                                     chain=m.segments, chord_max=m.chord_max, anchor=m.anchor)

        # The estimator's site positions, used to find markers again after they were lost
        self.predicted = None
        self.create_subscription(ROBOT_STATE, self._on_state, latest_only=True)

        # No latched topics: repeat the initial state so a late estimator still gets it
        self.init_timer = self.create_timer(0.5, self._publish_initial)
        self._publish_initial()
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _frame(self, centres):
        """Vicon -> MuJoCo frame as (R, t)."""
        log = self.get_logger()
        if self.base_marker_sites:
            data = mujoco.MjData(self.model)
            mujoco.mj_kinematics(self.model, data)
            ref = np.array([data.site(n).xpos.copy() for n in self.base_marker_sites])
            got = register_base_markers(centres, ref)
            if got is None:
                raise RuntimeError(f'base markers {self.base_marker_sites} not found among {len(centres)} '
                                   'Vicon markers')
            R, t, res = got
            if np.sqrt(np.mean(res ** 2)) > self.max_frame_rms:
                raise RuntimeError(f'base marker fit residual {np.round(1e3 * res, 1)} mm is too large')
            log.info(f'Vicon frame from base markers, residual {np.round(1e3 * res, 1)} mm')
            return R, t
        try:
            R, t, info = load_frame(self.frame_path)
        except FileNotFoundError:
            raise RuntimeError(f'no Vicon frame at {self.frame_path}: put the robot at rest and run '
                               '`python -m spirob_zmq.calibrate_frame`, or set base_marker_sites') from None
        log.info(f'Vicon frame from {self.frame_path} ({info["method"]}, {info["created"]}, '
                 f'residual {info["residual_mm"]} mm). Recalibrate if the Vicon origin or the base moved.')
        return R, t

    def _startup(self):
        log = self.get_logger()
        for attempt in range(1, self.startup_attempts + 1):
            stamps, frames = self.vicon.collect(self.warmup_frames)
            centres, still = stationary_clusters(frames)
            if not still:
                log.warn(f'startup {attempt}: markers are moving; waiting for the robot to be still')
                continue
            frame = self._frame(centres)
            points = transform(centres, frame)
            points = points[np.linalg.norm(points, axis=1) < self.max_marker_radius]
            result = self.markers.initialize(points)
            if result is None:
                log.warn(f'startup {attempt}: could not find all marker triangles among {len(points)} markers')
                continue
            per_marker = np.round([1e3 * result.residual[n] for n in self.tracked], 1)
            if result.rms > self.max_init_rms:
                log.warn(f'startup {attempt}: pose fit residual {1e3 * result.rms:.1f} mm too large {per_marker}')
                continue
            log.info(f'startup: labelled {len(self.tracked)} markers, pose fit rms {1e3 * result.rms:.1f} mm '
                     f'{per_marker}')
            self.frame = frame
            return stamps[-1], points, result
        raise RuntimeError(f'startup failed after {self.startup_attempts} attempts')

    def _on_state(self, msg):
        if msg.get('is_valid') and 'site_pos' in msg:
            site_pos = np.asarray(msg['site_pos'], dtype=float).reshape(-1, 3)
            unpublished = np.full((len(self.tracked) - len(site_pos), 3), np.nan)
            self.predicted = np.vstack([site_pos, unpublished])

    def _publish_initial(self):
        R, t = self.frame
        self.init_pub.publish({'stamp': time.time(), 'q': self.initial['q'], 'rms': self.initial['rms'],
                               'frame_R': R, 'frame_t': t})

    def _tick(self):
        got = self.vicon.read()
        if got is None:
            return
        frame_time, raw = got
        stamp = time.time()
        if self.publish_raw:
            self.raw_pub.publish({'stamp': stamp, 'frame_time': frame_time, 'points': raw.reshape(-1)})
        points = transform(raw, self.frame)
        points = points[np.linalg.norm(points, axis=1) < self.max_marker_radius]
        measured = self.tracker.update(points, frame_time, self.predicted)[:len(self.site_names)]
        self.meas_pub.publish({'stamp': stamp, 'data': measured.reshape(-1)})


def main(argv=None):
    run_node(RealMeasurementNode, argv)


if __name__ == '__main__':
    main()
