"""Real Vicon measurement node.

Initializes MuJoCo from the model's init keyframe to obtain reference marker positions,
pulls unlabeled Vicon markers, matches them by nearest position via Hungarian algorithm,
and publishes the ordered measurement vector on SITE_MEASUREMENT.
"""

import time

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from vicon_dssdk import ViconDataStream

from spirob_zmq.common import SITE_MEASUREMENT, load_model
from spirob_zmq.core import Node, run_node


class RealMeasurementNode(Node):
    def __init__(self, params=None):
        super().__init__('real_measurement_node', params)

        # Marker order used by the estimator.
        site_names = self.declare_parameter('site_names')
        self.site_names = list(site_names)
        self.site_meas = 3 * len(self.site_names)

        # Load MuJoCo model and get reference positions from init keyframe.
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)
        self.site_ids = [self.model.site(name).id for name in self.site_names]

        # Set to init keyframe and compute reference positions
        init_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'init')
        self.data.qpos[:] = self.model.key_qpos[init_key_id]
        mujoco.mj_forward(self.model, self.data)

        # Extract reference site positions from the init configuration
        self.ref_pos = np.array([
            self.data.site(site_id).xpos.copy() for site_id in self.site_ids], dtype=float)
        self.get_logger().info(f'Loaded reference positions from init keyframe: {self.ref_pos.shape}')

        # Vicon configuration.
        self.vicon_ip = self.declare_parameter('vicon_ip', '172.20.52.190')
        self.rate_hz = self.declare_parameter('vicon_rate_hz', 100.0)
        self.dt = 1.0 / self.rate_hz

        # Warmup / DBSCAN configuration.
        self.warmup_frames = self.declare_parameter('warmup_frames', 100)
        self.dbscan_eps = self.declare_parameter('dbscan_eps', 1.5e-3)  # 1.5 mm
        self.dbscan_min_samples = self.declare_parameter('dbscan_min_samples', 10)
        self.max_match_distance = self.declare_parameter('max_match_distance', 0.03)  # 3 cm
        # Reject any Vicon marker at or beyond this distance from the origin (m)
        self.max_marker_radius = self.declare_parameter('max_marker_radius', 0.45)

        # Vicon client.
        self.client = ViconDataStream.Client()
        self._connect_vicon()

        # Publishers.
        self.meas_pub = self.create_publisher(SITE_MEASUREMENT)

        # Initial ordered marker estimate from warmup. After that, each new frame is
        # matched against the previous ordered positions, with missing detections
        # carried forward.
        self.last_measurements = np.full((len(self.site_names), 3), np.nan, dtype=float)
        self._warmup_initial_measurements()

        # Run at the Vicon rate.
        self.timer = self.create_timer(self.dt, self._tick)

    def _connect_vicon(self):
        self.get_logger().info(f'Connecting to Vicon at {self.vicon_ip}...')
        while not self.client.IsConnected():
            try:
                self.client.Connect(self.vicon_ip)
            except ViconDataStream.DataStreamException:
                self.get_logger().warn('Vicon connect failed; retrying...', throttle_duration_sec=1.0)
                time.sleep(1)

        self.get_logger().info('Connected to Vicon.')
        self.client.EnableMarkerData()
        self.client.EnableUnlabeledMarkerData()
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)

    def _match_markers(self, ref_xyz, markers_xyz):
        """Match unlabeled markers to reference positions via Hungarian assignment.

        Rejects matches where distance exceeds max_match_distance (3.5 cm).
        """
        n_ref = ref_xyz.shape[0]
        matched = np.full((n_ref, 3), np.nan, dtype=float)
        max_match_distance = self.max_match_distance  # 3.5 cm in meters

        if markers_xyz.shape[0] == 0:
            return matched

        # If Vicon reports more markers than MuJoCo has sites, keep only the
        # detected markers that are closest to the reference set overall.
        if markers_xyz.shape[0] > n_ref:
            diff_all = ref_xyz[:, None, :] - markers_xyz[None, :, :]
            cost_all = np.linalg.norm(diff_all, axis=2)
            keep_idx = np.argsort(np.min(cost_all, axis=0))[:n_ref]
            markers_xyz = markers_xyz[keep_idx]

        # Cost matrix of pairwise Euclidean distances.
        diff = ref_xyz[:, None, :] - markers_xyz[None, :, :]
        cost = np.linalg.norm(diff, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_match_distance:
                matched[r] = markers_xyz[c]

        return matched

    def _match_current_frame_to_previous(self, current_markers):
        """Assign current-frame unlabeled detections to the previous ordered markers.

        Strategy:
            - keep the previous marker positions as the reference ordering
            - greedily match each expected marker to the nearest unused current-point
              within max_match_distance
            - if a marker is not seen this frame, keep the previous value
        """
        n_markers = len(self.site_names)
        matched = self.last_measurements.copy()
        if current_markers.size == 0:
            return matched

        used = np.zeros(len(current_markers), dtype=bool)
        for i in range(n_markers):
            prev = self.last_measurements[i]
            if np.any(np.isnan(prev)):
                continue
            diffs = current_markers - prev[np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=1)
            candidate_idx = np.where(~used & (dists <= self.max_match_distance))[0]
            if len(candidate_idx) == 0:
                continue
            j = candidate_idx[np.argmin(dists[candidate_idx])]
            matched[i] = current_markers[j]
            used[j] = True

        return matched

    def _read_markers(self, unlabeled):
        """Vicon unlabeled markers in meters, shape (N, 3), dropping any with |p| >= max_marker_radius."""
        pts = np.array([
            [marker[0][0] / 1000.0, marker[0][1] / 1000.0, marker[0][2] / 1000.0]
            for marker in unlabeled
        ], dtype=float).reshape(-1, 3)
        return pts[np.linalg.norm(pts, axis=1) < self.max_marker_radius]

    def _warmup_initial_measurements(self):
        """Collect a short Vicon burst and use DBSCAN to estimate initial marker centers."""
        self.get_logger().info(f'Running DBSCAN warmup for {self.warmup_frames} frames...')

        warmup_points = []
        while len(warmup_points) < self.warmup_frames:
            if not self.client.GetFrame():
                continue
            try:
                unlabeled = self.client.GetUnlabeledMarkers()
            except ViconDataStream.DataStreamException:
                continue

            raw = self._read_markers(unlabeled)

            if raw.size > 0:
                warmup_points.append(raw)

        if not warmup_points:
            self.get_logger().warn('DBSCAN warmup produced no Vicon points; falling back to MuJoCo init reference.')
            self.last_measurements = self.ref_pos.copy()
            return

        all_points = np.vstack(warmup_points)
        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(all_points)

        cluster_centers = []
        for cluster_id in sorted(set(labels)):
            if cluster_id < 0:
                continue
            cluster_points = all_points[labels == cluster_id]
            cluster_centers.append({
                'cluster_id': int(cluster_id),
                'count': int(len(cluster_points)),
                'center': cluster_points.mean(axis=0),
            })
        cluster_centers.sort(key=lambda item: item['count'], reverse=True)

        if not cluster_centers:
            self.get_logger().warn('DBSCAN warmup found no valid clusters; falling back to MuJoCo init reference.')
            self.last_measurements = self.ref_pos.copy()
            return

        cluster_xyz = np.array([entry['center'] for entry in cluster_centers], dtype=float)
        matched = self._match_markers(self.ref_pos, cluster_xyz)
        valid = ~np.any(np.isnan(matched), axis=1)
        self.last_measurements = matched[valid]
        if valid.sum() > 0:
            # Fill missing markers with initial reference pose if some were not matched.
            self.last_measurements = np.full((len(self.site_names), 3), np.nan, dtype=float)
            for i in range(len(self.ref_pos)):
                if not np.any(np.isnan(matched[i])):
                    self.last_measurements[i] = matched[i]
            if np.isnan(self.last_measurements).all():
                self.last_measurements = self.ref_pos.copy()
        else:
            self.last_measurements = self.ref_pos.copy()

        self.get_logger().info(
            f'DBSCAN warmup complete: {len(cluster_centers)} clusters, '
            f'initial marker estimate for {np.isfinite(self.last_measurements).sum() / 3:.0f} markers.'
        )

    def _tick(self):
        if not self.client.GetFrame():
            return

        try:
            unlabeled = self.client.GetUnlabeledMarkers()
        except ViconDataStream.DataStreamException:
            return

        current_markers = self._read_markers(unlabeled)

        smoothed = self._match_current_frame_to_previous(current_markers)
        self.last_measurements = smoothed

        stamp = time.time()
        self.meas_pub.publish({'stamp': stamp, 'data': smoothed.reshape(-1)})


def main(argv=None):
    run_node(RealMeasurementNode, argv)


if __name__ == '__main__':
    main()


