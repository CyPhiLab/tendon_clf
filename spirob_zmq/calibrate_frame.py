"""Measure the Vicon -> MuJoCo frame with the robot at rest and save it for real_measurement_node.

Put the robot at rest (motors off, hanging as in the model's settled init pose), then run

    python -m spirob_zmq.calibrate_frame [--out ekf_results/vicon_frame.json]

Rerun whenever the Vicon origin is re-set or the robot base moves. Not needed once static markers
on the base mount are configured (base_marker_sites).
"""

import argparse
import sys

import mujoco
import numpy as np

from spirob_zmq.common import ProfileParams, load_model, rest_state
from spirob_zmq.core import REPO_ROOT
from spirob_zmq.pose_init import MarkerModel, save_frame, stationary_clusters


def calibrate(centres, robot=None):
    params = ProfileParams(robot)
    model = load_model(params)
    data = mujoco.MjData(model)
    rest_state(params, model, data)
    key = model.key(params.declare_parameter('init_key', 'init')).id
    markers = MarkerModel(model, list(params.declare_parameter('site_names')), model.key_qpos[key])
    return markers, markers.calibrate_frame_at_rest(centres, data.qpos.copy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(REPO_ROOT / 'ekf_results' / 'vicon_frame.json'))
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--max-rms', type=float, default=2e-3, help='largest acceptable residual (m)')
    args = parser.parse_args(argv)

    from spirob_zmq.vicon import ViconClient
    params = ProfileParams()
    vicon = ViconClient(params.declare_parameter('vicon_ip'), params, params.declare_parameter('vicon_rate_hz', 100.0))
    _, frames = vicon.collect(args.frames)
    centres, still = stationary_clusters(frames)
    if not still:
        sys.exit('markers moved during the measurement; let the robot come to rest and rerun')
    markers, got = calibrate(centres)
    if got is None:
        sys.exit(f'could not find the marker triangles among {len(centres)} markers')
    R, t, res, _ = got
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f'rotation {angle:.2f} deg, translation {np.round(1e3 * t, 1)} mm, '
          f'segment {markers.segments[0]} residual {np.round(1e3 * res, 2)} mm')
    if np.sqrt(np.mean(res ** 2)) > args.max_rms:
        sys.exit('residual too large: check that the robot is at rest and the markers are intact')
    save_frame(args.out, R, t, res, 'rest_pose_segment_' + markers.segments[0])
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
