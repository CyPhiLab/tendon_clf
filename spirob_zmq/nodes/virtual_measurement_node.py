"""Simulated plant publishing noisy marker positions.

Like real_measurement_node it also publishes INITIAL_STATE: the pose fitted (spirob_zmq.pose_init)
to one shuffled, noisy snapshot of the markers, so the estimator starts the same way as on hardware.
init_pose='random' starts the plant curled: random tendon currents for 1.5 s, then released, with
tendon friction holding part of the curl.
"""

import mujoco
import numpy as np

from spirob_zmq.common import (INITIAL_STATE, MOTOR_STATE, SITE_MEASUREMENT, TRUE_STATE, VoltageLimit,
                               load_model, rest_state, step_current, substeps)
from spirob_zmq.pose_init import MarkerModel
from spirob_zmq.core import Node, run_node


class VirtualMeasurementNode(Node):
    def __init__(self, params=None):
        super().__init__('virtual_measurement_node', params)

        # Model used as the simulated "real" plant
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)
        rest_state(self, self.model, self.data)
        self.vlimit = VoltageLimit(self)
        seed = self.declare_parameter('seed', 0)
        self.rng = np.random.default_rng(seed if seed != 0 else None)
        if self.declare_parameter('init_pose', 'key') == 'random':
            current_max = self.declare_parameter('init_current_max', 0.8)
            n = int(round(1.5 / self.model.opt.timestep))
            pull = -self.rng.uniform(0.0, current_max, self.model.nu)
            step_current(self.model, self.data, pull, n, self.vlimit)
            step_current(self.model, self.data, np.zeros(self.model.nu), n, self.vlimit)
            self.data.time = 0.0
            self.get_logger().info(f'random start: pulled with {np.round(pull, 2)} A, then released')

        self.motor_ids = list(self.declare_parameter('motor_ids', [10, 1, 2]))

        # Sites to measure, match state_estimation_node's site_names
        site_names = self.declare_parameter(
            'site_names')
        self.site_ids = [self.model.site(name).id for name in site_names]

        # Simulation / publish rate
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)
        self.dt = 1.0 / self.rate_hz
        self.n_substeps = substeps(self.model, self.dt)

        # Measurement noise (meters, per xyz component, iid Gaussian)
        self.noise_std = self.declare_parameter('noise_std', 1e-4)
        # Noise mean/bias (meters): scalar, or [x, y, z] applied to every site
        self.noise_mean = np.asarray(self.declare_parameter('noise_mean', 0.0), dtype=float)

        # Latest applied motor current, from hardware_node
        self.ctrl_u = np.zeros(self.model.nu)
        self.create_subscription(MOTOR_STATE, self._on_motor_state)
        self.meas_pub = self.create_publisher(SITE_MEASUREMENT)
        self.truth_pub = self.create_publisher(TRUE_STATE)
        self.ee_site_id = self.model.site('ee').id

        # Initial state from one unlabelled snapshot, as on hardware
        mujoco.mj_kinematics(self.model, self.data)
        snapshot = self.data.site_xpos[self.site_ids] + self.rng.normal(0.0, self.noise_std, (len(self.site_ids), 3))
        perm = self.rng.permutation(len(self.site_ids))
        key = self.model.key(self.declare_parameter('init_key', 'init')).id
        result = MarkerModel(self.model, site_names, self.model.key_qpos[key]).initialize(snapshot[perm])
        if result is None or any(perm[result.labels[n]] != i for i, n in enumerate(site_names)):
            raise RuntimeError('initial pose fit failed to label the simulated markers')
        self.get_logger().info(f'initial pose fit rms {1e3 * result.rms:.2f} mm, '
                               f'max |q error| {np.abs(result.q - self.data.qpos).max():.3f} rad')
        self.initial = {'q': result.q, 'rms': result.rms}
        self.init_pub = self.create_publisher(INITIAL_STATE)
        self.init_timer = self.create_timer(0.5, self._publish_initial)
        self.timer = self.create_timer(self.dt, self._tick)

    def _publish_initial(self):
        self.init_pub.publish({'stamp': self.data.time, 'q': self.initial['q'], 'rms': self.initial['rms']})

    def _on_motor_state(self, msg):
        if msg['motor_id'] in self.motor_ids:
            self.ctrl_u[self.motor_ids.index(msg['motor_id'])] = msg['app_ctrl']

    def _tick(self):
        # Forward dynamics for one tick
        step_current(self.model, self.data, self.ctrl_u, self.n_substeps, self.vlimit)
        mujoco.mj_kinematics(self.model, self.data)

        # True site positions + additive Gaussian noise
        true_pos = np.concatenate([self.data.site(site_id).xpos.copy() for site_id in self.site_ids])
        mean = np.resize(self.noise_mean, true_pos.shape)  # broadcast scalar or tile [x, y, z]
        noisy_pos = true_pos + self.rng.normal(mean, self.noise_std, size=true_pos.shape)
        
        # Randomly set up to 3 sites as missing (NaN) to simulate occlusion
        noisy_pos_with_missing = noisy_pos.copy()
        if len(self.site_ids) > 0 and self.rng.random() < 0.5:  # 50% chance each step
            n_missing = self.rng.integers(1, min(4, len(self.site_ids) + 1))  # 1-3 missing markers
            missing_ids = self.rng.choice(len(self.site_ids), size=n_missing, replace=False)
            for missing_id in missing_ids:
                noisy_pos_with_missing[3*missing_id:3*missing_id+3] = np.nan
         
        stamp = self.data.time
        self.meas_pub.publish({'stamp': stamp, 'data': noisy_pos_with_missing})

        # Ground truth
        self.truth_pub.publish({
            'stamp': stamp,
            'q': self.data.qpos,
            'dq': self.data.qvel,
            'act': self.data.act,
            'site_pos': true_pos,
            'ee_pos': self.data.site_xpos[self.ee_site_id],
        })


def main(argv=None):
    run_node(VirtualMeasurementNode, argv)


if __name__ == '__main__':
    main()
