"""Simulated plant publishing noisy marker positions."""

import mujoco
import numpy as np

from spirob_zmq.common import MOTOR_STATE, SITE_MEASUREMENT, TRUE_STATE, load_model, rest_state, substeps
from spirob_zmq.core import Node, run_node


class VirtualMeasurementNode(Node):
    def __init__(self, params=None):
        super().__init__('virtual_measurement_node', params)

        # Model used as the simulated "real" plant
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)
        rest_state(self, self.model, self.data)

        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # Sites to measure, match state_estimation_node's site_names
        site_names = self.declare_parameter(
            'site_names')
        self.site_ids = [self.model.site(name).id for name in site_names]

        # Simulation / publish rate
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)
        self.dt = 1.0 / self.rate_hz
        self.n_substeps = substeps(self.model, self.dt)

        # Measurement noise (meters, per xyz component, iid Gaussian)
        self.noise_std = self.declare_parameter('noise_std', 1e-10)
        seed = self.declare_parameter('seed', 0)
        self.rng = np.random.default_rng(seed if seed != 0 else None)

        # Latest applied control, from hardware_node
        self.ctrl_u = np.zeros(self.model.nu)
        self.create_subscription(MOTOR_STATE, self._on_motor_state)
        self.meas_pub = self.create_publisher(SITE_MEASUREMENT)
        self.truth_pub = self.create_publisher(TRUE_STATE)
        self.ee_site_id = self.model.site('ee').id
        self.timer = self.create_timer(self.dt, self._tick)

    def _on_motor_state(self, msg):
        if msg['motor_id'] in self.motor_ids:
            self.ctrl_u[self.motor_ids.index(msg['motor_id'])] = msg['app_ctrl']

    def _tick(self):
        # Forward dynamics for one tick
        self.data.ctrl[:] = self.ctrl_u
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_kinematics(self.model, self.data)

        # True site positions + additive Gaussian noise
        true_pos = np.concatenate([self.data.site(sid).xpos.copy() for sid in self.site_ids])
        noisy_pos = true_pos + self.rng.normal(0.0, self.noise_std, size=true_pos.shape)
        stamp = self.data.time
        self.meas_pub.publish({'stamp': stamp, 'data': noisy_pos})

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
