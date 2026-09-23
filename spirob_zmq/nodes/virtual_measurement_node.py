"""Simulated "real" plant publishing noisy site positions (EKF architecture)."""

import mujoco
import numpy as np

from spirob_zmq.common import MOTOR_COMMAND, SITE_MEASUREMENT, command_to_u, load_model
from spirob_zmq.core import Node, run_node


class VirtualMeasurementNode(Node):
    def __init__(self, params=None):
        super().__init__('virtual_measurement_node', params)

        # Model used as the simulated "real" plant
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)

        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # Sites to measure, match state_estimation_node's site_names
        site_names = self.declare_parameter(
            'site_names', ['ee_seg5', 'ee_seg10', 'ee_seg15', 'ee_seg20', 'ee'])
        self.site_ids = [self.model.site(name).id for name in site_names]

        # Simulation / publish rate
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)
        self.dt = 1.0 / self.rate_hz
        self.model.opt.timestep = self.dt

        # Measurement noise (meters, per xyz component, iid Gaussian)
        self.noise_std = self.declare_parameter('noise_std', 1e-10)
        seed = self.declare_parameter('seed', 0)
        self.rng = np.random.default_rng(seed if seed != 0 else None)

        # Latest commanded tendon forces, defaults to zero until control_node starts
        self.ctrl_u = np.zeros(self.model.nu)
        self.create_subscription(MOTOR_COMMAND, self._on_command)
        self.meas_pub = self.create_publisher(SITE_MEASUREMENT)
        self.timer = self.create_timer(self.dt, self._tick)

    def _on_command(self, msg):
        self.ctrl_u = command_to_u(msg, self.motor_ids)

    def _tick(self):
        # Forward dynamics for one step
        self.data.ctrl[:] = self.ctrl_u
        mujoco.mj_step(self.model, self.data)

        # True site positions + additive Gaussian noise
        true_pos = np.concatenate([self.data.site(sid).xpos.copy() for sid in self.site_ids])
        noisy_pos = true_pos + self.rng.normal(0.0, self.noise_std, size=true_pos.shape)
        self.meas_pub.publish({'stamp': self.now(), 'data': noisy_pos})


def main(argv=None):
    run_node(VirtualMeasurementNode, argv)


if __name__ == '__main__':
    main()
