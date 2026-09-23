"""Simulated plant publishing the true state (no-EKF architecture)."""

import mujoco
import numpy as np

from spirob_zmq.common import MOTOR_COMMAND, ROBOT_STATE, command_to_u, load_model
from spirob_zmq.core import Node, run_node


class GetStateNode(Node):
    def __init__(self, params=None):
        super().__init__('get_state_node', params)
        self.model = load_model(self)

        # Initialize state to zero
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get the number of generalized coordinates, velocities, and controls
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        rate_hz = self.declare_parameter('rate_hz', 500.0)
        self.dt = 1.0 / rate_hz
        self.model.opt.timestep = self.dt
        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # Latest commanded control
        self.ctrl_u = np.zeros(self.nu)
        self.ee_site_id = self.model.site('ee').id
        self.create_subscription(MOTOR_COMMAND, self._on_command)
        self.state_pub = self.create_publisher(ROBOT_STATE)
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('get_state_node ready: simulated plant running')

    def _on_command(self, msg):
        self.ctrl_u = command_to_u(msg, self.motor_ids)

    def _tick(self):
        self.data.ctrl[:] = self.ctrl_u

        # Forward dynamics for one step
        mujoco.mj_step(self.model, self.data)

        # Get the current generalized coordinates and velocities
        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()

        # Get the current task position
        task_pos = self.data.site(self.ee_site_id).xpos.copy()

        # Get the current task velocity
        jac = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_site_id)
        task_vel = jac @ dq

        # Publish true robot state
        self.state_pub.publish({
            'stamp': self.now(),
            'q': q,
            'dq': dq,
            'task_pos': task_pos,
            'task_vel': task_vel,
            'is_valid': True,
        })


def main(argv=None):
    run_node(GetStateNode, argv)


if __name__ == '__main__':
    main()
