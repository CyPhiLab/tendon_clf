"""Simulated plant publishing the true state."""

import mujoco
import numpy as np

from spirob_zmq.common import (MOTOR_COMMAND, ROBOT_STATE, VoltageLimit, command_to_u, load_model,
                               rest_state, step_current, substeps)
from spirob_zmq.core import Node, run_node


class GetStateNode(Node):
    def __init__(self, params=None):
        super().__init__('get_state_node', params)
        self.model = load_model(self)

        # Initial state
        self.data = mujoco.MjData(self.model)
        rest_state(self, self.model, self.data)

        # Get the number of generalized coordinates, velocities, and controls
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        rate_hz = self.declare_parameter('rate_hz', 500.0)
        self.dt = 1.0 / rate_hz
        self.n_substeps = substeps(self.model, self.dt)
        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # Latest commanded motor current
        self.ctrl_u = np.zeros(self.nu)
        self.vlimit = VoltageLimit(self)
        self.ee_site_id = self.model.site('ee').id
        self.create_subscription(MOTOR_COMMAND, self._on_command)
        self.state_pub = self.create_publisher(ROBOT_STATE)
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('get_state_node ready: simulated plant running')

    def _on_command(self, msg):
        self.ctrl_u = command_to_u(msg, self.motor_ids)

    def _tick(self):
        # Forward dynamics for one tick
        step_current(self.model, self.data, self.ctrl_u, self.n_substeps, self.vlimit)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

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
