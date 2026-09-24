"""MuJoCo viewer showing the estimated robot state."""

import mujoco
import mujoco.viewer
import numpy as np

from spirob_zmq.common import ROBOT_STATE, load_model
from spirob_zmq.core import Node, run_node


class SimulationNode(Node):
    def __init__(self, params=None):
        super().__init__('simulation_node', params)

        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        target = self.declare_parameter('target_pos')
        if target is not None and self.model.body('target').mocapid[0] >= 0:
            self.data.mocap_pos[self.model.body('target').mocapid[0]] = target
        self.have_state = False
        viewer_rate_hz = self.declare_parameter('viewer_rate_hz', 50.0)

        # Subscribe to the ekf corrected state from state_estimation_node
        self.create_subscription(ROBOT_STATE, self._on_state, latest_only=True)

        # Viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer_timer = self.create_timer(1.0 / viewer_rate_hz, self._update_viewer)
        self.get_logger().info(f'simulation_node ready, waiting for {ROBOT_STATE}')

    def _on_state(self, msg):
        if not msg['is_valid']:
            return

        if len(msg['q']) != self.nq:
            self.get_logger().warn(
                f"Received q with {len(msg['q'])} elements, expected {self.nq}; ignoring",
                throttle_duration_sec=2.0)
            return

        with self.viewer.lock():
            self.data.qpos[:] = np.asarray(msg['q'], dtype=float)
            mujoco.mj_fwdPosition(self.model, self.data)

        self.have_state = True

    def _update_viewer(self):
        if not self.viewer.is_running():
            self.get_logger().info('Viewer closed, shutting down')
            self.shutdown()
            return
        self.viewer.sync()

    def destroy_node(self):
        viewer = getattr(self, 'viewer', None)
        if viewer is not None:
            try:
                if viewer.is_running():
                    viewer.close()
            except Exception as exc:
                self.get_logger().warn(f'Error closing viewer: {exc}')
        super().destroy_node()


def main(argv=None):
    run_node(SimulationNode, argv)


if __name__ == '__main__':
    main()
