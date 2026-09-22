from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from spirob_interfaces.msg import RobotState


class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')

        model_path = self.declare_parameter(
            'model_path',
            '/home/zach/huy/tendon_clf/mujoco_models/spirob/spirob_control.xml'
        ).get_parameter_value().string_value

        viewer_rate_hz = self.declare_parameter(
            'viewer_rate_hz', 50.0).get_parameter_value().double_value

        if not Path(model_path).exists():
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.have_state = False
        self._shutting_down = False

        # Subscribe to the ekf corrected state from state_estimation_node
        self.state_sub = self.create_subscription(RobotState, '/spirob/robot_state', self._on_state, 1)

        # Viewer 
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer_timer = self.create_timer(1.0 / viewer_rate_hz, self._update_viewer)
        self.get_logger().info('simulation_node ready, waiting for /spirob/robot_state')

    def _on_state(self, msg: RobotState):
        if not msg.is_valid:
            return

        if len(msg.q) != self.nq:
            self.get_logger().warn(
                f"Received q with {len(msg.q)} elements, expected {self.nq}; ignoring",
                throttle_duration_sec=2.0)
            return

        with self.viewer.lock():
            self.data.qpos[:] = np.asarray(msg.q, dtype=float)
            mujoco.mj_fwdPosition(self.model, self.data)

        self.have_state = True

    def _update_viewer(self):
        if self._shutting_down:
            return

        if not self.viewer.is_running():
            self.get_logger().info('Viewer closed, shutting down')
            self._shutting_down = True
            if rclpy.ok():
                rclpy.shutdown()
            return
        self.viewer.sync()

    def destroy_node(self):
        self._shutting_down = True
        viewer = getattr(self, 'viewer', None)
        if viewer is not None:
            try:
                if viewer.is_running():
                    viewer.close()
            except Exception as exc:
                self.get_logger().warn(f'Error closing viewer: {exc}')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()