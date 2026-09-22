import numpy as np
import mujoco
from pathlib import Path
import rclpy
from rclpy.node import Node
from spirob_interfaces.msg import MotorCommand, RobotState

class GetStateNode(Node):
    def __init__(self):
        super().__init__('get_state_node')

        model_path = self.declare_parameter(
            'model_path',
            'mujoco_models/spirob/spirob_control.xml'
        ).get_parameter_value().string_value

        if not Path(model_path).exists():
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)

        # Same physical parameters as standalone id-clf-qp script
        self.model.jnt_stiffness[:] = 0.3
        self.model.dof_damping[:] = 0.1
        self.model.opt.gravity[:] = [0, 0, -9.81]

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
        rate_hz = self.declare_parameter('rate_hz', 500.0).get_parameter_value().double_value
        self.dt = 1.0 / rate_hz
        self.model.opt.timestep = self.dt
        motor_ids = self.declare_parameter('motor_ids', [0, 1, 2]).get_parameter_value().integer_array_value
        self.motor_ids = list(motor_ids)

        # Latest commanded control
        self.ctrl_u = np.zeros(self.nu)
        self.ee_site_id = self.model.site('ee').id
        self.cmd_sub = self.create_subscription(MotorCommand, '/spirob/motor_command', self._on_command, 10)
        self.state_pub = self.create_publisher(RobotState, '/spirob/robot_state', 10)
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('get_state_node ready: simulated plant running')

    def _on_command(self, msg: MotorCommand):
        if list(msg.motor_ids) == self.motor_ids:
            self.ctrl_u = np.asarray(msg.u, dtype=float).copy()
            return
        id_to_u = dict(zip(msg.motor_ids, msg.u))
        self.ctrl_u = np.array([id_to_u.get(motor_id, 0.0) for motor_id in self.motor_ids], dtype=float)

    def _tick(self):
        t_wall = self.get_clock().now().nanoseconds * 1e-9
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
        msg = RobotState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.q = q.tolist()
        msg.dq = dq.tolist()
        msg.task_pos = task_pos.tolist()
        msg.task_vel = task_vel.tolist()
        msg.is_valid = True
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GetStateNode()
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