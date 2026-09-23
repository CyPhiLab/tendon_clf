from pathlib import Path
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from spirob_interfaces.msg import MotorCommand

class VirtualMeasurementNode(Node):
    def __init__(self):
        super().__init__('virtual_measurement_node')

        # Model used as the simulated "real" plant
        model_path = self.declare_parameter(
            'model_path', 'mujoco_models/spirob/spirob_control.xml').get_parameter_value().string_value
        if not Path(model_path).exists():
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        motor_ids = self.declare_parameter(
            'motor_ids', [0, 1, 2]).get_parameter_value().integer_array_value
        self.motor_ids = list(motor_ids)

        # Sites to measure, match state_estimation_node's site_names
        site_names = self.declare_parameter(
            'site_names',
            ['ee_seg5', 'ee_seg10', 'ee_seg15', 'ee_seg20', 'ee']
        ).get_parameter_value().string_array_value
        self.site_ids = [self.model.site(name).id for name in site_names]

        # Simulation / publish rate
        self.rate_hz = self.declare_parameter('rate_hz', 100.0).get_parameter_value().double_value
        self.dt = 1.0 / self.rate_hz
        self.model.opt.timestep = self.dt

        # Measurement noise (meters, per xyz component, iid Gaussian)
        self.noise_std = self.declare_parameter('noise_std', 1e-10).get_parameter_value().double_value
        seed = self.declare_parameter('seed', 0).get_parameter_value().integer_value
        self.rng = np.random.default_rng(seed if seed != 0 else None)

        # Latest commanded tendon forces, defaults to zero until control_node starts
        self.ctrl_u = np.zeros(self.model.nu)
        self.cmd_sub = self.create_subscription(MotorCommand, '/spirob/motor_command', self._on_command, 10)
        self.meas_pub = self.create_publisher(Float64MultiArray, '/spirob/site_measurement', 10)
        self.timer = self.create_timer(self.dt, self._tick)

    def _on_command(self, msg: MotorCommand):
        if list(msg.motor_ids) == self.motor_ids:
            self.ctrl_u = np.asarray(msg.u, dtype=float).copy()
            return
        id_to_u = dict(zip(msg.motor_ids, msg.u))
        self.ctrl_u = np.array([id_to_u.get(motor_id, 0.0) for motor_id in self.motor_ids], dtype=float)

    def _tick(self):
        # Forward dynamics for one step
        self.data.ctrl[:] = self.ctrl_u
        mujoco.mj_step(self.model, self.data)

        # True site positions + additive Gaussian noise
        true_pos = np.concatenate([self.data.site(sid).xpos.copy() for sid in self.site_ids])
        noisy_pos = true_pos + self.rng.normal(0.0, self.noise_std, size=true_pos.shape)
        msg = Float64MultiArray()
        msg.data = noisy_pos.tolist()
        self.meas_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualMeasurementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
