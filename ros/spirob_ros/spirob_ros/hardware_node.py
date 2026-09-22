import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import rclpy
from rclpy.node import Node
from spirob_interfaces.msg import MotorCommand, MotorState
from motor_control.ak_motor.motor_can import *

class HardwareNode(Node):
    def __init__(self):
        super().__init__('hardware_node')

        # Motor ids in the same order u_ctrl
        motor_ids = self.declare_parameter('motor_ids', [0, 1, 2]).get_parameter_value().integer_array_value
        self.motor_ids = list(motor_ids)

        # CAN bus settings
        self.channel = self.declare_parameter('channel', 'PCAN_USBBUS1').get_parameter_value().string_value
        self.bitrate = self.declare_parameter('bitrate', 1_000_000).get_parameter_value().integer_value

        # Motor parameters
        self.k_t = self.declare_parameter('k_t', 0.127).get_parameter_value().double_value
        self.r_spool = self.declare_parameter('r_spool', 0.05).get_parameter_value().double_value
        self.gear = self.declare_parameter('gear', 10.0).get_parameter_value().double_value
        self.current_rated = self.declare_parameter('current_rated', 1.9).get_parameter_value().double_value
        self.gear_mujoco = self.declare_parameter('gear_mujoco', self.k_t * self.gear * self.current_rated / self.r_spool).get_parameter_value().double_value

        # Control loop rate (CAN feedback loop is set to 100 Hz)
        self.rate_hz = self.declare_parameter('rate_hz', 100.0).get_parameter_value().double_value
        self.dry_run = self.declare_parameter('dry_run', True).get_parameter_value().bool_value
        self.cmd_u = None

        # Publisher for per-motor state
        self.state_pub = self.create_publisher(MotorState, '/spirob/motor_state', 10)

        # Subscriber for tendon-force commands
        self.cmd_sub = self.create_subscription(MotorCommand, '/spirob/motor_command', self._on_command, 10)

        # Open the motor controller (servo/current mode)
        self.controller = None
        if not self.dry_run:
            self.controller = AKController.make_servo(
                motor_ids=self.motor_ids,
                channel=self.channel,
                bitrate=self.bitrate,
            )
            self.controller.open()
        else:
            self.get_logger().warn('hardware_node running in dry_run mode -- no CAN bus opened')

        # Send current commands
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)

    def _on_command(self, msg: MotorCommand):
        if list(msg.motor_ids) == self.motor_ids:
            self.cmd_u = list(msg.u)
        else:
            id_to_u = dict(zip(msg.motor_ids, msg.u))
            self.cmd_u = [id_to_u.get(motor_id, 0.0) for motor_id in self.motor_ids]


    def _on_timer(self):
        # Convert tendon force (N) -> motor current (A)
        if self.cmd_u is None:
            return

        currents = [u * self.gear_mujoco * self.r_spool / (self.k_t * self.gear) for u in self.cmd_u]
        if self.dry_run:
            feedbacks = [None] * len(self.motor_ids)
        else:
            self.controller.set_current(currents)
            motors_relaxed = [self.controller.motors[i] for i, current in enumerate(currents) if current == 0.0]
            self.controller.disable(motors_relaxed)
            feedbacks = self.controller.get_feedback_all_motors()

        stamp = self.get_clock().now().to_msg()
        for motor_id, fb in zip(self.motor_ids, feedbacks):
            state = MotorState()
            state.header.stamp = stamp
            state.motor_id = motor_id
            if fb is not None:
                state.position = fb['position']
                state.velocity = fb['speed']
                state.current = fb['current']
                state.app_ctrl = fb['current'] * self.k_t * self.gear / (self.r_spool * self.gear_mujoco)
                state.fault_code = fb['error_code']
            else:
                state.position = 0.0
                state.velocity = 0.0
                state.current = 0.0
                state.app_ctrl = self.cmd_u[self.motor_ids.index(motor_id)]
                state.fault_code = 0
            self.state_pub.publish(state)

    def destroy_node(self):
        if self.controller is not None:
            try:
                self.controller.disable(self.controller.motors)
                self.controller.close()
            except Exception as exc:
                self.get_logger().warn(f'Error while shutting down motor controller: {exc}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
