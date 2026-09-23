"""Converts tendon-force commands to AK motor currents and publishes motor feedback."""

from spirob_zmq.common import MOTOR_COMMAND, MOTOR_STATE, command_to_u
from spirob_zmq.core import Node, run_node


class HardwareNode(Node):
    def __init__(self, params=None):
        super().__init__('hardware_node', params)

        # Motor ids in the same order u_ctrl
        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # CAN bus settings
        self.channel = self.declare_parameter('channel', 'PCAN_USBBUS1')
        self.bitrate = self.declare_parameter('bitrate', 1_000_000)

        # Motor parameters
        self.k_t = self.declare_parameter('k_t', 0.127)
        self.r_spool = self.declare_parameter('r_spool', 0.05)
        self.gear = self.declare_parameter('gear', 10.0)
        self.current_rated = self.declare_parameter('current_rated', 1.9)
        self.gear_mujoco = self.declare_parameter(
            'gear_mujoco', self.k_t * self.gear * self.current_rated / self.r_spool)

        # Control loop rate (CAN feedback loop is set to 100 Hz)
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)
        self.dry_run = self.declare_parameter('dry_run', True)
        self.cmd_u = None

        # Publisher for per-motor state
        self.state_pub = self.create_publisher(MOTOR_STATE)

        # Subscriber for tendon-force commands
        self.create_subscription(MOTOR_COMMAND, self._on_command)

        # Open the motor controller (servo/current mode)
        self.controller = None
        if not self.dry_run:
            # Imported lazily so dry runs don't need python-can installed
            from motor_control.ak_motor.motor_can import AKController
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

    def _on_command(self, msg):
        self.cmd_u = command_to_u(msg, self.motor_ids).tolist()

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

        stamp = self.now()
        for motor_id, fb in zip(self.motor_ids, feedbacks):
            if fb is not None:
                state = {
                    'position': fb['position'],
                    'velocity': fb['speed'],
                    'current': fb['current'],
                    'app_ctrl': fb['current'] * self.k_t * self.gear / (self.r_spool * self.gear_mujoco),
                    'fault_code': fb['error_code'],
                }
            else:
                state = {
                    'position': 0.0,
                    'velocity': 0.0,
                    'current': 0.0,
                    'app_ctrl': self.cmd_u[self.motor_ids.index(motor_id)],
                    'fault_code': 0,
                }
            state['stamp'] = stamp
            state['motor_id'] = motor_id
            self.state_pub.publish(state)

    def destroy_node(self):
        if self.controller is not None:
            try:
                self.controller.disable(self.controller.motors)
                self.controller.close()
            except Exception as exc:
                self.get_logger().warn(f'Error while shutting down motor controller: {exc}')
        super().destroy_node()


def main(argv=None):
    run_node(HardwareNode, argv)


if __name__ == '__main__':
    main()
