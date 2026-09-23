"""Converts ctrl commands to AK motor currents and publishes motor feedback.

The conversion follows the MuJoCo actuator law, probed from the model
(common.ActuatorLaw), so the same code serves the vertical model's plain
``motor`` (ctrl = actuator force) and the horizontal model's ``dcmotor``
(ctrl = volts, with back-EMF and a force limit):

    actuator force  f   = clip(k_v u - k_e v_act, forcerange)
    tendon tension      = gear f
    motor torque    tau = gear f r_spool           (= f for the dcmotor, gear = 1/r_spool)
    current         I   = tau / (k_t gear_motor),  clipped to +-max_current

v_act, the actuator velocity, is gear * r_spool * shaft speed. For feedback,
``app_ctrl`` is the ctrl that would produce the measured current at the
measured speed, which is what the plant and the EKF consume.

!! Not yet verified on hardware: the feedback units (servo-mode position in
!! degrees, speed in electrical RPM, converted with pole_pairs), the sign
!! convention, and the torque constant. dry_run (the default) is unaffected.
"""

import math

import numpy as np

from spirob_zmq.common import MOTOR_COMMAND, MOTOR_STATE, ActuatorLaw, command_to_u, load_model
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
        self.k_t = self.declare_parameter('k_t', 0.127)             # N m / A, before the motor's gearbox
        self.gear = self.declare_parameter('gear', 10.0)            # motor's internal gearbox
        self.pole_pairs = self.declare_parameter('pole_pairs', 21)  # for ERPM -> rad/s
        self.r_spool = self.declare_parameter('r_spool', 0.05)
        self.current_rated = self.declare_parameter('current_rated', 1.9)
        self.max_current = self.declare_parameter('max_current', self.current_rated)

        # Actuator law and gear, from the same model the plant and EKF use
        model = load_model(self)
        self.law = ActuatorLaw(model)
        self.gear_model = model.actuator_gear[:, 0].copy()
        self.u_min = self.declare_parameter('u_min', -12.0)
        self.u_max = self.declare_parameter('u_max', 0.0)

        # Control loop rate (CAN feedback loop is set to 100 Hz)
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)
        self.dry_run = self.declare_parameter('dry_run', True)
        self.cmd_u = None

        # Publisher for per-motor state
        self.state_pub = self.create_publisher(MOTOR_STATE)

        # Subscriber for ctrl commands
        self.create_subscription(MOTOR_COMMAND, self._on_command)

        # Open the motor controller (servo/current mode)
        self.controller = None
        self.shaft_speed = np.zeros(len(self.motor_ids))   # rad/s, from the last feedback
        if not self.dry_run:
            # Imported lazily so dry runs don't need python-can installed
            from motor_control.ak_motor.motor_can import AKController
            f_max = np.max(np.abs(self.law.force(np.full(len(self.motor_ids), self.u_min), 0.0)))
            i_needed = f_max * np.max(self.gear_model) * self.r_spool / (self.k_t * self.gear)
            if i_needed > self.max_current:
                self.get_logger().warn(
                    f'model can ask for {i_needed:.1f} A but max_current is {self.max_current:.1f} A; '
                    'commands beyond that will be clipped')
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
        self.cmd_u = command_to_u(msg, self.motor_ids)

    def ctrl_to_current(self, u, shaft_speed):
        u = np.clip(u, self.u_min, self.u_max)
        act_vel = self.gear_model * self.r_spool * shaft_speed
        torque = self.gear_model * self.law.force(u, act_vel) * self.r_spool
        return np.clip(torque / (self.k_t * self.gear), -self.max_current, self.max_current)

    def current_to_ctrl(self, current, shaft_speed):
        act_vel = self.gear_model * self.r_spool * shaft_speed
        force = current * self.k_t * self.gear / (self.gear_model * self.r_spool)
        return (force + self.law.k_e * act_vel) / self.law.k_v

    def _on_timer(self):
        if self.cmd_u is None:
            return

        currents = self.ctrl_to_current(self.cmd_u, self.shaft_speed)
        if self.dry_run:
            feedbacks = [None] * len(self.motor_ids)
        else:
            self.controller.set_current(currents.tolist())
            motors_relaxed = [self.controller.motors[i] for i, current in enumerate(currents) if current == 0.0]
            self.controller.disable(motors_relaxed)
            feedbacks = self.controller.get_feedback_all_motors()

        stamp = self.now()
        for i, (motor_id, fb) in enumerate(zip(self.motor_ids, feedbacks)):
            if fb is not None:
                # AK servo-mode feedback: position in degrees, speed in electrical RPM
                speed = fb['speed'] / self.pole_pairs / self.gear * 2.0 * math.pi / 60.0
                self.shaft_speed[i] = speed
                state = {
                    'position': math.radians(fb['position']),
                    'velocity': speed,
                    'current': fb['current'],
                    'app_ctrl': float(self.current_to_ctrl(fb['current'], speed)[i]),
                    'fault_code': fb['error_code'],
                }
            else:
                state = {
                    'position': 0.0,
                    'velocity': 0.0,
                    'current': float(currents[i]),
                    'app_ctrl': float(self.cmd_u[i]),
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
