"""Sends ctrl commands to the AK motors as currents and publishes motor feedback."""

import math
import threading

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

        model = load_model(self)
        self.law = ActuatorLaw(model)
        self.gear_model = model.actuator_gear[:, 0].copy()
        u_min, u_max = self.declare_parameter('u_min'), self.declare_parameter('u_max')
        self.u_min = self.law.u_min if u_min is None else np.full(model.nu, float(u_min))
        self.u_max = self.law.u_max if u_max is None else np.full(model.nu, float(u_max))

        # Feedback publish rate (CAN status frames are at 100 Hz)
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

        self._fb_lock = threading.Lock()
        self._latest_fb = [None] * len(self.motor_ids)
        self._fb_fresh = False
        self._reader = None
        if self.controller is not None:
            self._reader = threading.Thread(target=self._feedback_loop, daemon=True)
            self._reader.start()
            self.timer = self.create_timer(1.0 / self.rate_hz, self._publish_feedback)

    def _on_command(self, msg):
        self.cmd_u = command_to_u(msg, self.motor_ids)
        currents = self.ctrl_to_current(self.cmd_u, self.shaft_speed)
        if self.dry_run:
            applied = np.clip(self.cmd_u, self.u_min, self.u_max)
            self._publish_states([{
                'position': 0.0,
                'velocity': 0.0,
                'current': float(currents[i]),
                'app_ctrl': float(applied[i]),
                'fault_code': 0,
            } for i in range(len(self.motor_ids))])
            return
        self.controller.set_current(currents.tolist())
        motors_relaxed = [self.controller.motors[i] for i, current in enumerate(currents) if current == 0.0]
        self.controller.disable(motors_relaxed)

    def ctrl_to_current(self, u, shaft_speed):
        u = np.clip(u, self.u_min, self.u_max)
        act_vel = self.gear_model * self.r_spool * shaft_speed
        torque = self.gear_model * self.law.force(u, act_vel) * self.r_spool
        return np.clip(torque / (self.k_t * self.gear), -self.max_current, self.max_current)

    def current_to_ctrl(self, current, shaft_speed):
        act_vel = self.gear_model * self.r_spool * shaft_speed
        force = current * self.k_t * self.gear / (self.gear_model * self.r_spool)
        return (force + self.law.k_e * act_vel) / self.law.k_v

    def _feedback_loop(self):
        while self.ok():
            feedbacks = self.controller.get_feedback_all_motors(timeout=0.05)
            with self._fb_lock:
                for i, fb in enumerate(feedbacks):
                    if fb is not None:
                        self._latest_fb[i] = fb
                        self._fb_fresh = True

    def _publish_feedback(self):
        with self._fb_lock:
            if not self._fb_fresh:
                return
            feedbacks = list(self._latest_fb)
            self._fb_fresh = False
        states = []
        for i, fb in enumerate(feedbacks):
            if fb is None:
                states.append(None)
                continue
            # position in degrees, speed in ERPM
            speed = fb['speed'] / self.pole_pairs / self.gear * 2.0 * math.pi / 60.0
            self.shaft_speed[i] = speed
            states.append({
                'position': math.radians(fb['position']),
                'velocity': speed,
                'current': fb['current'],
                'app_ctrl': float(self.current_to_ctrl(fb['current'], speed)[i]),
                'fault_code': fb['error_code'],
            })
        self._publish_states(states)

    def _publish_states(self, states):
        stamp = self.now()
        for motor_id, state in zip(self.motor_ids, states):
            if state is None:
                continue
            state['stamp'] = stamp
            state['motor_id'] = motor_id
            self.state_pub.publish(state)

    def destroy_node(self):
        self.shutdown()
        if self._reader is not None:
            self._reader.join(timeout=0.2)
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
