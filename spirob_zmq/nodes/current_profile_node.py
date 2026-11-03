"""Scripted open-loop motor currents for model identification (launch modes sysid, real_sysid).

Waits until the markers are labelled (INITIAL_STATE), then for each group of motors in `groups`
steps the current up a staircase and back down, holding each level (static poses, loading and
unloading, so friction shows up as hysteresis), then runs one triangle ramp (dynamics), then rests.
Motors outside the group hold `floor_current`. Each command carries the profile `phase`, so
replay_hw can score the model per phase. Sends the floor current and exits when done.
"""

import time

import numpy as np

from spirob_zmq.common import INITIAL_STATE, MOTOR_COMMAND
from spirob_zmq.core import Node, run_node


def build_profile(n_motors, groups, floor, level_max, n_levels, hold_s, transition_s, ramp_s, rest_s, lead_s):
    """Piecewise-linear profile: knot times (s), currents (k, n_motors), and (start, phase name) list."""
    times, currents, phases = [0.0], [np.full(n_motors, floor)], [(0.0, 'lead')]

    def go(dt, u):
        times.append(times[-1] + dt)
        currents.append(np.array(u, dtype=float))

    go(lead_s, currents[-1])
    levels = floor + (level_max - floor) * np.arange(1, n_levels + 1) / n_levels
    for group in groups:
        name = '+'.join(str(i) for i in group)
        u = np.full(n_motors, floor)
        for direction, seq in (('up', levels), ('down', levels[-2::-1].tolist() + [floor])):
            phases.append((times[-1], f'stairs {name} {direction}'))
            for level in seq:
                u[list(group)] = level
                go(transition_s, u)
                go(hold_s, u)
        phases.append((times[-1], f'ramp {name}'))
        u[list(group)] = level_max
        go(ramp_s / 2, u)
        u[list(group)] = floor
        go(ramp_s / 2, u)
        phases.append((times[-1], f'rest {name}'))
        go(rest_s, u)
    return np.array(times), np.array(currents), phases


class CurrentProfileNode(Node):
    def __init__(self, params=None):
        super().__init__('current_profile_node', params)
        self.motor_ids = list(self.declare_parameter('motor_ids', [10, 1, 2]))
        n = len(self.motor_ids)
        # Pull is negative. hardware_node clips to the model ctrlrange (-1.9 .. -0.1 A), so the floor
        # is the least pull it will send.
        self.floor = float(self.declare_parameter('floor_current', -0.1))
        level_max = float(self.declare_parameter('level_max', -0.8))
        limit = float(self.declare_parameter('current_limit', 1.5))
        if not -limit <= level_max < self.floor <= 0.0:
            raise ValueError(f'need -current_limit <= level_max < floor_current <= 0, got {level_max}, {self.floor}')
        # Motor indices (into motor_ids) pulled together; default each motor alone, then 2 and 3
        groups = self.declare_parameter('groups', [[i] for i in range(n)] + ([[1, 2]] if n >= 3 else []))
        self.times, self.currents, self.phases = build_profile(
            n, groups, self.floor, level_max,
            n_levels=int(self.declare_parameter('n_levels', 4)),
            hold_s=float(self.declare_parameter('hold_s', 2.0)),
            transition_s=float(self.declare_parameter('transition_s', 0.3)),
            ramp_s=float(self.declare_parameter('ramp_s', 4.0)),
            rest_s=float(self.declare_parameter('rest_s', 3.0)),
            lead_s=float(self.declare_parameter('lead_s', 2.0)))
        self.wait_for_markers = self.declare_parameter('wait_for_markers', True)
        self.start = None
        self.phase = None
        self.done = False

        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)
        self.create_subscription(INITIAL_STATE, self._on_initial_state)
        self.timer = self.create_timer(1.0 / self.declare_parameter('rate_hz', 50.0), self._tick)
        self.get_logger().info(f'motors {self.motor_ids}, groups {groups}, {self.floor} .. {level_max} A, '
                               f'{self.times[-1]:.0f} s after the markers are labelled')

    def _on_initial_state(self, msg):
        if self.start is None:
            self.start = self.now()
            self.get_logger().info('markers labelled; starting the current profile')

    def _tick(self):
        if self.start is None and not self.wait_for_markers:
            self.start = self.now()
        if self.start is None or self.done:
            self._send(np.full(len(self.motor_ids), self.floor), 'wait' if not self.done else 'done')
            return
        t = self.now() - self.start
        if t >= self.times[-1]:
            self.done = True
            self.get_logger().info('current profile finished')
            self._send(np.full(len(self.motor_ids), self.floor), 'done')
            self.shutdown()
            return
        u = np.array([np.interp(t, self.times, self.currents[:, i]) for i in range(len(self.motor_ids))])
        phase = next(name for start, name in reversed(self.phases) if start <= t)
        if phase != self.phase:
            self.phase = phase
            self.get_logger().info(f't={t:5.1f} s  {phase}')
        self._send(u, phase)

    def _send(self, u, phase):
        self.cmd_pub.publish({'stamp': self.now(), 'motor_ids': self.motor_ids, 'u': u, 'phase': phase})

    def destroy_node(self):
        self._send(np.full(len(self.motor_ids), self.floor), 'done')
        time.sleep(0.05)  # let the last command go out before the socket closes
        super().destroy_node()


def main(argv=None):
    run_node(CurrentProfileNode, argv)


if __name__ == '__main__':
    main()
