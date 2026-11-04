"""Low-current drive identification tests (launch modes sysid, real_sysid).

Waits until the markers are labelled (INITIAL_STATE), then runs `tests` in order, reading the motor
angles (MOTOR_STATE) every tick. Pulling is negative current; the motor angle decreases as it winds in.

  out <m>       pay-out breakaway: ramp motor m from the floor towards +out_max until it turns, then
                0 A and wind back to the start angle. The tendon goes slack, so this is the drive's
                own friction (less any pull of the robot on the tendon, small at rest).
  in <m>        wind-in breakaway: ramp towards in_max until it turns, then back to the floor at once
                (limits the slip). Drive friction plus the robot and the other tendons.
  cycles <m>    load/unload n_cycles times between the floor and cycle_margin past m's breakaway:
                does it ratchet further, does anything spring back.
  reset <m>     pay motor m out until it is back at its angle before `in`, so the next motor starts
                from near the rest pose.
  together      ramp all motors to together_max, hold, release: equal pulls barely bend the robot,
                so tendon wound in that the markers do not show as bending is stretch or compression.

Motion means the angle moved more than motion_threshold from where it was when the ramp started (a
held motor does not move at all; slips cross this within milliseconds, creep within seconds). Any motor moving more than max_travel within a test, or past the start
by more than max_total_travel, aborts the run (floor current, exit). Every command carries `phase`,
and detected breakaways carry an `event`; replay_hw lists them. Positive current needs the model
control range opened (launch.py sets actuator_ctrlrange for the sysid modes).
"""

import collections
import time

import numpy as np

from spirob_zmq.common import INITIAL_STATE, MOTOR_COMMAND, MOTOR_STATE
from spirob_zmq.core import Node, run_node


class Abort(Exception):
    pass


class CurrentProfileNode(Node):
    def __init__(self, params=None):
        super().__init__('current_profile_node', params)
        p = self.declare_parameter
        self.motor_ids = list(p('motor_ids', [10, 1, 2]))
        n = len(self.motor_ids)
        self.floor = float(p('floor_current', -0.1))
        self.in_max = float(p('in_max', -0.5))
        self.out_max = float(p('out_max', 0.5))
        self.ramp_rate = float(p('ramp_rate', 0.02))           # A/s while looking for breakaway
        self.fast_rate = float(p('fast_rate', 0.1))            # A/s for returns and releases
        self.motion_threshold = float(p('motion_threshold', 0.03))
        self.onset = float(p('onset_threshold', 0.005))        # rad: where a slip is taken to start
        self.max_travel = float(p('max_travel', 0.5))
        self.max_total_travel = float(p('max_total_travel', 1.5))
        self.hold_s = float(p('hold_s', 2.0))
        self.n_cycles = int(p('n_cycles', 3))
        self.cycle_margin = float(p('cycle_margin', 0.04))
        self.cycle_ramp_s = float(p('cycle_ramp_s', 2.0))
        self.return_tol = float(p('return_tol', 0.02))
        self.together_max = float(p('together_max', -0.45))
        self.lead_s = float(p('lead_s', 2.0))
        limit = float(p('current_limit', 1.0))
        if not (-limit <= self.in_max < self.floor <= 0.0 < self.out_max <= limit
                and -limit <= self.together_max < self.floor):
            raise ValueError('need -current_limit <= in_max, together_max < floor_current <= 0 < out_max '
                             '<= current_limit')
        default = [f'{t} {i}' for i in range(n) for t in ('out', 'in', 'cycles', 'reset')] + ['together']
        self.tests = [t.split() for t in p('tests', default)]
        for t in self.tests:
            if t[0] not in ('out', 'in', 'cycles', 'reset', 'together') or (t[0] != 'together' and
                                                                            not 0 <= int(t[1]) < n):
                raise ValueError(f'unknown test {" ".join(t)!r}')

        self.u = np.full(n, self.floor)
        self.angle = np.full(n, np.nan)
        self.history = [collections.deque() for _ in range(n)]   # (t, angle, u)
        self.breakaway = {}                                      # (motor index, 'in'|'out') -> A
        self.before_in = {}                                      # motor index -> angle before `in`
        self.start_angle = None
        self.start = None
        self.events = []
        self.event = None
        self.phase = 'wait'
        self.finished = False
        self.run = self._sequence()

        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)
        self.create_subscription(INITIAL_STATE, self._on_initial_state)
        self.create_subscription(MOTOR_STATE, self._on_motor_state)
        self.dt = 1.0 / p('rate_hz', 100.0)
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(f'motors {self.motor_ids}; tests: {", ".join(" ".join(t) for t in self.tests)}; '
                               f'{self.in_max} .. {self.out_max} A')

    # ---- inputs -------------------------------------------------------------------------------------
    def _on_initial_state(self, msg):
        if self.start is None:
            self.start = self.now()
            self.get_logger().info('markers labelled; starting the tests')

    def _on_motor_state(self, msg):
        if msg['motor_id'] in self.motor_ids:
            self.angle[self.motor_ids.index(msg['motor_id'])] = msg['position']

    def _record(self):
        t = self.now()
        for i, h in enumerate(self.history):
            h.append((t, self.angle[i], self.u[i]))
            while h and h[0][0] < t - 60.0:
                h.popleft()

    def moving(self, i, direction):
        """stop() for ramp: True once motor i has moved motion_threshold in `direction` (-1: winding in)."""
        a0 = self.angle[i]
        return lambda: direction * (self.angle[i] - a0) > self.motion_threshold

    def onset_current(self, i, direction, since):
        """Command at the first sample after `since` where motor i had moved onset_threshold."""
        h = [(s, a, u) for s, a, u in self.history[i] if s >= since]
        a0 = h[0][1]
        return next((u for s, a, u in h if direction * (a - a0) > self.onset), h[-1][2])

    # ---- building blocks (generators: one yield per tick) -------------------------------------------
    def hold(self, seconds):
        t_end = self.now() + seconds
        while self.now() < t_end:
            yield

    def ramp(self, idx, target, rate, stop=None):
        """Move motors idx to target at rate A/s; returns True if stop() fired first."""
        while True:
            if stop is not None and stop():
                return True
            step = rate * self.dt
            diff = target - self.u[idx]
            if np.all(np.abs(diff) <= step):
                self.u[idx] = target
                yield
                return False
            self.u[idx] += np.clip(diff, -step, step)
            yield

    def _event(self, text, **data):
        self.event = dict(data, text=text)
        self.events.append(text)
        self.get_logger().info(text)

    # ---- tests --------------------------------------------------------------------------------------
    def test_out(self, i):
        m = self.motor_ids[i]
        a0, t0 = self.angle[i], self.now()
        hit = yield from self.ramp([i], self.out_max, self.ramp_rate, self.moving(i, +1))
        self.u[i] = 0.0
        if not hit:
            self._event(f'motor {m}: no pay-out up to {self.out_max:+.2f} A')
            yield from self.ramp([i], self.floor, self.fast_rate)
            return
        b = self.onset_current(i, +1, t0)
        self.breakaway[(i, 'out')] = b
        yield from self.hold(1.0)
        self._event(f'motor {m}: pay-out breakaway at {b:+.3f} A, paid out {self.angle[i] - a0:+.3f} rad',
                    motor_id=m, kind='out', current=b, travel=float(self.angle[i] - a0))
        # Wind the paid-out tendon back in (slack: only the motor's friction resists)
        back = yield from self.ramp([i], self.in_max, self.fast_rate, lambda: self.angle[i] <= a0 + self.return_tol)
        self.u[i] = self.floor
        if not back:
            self._event(f'motor {m}: could not wind back to the start angle ({self.angle[i] - a0:+.3f} rad left)')
        yield from self.hold(self.hold_s)

    def test_in(self, i):
        m = self.motor_ids[i]
        self.before_in[i] = a0 = self.angle[i]
        t0 = self.now()
        hit = yield from self.ramp([i], self.in_max, self.ramp_rate, self.moving(i, -1))
        if not hit:
            self.u[i] = self.floor
            self._event(f'motor {m}: no wind-in up to {self.in_max:+.2f} A')
            yield from self.hold(self.hold_s)
            return
        b = self.onset_current(i, -1, t0)
        self.breakaway[(i, 'in')] = b
        self.u[i] = self.floor                      # stop pulling at once: limits the slip
        yield from self.hold(self.hold_s)
        self._event(f'motor {m}: wind-in breakaway at {b:+.3f} A, slipped {self.angle[i] - a0:+.3f} rad',
                    motor_id=m, kind='in', current=b, travel=float(self.angle[i] - a0))

    def test_cycles(self, i):
        m = self.motor_ids[i]
        if (i, 'in') not in self.breakaway:
            self._event(f'motor {m}: no wind-in breakaway, skipping cycles')
            return
        peak = max(self.breakaway[(i, 'in')] - self.cycle_margin, self.in_max)
        rate = abs(peak - self.floor) / self.cycle_ramp_s
        for k in range(self.n_cycles):
            a0 = self.angle[i]
            yield from self.ramp([i], peak, rate)
            yield from self.hold(self.hold_s / 2)
            a1 = self.angle[i]
            yield from self.ramp([i], self.floor, rate)
            yield from self.hold(self.hold_s / 2)
            self._event(f'motor {m}: cycle {k + 1} to {peak:+.3f} A: loading {a1 - a0:+.3f} rad, unloading '
                        f'{self.angle[i] - a1:+.3f} rad', motor_id=m, kind='cycle', current=float(peak),
                        loading=float(a1 - a0), unloading=float(self.angle[i] - a1))

    def test_reset(self, i):
        m = self.motor_ids[i]
        if i not in self.before_in or self.angle[i] >= self.before_in[i] - self.return_tol:
            return
        target = self.before_in[i]
        t0 = self.now()
        back = yield from self.ramp([i], self.out_max, self.ramp_rate * 2,
                                    lambda: self.angle[i] >= target - self.return_tol)
        b = self.onset_current(i, +1, t0)
        self.u[i] = self.floor
        yield from self.hold(self.hold_s)
        self._event(f'motor {m}: reset by paying out (started at {b:+.3f} A), {self.angle[i] - target:+.3f} rad '
                    f'from the angle before `in`' + ('' if back else ' (pay-out limit reached)'),
                    motor_id=m, kind='reset', current=b, residual=float(self.angle[i] - target))

    def test_together(self):
        idx = list(range(len(self.motor_ids)))
        a0 = self.angle.copy()
        yield from self.ramp(idx, self.together_max, self.ramp_rate * 1.5)
        yield from self.hold(self.hold_s * 2)
        a1 = self.angle.copy()
        yield from self.ramp(idx, self.floor, self.ramp_rate * 1.5)
        yield from self.hold(self.hold_s * 2)
        self._event(f'all motors to {self.together_max:+.2f} A: wound in {np.round(a1 - a0, 3)} rad, after release '
                    f'{np.round(self.angle - a0, 3)} rad', kind='together', current=self.together_max,
                    loaded=(a1 - a0).tolist(), released=(self.angle - a0).tolist())

    def _sequence(self):
        self.phase = 'lead'
        yield from self.hold(self.lead_s)
        for t in self.tests:
            self.phase = ' '.join(t)
            self.get_logger().info(f't={self.now() - self.start:6.1f} s  {self.phase}')
            self.test_start = self.angle.copy()
            if t[0] == 'together':
                yield from self.test_together()
            else:
                yield from getattr(self, f'test_{t[0]}')(int(t[1]))

    # ---- tick ---------------------------------------------------------------------------------------
    def _tick(self):
        if self.finished:
            self._send('done')
            return
        if self.start is None or np.isnan(self.angle).any():
            self._send('wait')
            return
        if self.start_angle is None:
            self.start_angle = self.angle.copy()
            self.test_start = self.angle.copy()
        self._record()
        try:
            travel = np.abs(self.angle - self.test_start)
            total = np.abs(self.angle - self.start_angle)
            if travel.max() > self.max_travel or total.max() > self.max_total_travel:
                raise Abort(f'motor travel {np.round(self.angle - self.test_start, 3)} rad in this test, '
                            f'{np.round(self.angle - self.start_angle, 3)} rad since the start')
            next(self.run)
        except StopIteration:
            self._finish('tests finished')
            return
        except Abort as exc:
            self._finish(f'ABORTED: {exc}')
            return
        np.clip(self.u, self.in_max, self.out_max, out=self.u)
        self._send(self.phase)

    def _finish(self, why):
        self.finished = True
        self.u[:] = self.floor
        self._send('done')
        self.get_logger().info(why)
        for text in self.events:
            self.get_logger().info(f'  {text}')
        self.shutdown()

    def _send(self, phase):
        msg = {'stamp': self.now(), 'motor_ids': self.motor_ids, 'u': self.u.copy(), 'phase': phase}
        if self.event is not None:
            msg['event'], self.event = self.event, None
        self.cmd_pub.publish(msg)

    def destroy_node(self):
        self.u[:] = self.floor
        self._send('done')
        time.sleep(0.05)  # let the last command go out before the socket closes
        super().destroy_node()


def main(argv=None):
    run_node(CurrentProfileNode, argv)


if __name__ == '__main__':
    main()
