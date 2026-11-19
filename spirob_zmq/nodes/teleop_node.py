"""Open-loop keyboard control: each key pair raises/lowers one tendon's motor current.

  q / a   motor 1 pull more / less
  w / s   motor 2 pull more / less
  e / d   motor 3 pull more / less
  + / -   bigger / smaller step
  space   release all (0 A)
  x, Esc  quit
"""

GAMEPAD_HELP = """Open-loop gamepad control (input=gamepad): the left stick bends the robot.

  left stick   bend direction and amount, as seen from behind the base (let go: release)
  LB / RB      strength down / up (fraction of current_limit at full stick)
  B (held)     release all
  Menu         quit          (x / Esc in this terminal also quit)
"""

import math
import os
import sys
import time

import numpy as np

from spirob_zmq.common import MOTOR_COMMAND
from spirob_zmq.core import Node, run_node

KEYS = {'q': (0, 1), 'a': (0, -1), 'w': (1, 1), 's': (1, -1), 'e': (2, 1), 'd': (2, -1)}


def allocate(stick, angles):
    """Non-negative tendon pulls w (each <= 1) whose directions sum to the stick vector.

    `angles` are the bend directions (rad) of the tendons pulled alone. The stick direction is split
    between the two tendons whose directions bracket it, so full stick between two tendons pulls both
    fully; with three tendons 120 degrees apart the third stays at zero.
    """
    w = np.zeros(len(angles))
    r = math.hypot(*stick)
    if r == 0.0:
        return w
    phi = math.atan2(stick[1], stick[0])
    order = np.argsort(np.mod(angles, 2 * math.pi))
    for k in range(len(order)):
        a, b = order[k], order[(k + 1) % len(order)]
        span = (angles[b] - angles[a]) % (2 * math.pi)
        if 0.0 < span < math.pi and (phi - angles[a]) % (2 * math.pi) <= span:
            basis = np.array([[math.cos(angles[a]), math.cos(angles[b])],
                              [math.sin(angles[a]), math.sin(angles[b])]])
            w[[a, b]] = np.maximum(np.linalg.solve(basis, stick), 0.0)
            return w / max(1.0, w.max())  # scale down rather than clip, so the direction is kept
    # No bracketing pair under 180 degrees (fewer than three tendons): project onto each direction
    return np.clip([math.cos(phi - t) * r for t in angles], 0.0, 1.0)


class _SdlPad:
    """First SDL game controller through pygame (Linux, Windows); polled, no window needed."""

    def __init__(self):
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        try:
            import pygame
            from pygame._sdl2 import controller
        except ImportError:
            sys.exit('input=gamepad needs pygame: pip install pygame')
        self._pg, self._sdl = pygame, controller
        pygame.joystick.init()
        controller.init()
        self._pad = None

    def connect(self):
        self._pg.joystick.quit()  # re-scan for controllers plugged in since the last try
        self._pg.joystick.init()
        for i in range(self._sdl.get_count()):
            if self._sdl.is_controller(i):
                self._pad = self._sdl.Controller(i)
                return self._sdl.name_forindex(i)
        return None

    def read(self):
        if self._pad is None or not self._pad.attached():
            self._pad = None
            return None
        pg = self._pg
        self._sdl.update()
        x = self._pad.get_axis(pg.CONTROLLER_AXIS_LEFTX) / 32767.0
        y = -self._pad.get_axis(pg.CONTROLLER_AXIS_LEFTY) / 32767.0  # SDL's y points down
        names = {'lb': pg.CONTROLLER_BUTTON_LEFTSHOULDER, 'rb': pg.CONTROLLER_BUTTON_RIGHTSHOULDER,
                 'b': pg.CONTROLLER_BUTTON_B, 'menu': pg.CONTROLLER_BUTTON_START}
        return x, y, {n for n, b in names.items() if self._pad.get_button(b)}


class _MacPad:
    """First controller through Apple's Game Controller framework (pip install pyobjc-framework-GameController).

    On macOS SDL (pygame) reads controllers through this framework, which only delivers input to the
    focused app: from a terminal the stick reads 0. Here background monitoring is switched on, and the
    main run loop is serviced on every read, since the framework updates its values from there.
    """

    def __init__(self):
        try:
            import GameController
            from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode
        except ImportError:
            sys.exit('input=gamepad on macOS needs pip install pyobjc-framework-GameController')
        self._gc = GameController.GCController
        self._gc.setShouldMonitorBackgroundEvents_(True)
        self._spin = lambda s: CFRunLoopRunInMode(kCFRunLoopDefaultMode, s, False)
        self._pad = None

    def connect(self):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:  # the framework finds connected controllers asynchronously
            self._spin(0.05)
            for c in self._gc.controllers():
                if c.extendedGamepad() is not None:
                    self._pad = c
                    return c.vendorName() or 'controller'
        return None

    def read(self):
        self._spin(0.0)
        if self._pad is None or self._pad not in self._gc.controllers():
            self._pad = None
            return None
        g = self._pad.extendedGamepad()
        stick = g.leftThumbstick()
        buttons = {'lb': g.leftShoulder(), 'rb': g.rightShoulder(), 'b': g.buttonB(), 'menu': g.buttonMenu()}
        return stick.xAxis().value(), stick.yAxis().value(), {n for n, b in buttons.items() if b.isPressed()}


class _Gamepad:
    """Left stick and a few buttons of the first game controller, with reconnects."""

    def __init__(self, logger):
        self._backend = _MacPad() if sys.platform == 'darwin' else _SdlPad()
        self._log = logger
        self._connected = False
        self._next_try = 0.0
        self._prev = set()
        self._connect()

    def _connect(self):
        self._next_try = time.monotonic() + 1.0
        name = self._backend.connect()
        self._connected = name is not None
        if self._connected:
            self._log.info(f'gamepad: {name}')
        else:
            self._log.warn('no gamepad found; holding all motors released, retrying every second')

    def read(self):
        """(stick xy in the unit disc, set of buttons newly pressed, set of buttons held), or None."""
        state = self._backend.read() if self._connected else None
        if state is None:
            if self._connected:
                self._log.warn('gamepad disconnected; releasing all motors')
                self._connected = False
            if time.monotonic() >= self._next_try:
                self._connect()
            return None
        x, y, held = state
        pressed, self._prev = held - self._prev, held
        return np.array([x, y]), pressed, held


class _Keyboard:
    """Non-blocking single-key reads from the terminal (cbreak mode, so Ctrl+C still works)."""

    def __init__(self):
        self._old = None
        if sys.platform == 'win32':
            import msvcrt
            self._msvcrt = msvcrt
        elif sys.stdin.isatty():
            import termios
            import tty
            self._termios = termios
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def keys(self):
        if sys.platform == 'win32':
            out = []
            while self._msvcrt.kbhit():
                out.append(self._msvcrt.getwch())
            return out
        import os
        import select
        out = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(sys.stdin.fileno(), 1).decode(errors='ignore')
            if not ch:
                break
            out.append(ch)
        return out

    def restore(self):
        if self._old is not None:
            self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old)
            self._old = None


class TeleopNode(Node):
    def __init__(self, params=None):
        super().__init__('teleop_node', params)
        self.motor_ids = list(self.declare_parameter('motor_ids', [10, 1, 2]))
        # Pull is negative current. Kept under max_current by default; change with -p current_limit=...
        self.current_limit = self.declare_parameter('current_limit', 1.5)
        self.step = self.declare_parameter('step', 0.05)
        self.u = np.full(len(self.motor_ids), float(self.declare_parameter('initial_current', 0.0)))

        self.input = self.declare_parameter('input', 'keyboard')
        if self.input not in ('keyboard', 'gamepad'):
            raise ValueError(f"input must be 'keyboard' or 'gamepad', got {self.input!r}")
        self.rate_hz = self.declare_parameter('rate_hz', 50.0)

        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)
        self.keyboard = _Keyboard()
        if self.input == 'gamepad':
            # Bend direction of each motor's tendon pulled alone, in the plane across the robot as seen
            # from behind the base (x right, y up). Measured in the model: tendon 1 (motor 10) bends up,
            # tendon 2 (motor 1) down-left, tendon 3 (motor 2) down-right.
            self.tendon_angles = np.deg2rad(self.declare_parameter('tendon_angles_deg', [90.0, 210.0, 330.0]))
            # Rotates the stick, for an operator who is not standing behind the base
            self.stick_rotation = math.radians(self.declare_parameter('stick_rotation_deg', 0.0))
            self.deadzone = self.declare_parameter('deadzone', 0.12)
            self.gain = self.declare_parameter('gain', 0.5)          # full stick = gain * current_limit
            self.slew_rate = self.declare_parameter('slew_rate', 5.0)  # A/s, so a flicked stick can't jerk
            self.gamepad = _Gamepad(self.get_logger())
            self._last_status = 0.0
            print(GAMEPAD_HELP, file=sys.stderr, flush=True)
            self.get_logger().info(f'motors {self.motor_ids}, limit {self.current_limit} A, '
                                   f'gain {self.gain:.1f}, slew {self.slew_rate} A/s')
            self.timer = self.create_timer(1.0 / self.rate_hz, self._tick_gamepad)
        else:
            print(__doc__, file=sys.stderr, flush=True)
            self.get_logger().info(f'motors {self.motor_ids}, step {self.step} A, limit {self.current_limit} A')
            self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _tick_gamepad(self):
        for ch in self.keyboard.keys():
            if ch.lower() in ('x', '\x1b'):
                self.shutdown()
                return
        target = np.zeros_like(self.u)
        state = self.gamepad.read()
        if state is not None:
            stick, pressed, held = state
            if 'menu' in pressed:
                self.shutdown()
                return
            if 'rb' in pressed:
                self.gain = min(round(self.gain + 0.1, 2), 1.0)
            if 'lb' in pressed:
                self.gain = max(round(self.gain - 0.1, 2), 0.1)
            r = min(float(np.hypot(*stick)), 1.0)
            if r > self.deadzone and 'b' not in held:
                c, s = math.cos(self.stick_rotation), math.sin(self.stick_rotation)
                direction = np.array([c * stick[0] - s * stick[1], s * stick[0] + c * stick[1]]) / np.hypot(*stick)
                amount = (r - self.deadzone) / (1.0 - self.deadzone)
                w = allocate(direction * amount, self.tendon_angles)
                target = -self.gain * self.current_limit * w[:len(self.u)]
        step = self.slew_rate / self.rate_hz
        self.u += np.clip(target - self.u, -step, step)
        np.clip(self.u, -self.current_limit, 0.0, out=self.u)
        now = time.monotonic()
        if now - self._last_status > 0.1:
            self._last_status = now
            print(f'\r  u [A]: ' + '  '.join(f'{v:+.3f}' for v in self.u)
                  + (f'   stick {state[0][0]:+.2f} {state[0][1]:+.2f}' if state is not None else '')
                  + f'   gain {self.gain:.1f} ({self.gain * self.current_limit:.2f} A max)'
                  + ('' if state is not None else '   NO GAMEPAD') + '   ',
                  end='', file=sys.stderr, flush=True)
        self._send()

    def _tick(self):
        changed = False
        for ch in self.keyboard.keys():
            ch = ch.lower()
            if ch in KEYS:
                i, sign = KEYS[ch]
                if i < len(self.u):
                    self.u[i] -= sign * self.step
            elif ch == ' ':
                self.u[:] = 0.0
            elif ch in '+=':
                self.step = min(self.step * 2.0, self.current_limit)
            elif ch in '-_':
                self.step = max(self.step / 2.0, 0.005)
            elif ch in ('x', '\x1b'):
                self.shutdown()
                return
            else:
                continue
            changed = True
        np.clip(self.u, -self.current_limit, 0.0, out=self.u)
        if changed:
            print(f'\r  u [A]: ' + '  '.join(f'{v:+.3f}' for v in self.u) + f'   step {self.step:.3f}   ',
                  end='', file=sys.stderr, flush=True)
        self._send()

    def _send(self):
        self.cmd_pub.publish({'stamp': self.now(), 'motor_ids': self.motor_ids, 'u': self.u})

    def destroy_node(self):
        self.keyboard.restore()
        self.u[:] = 0.0
        self._send()
        time.sleep(0.05)  # let the zero command go out before the socket closes
        print(file=sys.stderr)
        super().destroy_node()


def main(argv=None):
    run_node(TeleopNode, argv)


if __name__ == '__main__':
    main()
