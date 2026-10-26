"""Open-loop keyboard control: each key pair raises/lowers one tendon's motor current.

  q / a   motor 1 pull more / less
  w / s   motor 2 pull more / less
  e / d   motor 3 pull more / less
  + / -   bigger / smaller step
  space   release all (0 A)
  x, Esc  quit
"""

import sys
import time

import numpy as np

from spirob_zmq.common import MOTOR_COMMAND
from spirob_zmq.core import Node, run_node

KEYS = {'q': (0, 1), 'a': (0, -1), 'w': (1, 1), 's': (1, -1), 'e': (2, 1), 'd': (2, -1)}


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
        # Pull is negative current. Kept well under max_current by default; raise with -p current_limit=...
        self.current_limit = self.declare_parameter('current_limit', 0.5)
        self.step = self.declare_parameter('step', 0.05)
        self.u = np.full(len(self.motor_ids), float(self.declare_parameter('initial_current', 0.0)))

        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)
        self.keyboard = _Keyboard()
        self.timer = self.create_timer(1.0 / self.declare_parameter('rate_hz', 50.0), self._tick)
        print(__doc__, file=sys.stderr, flush=True)
        self.get_logger().info(f'motors {self.motor_ids}, step {self.step} A, limit {self.current_limit} A')

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
