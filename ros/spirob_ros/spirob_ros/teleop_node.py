import math
import select
import sys
import termios
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from spirob_interfaces.msg import MotorCommand

HELP = """
SpiRob open-loop keyboard teleop  (publishes /spirob/motor_command)
-------------------------------------------------------------------
  w / s : bend toward +x / -x
  a / d : bend toward +y / -y
  r / f : increase / decrease pretension on all tendons (stiffens, no bend)
  space : release all tendons (u = 0)
  q     : release and quit   (Ctrl-C also works)

Commands are latched: each key press nudges the command by one step.
"""


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        # Motor ids, in the same order as the u vector
        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]).get_parameter_value().integer_array_value)

        # Control input bounds (negative u = pull on tendon), same convention as control_node
        self.u_min = self.declare_parameter('u_min', -1.0).get_parameter_value().double_value
        self.u_max = self.declare_parameter('u_max', 0.0).get_parameter_value().double_value

        # Increment per key press
        self.step = self.declare_parameter('step', 0.05).get_parameter_value().double_value

        # Direction (in the base xy-plane) the tip bends when each tendon is pulled.
        # Defaults measured from spirob_control.xml: tendon1 -> -x, tendon2 -> +x/-y, tendon3 -> +x/+y
        angles = self.declare_parameter('tendon_angles_deg', [180.0, -60.0, 60.0]).get_parameter_value().double_array_value
        self.tendon_dirs = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles])

        # Publish rate
        self.rate_hz = self.declare_parameter('rate_hz', 50.0).get_parameter_value().double_value

        # Latched command state
        self.bend = np.zeros(2)
        self.pull = 0.0

        self.cmd_pub = self.create_publisher(MotorCommand, '/spirob/motor_command', 10)

    def handle_key(self, key):
        max_pull = -self.u_min
        if key == 'w':
            self.bend[0] += self.step
        elif key == 's':
            self.bend[0] -= self.step
        elif key == 'a':
            self.bend[1] += self.step
        elif key == 'd':
            self.bend[1] -= self.step
        elif key == 'r':
            self.pull += self.step
        elif key == 'f':
            self.pull -= self.step
        elif key == ' ':
            self.bend[:] = 0.0
            self.pull = 0.0
        else:
            return False

        # Keep the latched state inside what the actuators can deliver
        norm = np.linalg.norm(self.bend)
        if norm > max_pull:
            self.bend *= max_pull / norm
        self.pull = float(np.clip(self.pull, 0.0, max_pull))
        return True

    def compute_u(self):
        # Each tendon pulls in proportion to how well it is aligned with the bend direction
        tension = self.pull + np.maximum(self.tendon_dirs @ self.bend, 0.0)
        return np.clip(-tension, self.u_min, self.u_max)

    def publish(self, u):
        cmd = MotorCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.motor_ids = self.motor_ids
        cmd.u = [float(x) for x in u]
        self.cmd_pub.publish(cmd)

    def release(self):
        self.bend[:] = 0.0
        self.pull = 0.0
        for _ in range(5):
            self.publish(np.zeros(len(self.motor_ids)))


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


def main(args=None):
    if not sys.stdin.isatty():
        print('teleop_node needs an interactive terminal: use `ros2 run spirob_ros teleop_node`, not ros2 launch.')
        return

    rclpy.init(args=args)
    node = TeleopNode()
    print(HELP)

    old_attrs = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        period = 1.0 / node.rate_hz
        while rclpy.ok():
            key = read_key(period)
            if key is not None:
                key = key.lower()
                if key == 'q':
                    break
                node.handle_key(key)

            u = node.compute_u()
            node.publish(u)
            rclpy.spin_once(node, timeout_sec=0.0)
            sys.stdout.write(
                f'\rbend=({node.bend[0]:+.2f}, {node.bend[1]:+.2f})  pull={node.pull:.2f}  '
                f'u=[{", ".join(f"{x:+.2f}" for x in u)}]   ')
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        print('\nReleasing tendons...')
        node.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
