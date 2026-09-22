#!/usr/bin/env python3
"""
plot_node
=========
Subscribes to hardware/estimation topics and live-plots states over time,
mirroring the offline behaviour of plot.py but in real time.

Subscribes:
    TODO: /spirob/motor_state  (spirob_interfaces/MotorState)
    TODO: /spirob/robot_state  (spirob_interfaces/RobotState)
"""

import rclpy
from rclpy.node import Node

# TODO: import matplotlib and your message types

# TODO: import your message types, e.g.
# from spirob_interfaces.msg import MotorState, RobotState


class PlotNode(Node):
    def __init__(self):
        super().__init__('plot_node')

        # TODO: declare_parameter for window_s, redraw_hz, save_csv, output_dir

        # TODO: set up rolling buffers (e.g. collections.deque) per topic/field

        # TODO: create subscriptions for MotorState and RobotState

        # TODO: create a separate timer (NOT the subscription callback) for redrawing
        pass

    # TODO: subscription callbacks: append incoming data into buffers

    # TODO: timer callback: clear + redraw matplotlib axes from buffers

    # TODO: override destroy_node() to optionally dump buffers to CSV


def main(args=None):
    rclpy.init(args=args)
    node = PlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
