"""Helpers shared by several nodes."""

from pathlib import Path

import mujoco
import numpy as np

from spirob_zmq.core import DEFAULT_MODEL_PATH

# Topic names, identical to the ROS package.
MOTOR_COMMAND = '/spirob/motor_command'
MOTOR_STATE = '/spirob/motor_state'
ROBOT_STATE = '/spirob/robot_state'
SITE_MEASUREMENT = '/spirob/site_measurement'


def load_model(node):
    model_path = node.declare_parameter('model_path', DEFAULT_MODEL_PATH)
    if not Path(model_path).exists():
        node.get_logger().error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return mujoco.MjModel.from_xml_path(model_path)


def command_to_u(msg, motor_ids):
    """Reorder a MotorCommand's ``u`` to match ``motor_ids`` (missing ids -> 0)."""
    if list(msg['motor_ids']) == list(motor_ids):
        return np.asarray(msg['u'], dtype=float).copy()
    id_to_u = dict(zip(msg['motor_ids'], msg['u']))
    return np.array([id_to_u.get(motor_id, 0.0) for motor_id in motor_ids], dtype=float)
