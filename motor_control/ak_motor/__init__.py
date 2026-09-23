"""
ak_motor
========
CubeMars AK Series Motor driver (Protocol V3.2.0).

Quick imports:

    from ak_motor import AKMotor, AKModGroup          # serial
    from ak_motor import AKMotorCAN, AKMotorMIT       # CAN individual
    from ak_motor import AKController                  # CAN group
    from ak_motor import AKFault, AKMode               # enums
    from ak_motor import MIT_MODEL_PARAMS              # model table
"""

from .protocol import (
    # Serial frame IDs
    COMM_GET_VALUES, COMM_SET_DUTY, COMM_SET_CURRENT, COMM_SET_CURRENT_BRAKE,
    COMM_SET_RPM, COMM_SET_POS, COMM_SET_HANDBRAKE, COMM_SET_DETECT,
    COMM_ROTOR_POSITION, COMM_GET_VALUES_SETUP, COMM_SET_POS_SPD,
    COMM_SET_POS_MULTI, COMM_SET_POS_SINGLE, COMM_SET_POS_ORIGIN, COMM_MIT,
    # Bitmask constants
    PARAM_BIT_MOS_TEMP, PARAM_BIT_MOTOR_TEMP, PARAM_BIT_OUTPUT_CUR,
    PARAM_BIT_INPUT_CUR, PARAM_BIT_DUTY, PARAM_BIT_RPM, PARAM_BIT_INPUT_V,
    PARAM_BIT_POSITION, PARAM_BIT_MOTOR_ID, PARAM_BIT_ERROR,
    # CAN packet IDs
    CAN_PKT_SET_DUTY, CAN_PKT_SET_CURRENT, CAN_PKT_SET_CURRENT_BRAKE,
    CAN_PKT_SET_RPM, CAN_PKT_SET_POS, CAN_PKT_SET_ORIGIN_HERE,
    CAN_PKT_SET_POS_SPD, CAN_PKT_SET_MIT, CAN_PKT_SET_DISABLE, CAN_PKT_SET_FRAME_CONFIG,
    # MIT model table and enums
    MIT_MODEL_PARAMS,
    AKMode, AKFault,
    # Encode / decode helpers
    build_packet, crc16,
    encode_duty, encode_current, encode_current_brake, encode_rpm,
    encode_position, encode_position_velocity, encode_origin,
    encode_get_values, encode_get_position, encode_get_params, encode_mit,
    encode_mit_can, decode_mit_can, decode_feedback,
    decode_state, decode_position,
    float_to_uint, uint_to_float,
)

from .motor_serial import AKMotor, AKModGroup
from .motor_can import AKMotorCAN, AKMotorMIT, AKController

__all__ = [
    # Serial classes
    "AKMotor", "AKModGroup",
    # CAN classes
    "AKMotorCAN", "AKMotorMIT", "AKController",
    # Enums
    "AKFault", "AKMode",
    # Model table
    "MIT_MODEL_PARAMS",
    # Serial frame IDs
    "COMM_GET_VALUES", "COMM_SET_DUTY", "COMM_SET_CURRENT",
    "COMM_SET_CURRENT_BRAKE", "COMM_SET_RPM", "COMM_SET_POS",
    "COMM_SET_HANDBRAKE", "COMM_SET_DETECT", "COMM_ROTOR_POSITION",
    "COMM_GET_VALUES_SETUP", "COMM_SET_POS_SPD", "COMM_SET_POS_MULTI",
    "COMM_SET_POS_SINGLE", "COMM_SET_POS_ORIGIN", "COMM_MIT",
    # Bitmasks
    "PARAM_BIT_MOS_TEMP", "PARAM_BIT_MOTOR_TEMP", "PARAM_BIT_OUTPUT_CUR",
    "PARAM_BIT_INPUT_CUR", "PARAM_BIT_DUTY", "PARAM_BIT_RPM",
    "PARAM_BIT_INPUT_V", "PARAM_BIT_POSITION", "PARAM_BIT_MOTOR_ID",
    "PARAM_BIT_ERROR",
    # CAN packet IDs
    "CAN_PKT_SET_DUTY", "CAN_PKT_SET_CURRENT", "CAN_PKT_SET_CURRENT_BRAKE",
    "CAN_PKT_SET_RPM", "CAN_PKT_SET_POS", "CAN_PKT_SET_ORIGIN_HERE",
    "CAN_PKT_SET_POS_SPD", "CAN_PKT_SET_MIT", "CAN_PKT_SET_DISABLE",
    "CAN_PKT_SET_FRAME_CONFIG",
    # Low-level helpers
    "build_packet", "crc16",
    "encode_duty", "encode_current", "encode_current_brake", "encode_rpm",
    "encode_position", "encode_position_velocity", "encode_origin",
    "encode_get_values", "encode_get_position", "encode_get_params",
    "encode_mit", "encode_mit_can",
    "decode_mit_can", "decode_feedback", "decode_state", "decode_position",
    "float_to_uint", "uint_to_float",
]
