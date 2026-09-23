#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ak_motor/protocol.py
====================
All protocol constants, enumerations, and pure encoding/decoding functions
for the CubeMars AK Series motor (Protocol V3.2.0).

Nothing in this file opens a serial port or CAN bus.
"""

import struct

# ===========================================================================
# Command packet IDs – Communication via R-link (serial UART)
# ===========================================================================
COMM_GET_VALUES        = 69   # 0x45  Get all motor operating parameters
COMM_GET_VALUES_SETUP  = 16   # 0x10  Get selected motor parameters (bitmask), e.g, bitmask = PARAM_BIT_RPM | PARAM_BIT_POSITION 
COMM_SET_DUTY          = 70   # 0x46  Duty cycle mode
COMM_SET_CURRENT       = 71   # 0x47  Current control mode
COMM_SET_CURRENT_BRAKE = 72   # 0x48  Current brake mode
COMM_SET_RPM           = 73   # 0x49  Velocity control mode (ERPM = shaft RPM×number of pole pairs)
COMM_SET_POS           = 74   # 0x4A  Position control mode (degrees), finer than COMM_SET_POS_MULTI
COMM_SET_POS_SPD       = 60   # 0x3C  Position + velocity control mode
COMM_SET_POS_MULTI     = 61   # 0x3D  Multi-turn position (±100 turns)
COMM_SET_POS_SINGLE    = 62   # 0x3E  Single-turn position (0–360°)
COMM_SET_HANDBRAKE     = 75   # 0x4B  Handbrake current mode
COMM_SET_DETECT        = 76   # 0x4C  Continuous position feedback mode (degrees)
COMM_ROTOR_POSITION    = 87   # 0x57  One-shot position feedback mode (degrees)
COMM_SET_POS_ORIGIN    = 64   # 0x40  Set home position 
COMM_MIT               = 96   # 0x60  MIT force-control (serial) – impedance control mode (§4.2)

# ===========================================================================
# CAN Servo Mode packet IDs  (§4.1, CAN_PACKET_ID enum)
# Extended frame CAN-ID = motor_id | (CAN_PACKET_ID << 8)
# ===========================================================================
CAN_PKT_SET_DUTY          = 0    # Duty cycle mode
CAN_PKT_SET_CURRENT       = 1    # Current control mode
CAN_PKT_SET_CURRENT_BRAKE = 2    # Current brake mode
CAN_PKT_SET_RPM           = 3    # Velocity control mode (ERPM)
CAN_PKT_SET_POS           = 4    # Position control mode (degrees)
CAN_PKT_SET_ORIGIN_HERE   = 5    # Set origin
CAN_PKT_SET_POS_SPD       = 6    # Position + velocity control mode
CAN_PKT_SET_MIT               = 8    # MIT (force-control) mode
CAN_PKT_SET_DISABLE           = 15   # Motor disable  (NEW in V3.2)
CAN_PKT_SET_FRAME_CONFIG      = 16   # Feedback message configuration (NEW in V3.2)

# ---------------------------------------------------------------------------
# COMM_GET_VALUES_SETUP bitmask (§4.3.2.1 table)
# ---------------------------------------------------------------------------
PARAM_BIT_MOS_TEMP    = 1 << 0    # MOS temperature (2 bytes)
PARAM_BIT_MOTOR_TEMP  = 1 << 1    # Motor temperature (2 bytes)
PARAM_BIT_OUTPUT_CUR  = 1 << 2    # Output current (4 bytes)
PARAM_BIT_INPUT_CUR   = 1 << 3    # Input current (4 bytes)
PARAM_BIT_DUTY        = 1 << 6    # Duty cycle (2 bytes)
PARAM_BIT_RPM         = 1 << 7    # RPM (4 bytes)
PARAM_BIT_INPUT_V     = 1 << 8    # Input voltage (2 bytes)
PARAM_BIT_POSITION    = 1 << 15   # Motor outer-loop position (4 bytes)
PARAM_BIT_MOTOR_ID    = 1 << 16   # Motor ID (1 byte)
PARAM_BIT_ERROR       = 1 << 17   # Error status (1 byte)

# ===========================================================================
# MIT mode parameter ranges (§4.2, parameter table)
# Default = AK10-9; override per model in constructor.
# Position PD controller - gain scheduling
# ===========================================================================
_MIT_P_MIN  = -12.56; _MIT_P_MAX  = 12.56   # rad  (all models)
_MIT_V_MIN  = -28.0;  _MIT_V_MAX  = 28.0    # rad/s (AK10-9 default)
_MIT_T_MIN  = -54.0;  _MIT_T_MAX  = 54.0    # N·m  (AK10-9 default)
_MIT_KP_MIN = 0.0;    _MIT_KP_MAX = 500.0
_MIT_KD_MIN = 0.0;    _MIT_KD_MAX = 5.0

# Convenience dict: motor_model → (Kt, v_max, t_max)
MIT_MODEL_PARAMS = {
    "AK10-9":  (1.3137, 28.0,  54.0),
    "AK60-6":  (0.5994, 60.0,  12.0),
    "AK60-39": (3.4616, 10.0,  80.0),
    "AK70-9":  (1.0621, 30.0,  32.0),
    "AK80-8":  (1.0569, 38.0,  32.0),
    "AK80-9":  (0.5701, 65.0,  18.0),
    "AKE60-8": (0.7382, 40.0,  15.0),
    "AKE80-8": (1.7909, 20.0,  35.0),
    "AKE90-8": (1.8265, 20.0, 150.0),
    "AKH70-16":(2.8334, 13.0, 110.0),
    "AKH70-48":(8.8123,  5.0, 280.0),
}

# ===========================================================================
# AKMode – control mode identifiers returned in get_state()
# ===========================================================================
class AKMode:
    DUTY     = 0
    CURRENT  = 1
    BRAKE    = 2
    RPM      = 3
    POSITION = 4
    POS_SPD  = 5
    MIT      = 6

# ===========================================================================
# AKFault – fault codes (§4.3.2.1)
# ===========================================================================
class AKFault:
    NONE                = 0
    OVER_VOLTAGE        = 1
    UNDER_VOLTAGE       = 2
    DRV                 = 3
    ABS_OVER_CURRENT    = 4
    OVER_TEMP_FET       = 5
    OVER_TEMP_MOTOR     = 6
    GATE_DRIVER_OV      = 7
    GATE_DRIVER_UV      = 8
    MCU_UNDER_VOLTAGE   = 9
    BOOTING_FROM_WDT    = 10
    ENCODER_SPI         = 11
    ENCODER_SINCOS_LOW  = 12
    ENCODER_SINCOS_HIGH = 13
    FLASH_CORRUPTION    = 14
    CURRENT_SENSOR_1    = 15
    CURRENT_SENSOR_2    = 16
    CURRENT_SENSOR_3    = 17
    UNBALANCED_CURRENTS = 18

    NAMES = {
        0:  "None",
        1:  "Over-voltage",
        2:  "Under-voltage",
        3:  "Driver fault",
        4:  "Motor over-current",
        5:  "MOS over-temperature",
        6:  "Motor over-temperature",
        7:  "Gate driver over-voltage",
        8:  "Gate driver under-voltage",
        9:  "MCU under-voltage",
        10: "Watchdog reset",
        11: "SPI encoder fault",
        12: "Encoder sincos below min amplitude",
        13: "Encoder sincos above max amplitude",
        14: "Flash corruption",
        15: "Current sampling channel 1 fault",
        16: "Current sampling channel 2 fault",
        17: "Current sampling channel 3 fault",
        18: "Unbalanced currents",
    }

    @classmethod
    def name(class_name, code: int) -> str:
        return class_name.NAMES.get(code, f"Unknown({code})")


# ---------------------------------------------------------------------------
# CAN Servo feedback error codes (§4.3.1)
# These differ from the serial mc_fault_code enum used in AKFault above.
# ---------------------------------------------------------------------------
_CAN_SERVO_FAULT_NAMES = {
    0: "None",
    1: "Motor over-temperature",
    2: "Over-current",
    3: "Over-voltage",
    4: "Under-voltage",
    5: "Encoder fault",
    6: "MOSFET over-temperature",
    7: "Motor lock-up",
}


def can_servo_fault_name(code: int) -> str:
    """Human-readable name for a CAN Servo feedback error code (§4.3.1)."""
    return _CAN_SERVO_FAULT_NAMES.get(code, f"Unknown({code})")


# ===========================================================================
# CRC-16 / CCITT  (§4.3.2.3)
# ===========================================================================
_CRC16_TAB = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
]


def crc16(data: bytes) -> int:
    """CRC-16/CCITT over a byte sequence (§4.3.2.3)."""
    cksum = 0
    for b in data:
        cksum = _CRC16_TAB[((cksum >> 8) ^ b) & 0xFF] ^ (cksum << 8)
        cksum &= 0xFFFF
    return cksum


# ===========================================================================
# Serial frame encoding
# ===========================================================================

def build_packet(frame_id: int, payload: bytes = b"") -> bytes:
    """
    Build a complete serial TX frame (V3.2.0 format).

    Layout (short, len ≤ 255):
        0xAA | len | frame_id | payload | CRC_hi | CRC_lo | 0xBB
    Layout (long, len > 255):
        0xAB | len_hi | len_lo | frame_id | payload | CRC_hi | CRC_lo | 0xBB

    len = 1 (frame_id) + len(payload).
    """
    data   = bytes([frame_id]) + payload
    length = len(data)
    crc    = crc16(data)
    suffix = bytes([crc >> 8, crc & 0xFF, 0xBB])
    if length <= 255:
        return bytes([0xAA, length]) + data + suffix
    else:
        return bytes([0xAB, length >> 8, length & 0xFF]) + data + suffix


# Convenience encode_* helpers (one per command)

def encode_duty(duty: float) -> bytes:
    return build_packet(COMM_SET_DUTY, struct.pack(">i", int(duty * 100_000.0)))

def encode_current(current_a: float) -> bytes:
    return build_packet(COMM_SET_CURRENT, struct.pack(">i", int(current_a * 1000.0)))

def encode_current_brake(current_a: float) -> bytes:
    return build_packet(COMM_SET_CURRENT_BRAKE, struct.pack(">i", int(current_a * 1000.0)))

def encode_rpm(rpm: float) -> bytes:
    return build_packet(COMM_SET_RPM, struct.pack(">i", int(rpm)))

def encode_position(degrees: float) -> bytes:
    return build_packet(COMM_SET_POS, struct.pack(">i", int(degrees * 1_000_000.0)))
  # @staticmethod
    # def _bc(val, n: int) -> list:
    #     if isinstance(val, (int, float)):
    #         return [val] * n
    #     val = list(val)
    #     if len(val) < n:
    #         val += [val[-1]] * (n - len(val))
    #     return val[:n]
def encode_position_velocity(degrees: float,
                              speed_erpm: int = 5000,
                              accel_erpm_s: int = 30000) -> bytes:
    # Serial §4.3.2.2 item 7: pos=int32×1000, speed=int32 raw ERPM, accel=int32 raw ERPM/s
    payload = (struct.pack(">i", int(degrees * 1000.0))
               + struct.pack(">i", int(speed_erpm))
               + struct.pack(">i", int(accel_erpm_s)))
    return build_packet(COMM_SET_POS_SPD, payload)

def encode_origin(permanent: bool = False) -> bytes:
    return build_packet(COMM_SET_POS_ORIGIN, bytes([1 if permanent else 0]))

def encode_get_values() -> bytes:
    return build_packet(COMM_GET_VALUES)

def encode_get_position() -> bytes:
    return build_packet(COMM_ROTOR_POSITION)

def encode_get_params(bitmask: int) -> bytes:
    return build_packet(COMM_GET_VALUES_SETUP, struct.pack(">I", bitmask))

# ===========================================================================
# CAN Servo Mode encoding  (§4.1)
# Each function returns the 4-byte data payload only.
# CAN-ID must be built by the caller: motor_id | (CAN_SET_* << 8)
# ===========================================================================

def encode_can_duty(duty: float) -> bytes:
    """CAN duty cycle payload. duty ∈ [-1.0, 1.0]."""
    return struct.pack(">i", int(duty * 100_000.0))

def encode_can_current(current_a: float) -> bytes:
    """CAN current payload (A → mA as int32)."""
    return struct.pack(">i", int(current_a * 1000.0))

def encode_can_current_brake(current_a: float) -> bytes:
    """CAN current brake payload (A → mA as int32)."""
    return struct.pack(">i", int(current_a * 1000.0))

def encode_can_rpm(rpm: float) -> bytes:
    """CAN velocity payload (raw ERPM as int32)."""
    return struct.pack(">i", int(rpm))

def encode_can_pos(degrees: float) -> bytes:
    """CAN position payload (degrees × 10,000 as int32). Range ±36000°."""
    return struct.pack(">i", int(degrees * 10000.0))

def encode_can_pos_spd(degrees: float,
                       speed_erpm: int = 5000,
                       accel_erpm_s: int = 30000) -> bytes:
    """
    CAN position+velocity payload (§4.1.7).
    pos   = int32 × 10000 (degrees), range ±36000°
    speed = int16, value = ERPM / 10,  range ±327680 ERPM
    accel = int16, value = ERPM/s / 10, range 0–327670 ERPM/s
    """
    return (struct.pack(">i", int(degrees * 10000.0))
            + struct.pack(">h", int(speed_erpm / 10.0))
            + struct.pack(">h", int(accel_erpm_s / 10.0)))


def encode_mit(p_des: float, v_des: float,
               kp: float, kd: float, t_ff: float) -> bytes:
    """MIT serial command (COMM_MIT = 0x60). All values ×1000 as int32."""
    payload = (struct.pack(">i", int(p_des * 1000.0))
               + struct.pack(">i", int(v_des * 1000.0))
               + struct.pack(">i", int(t_ff  * 1000.0))
               + struct.pack(">i", int(kp    * 1000.0))
               + struct.pack(">i", int(kd    * 1000.0)))
    return build_packet(COMM_MIT, payload)


# ===========================================================================
# Serial frame decoding
# ===========================================================================

def decode_state(resp: bytes) -> dict | None:
    """
    Decode a COMM_GET_VALUES response payload into a dict.
    Returns None if the payload is too short.
    """
    if len(resp) < 84:   # full field parse consumes exactly 84 bytes
        return None

    idx = 0

    def gi16():
        nonlocal idx
        v = struct.unpack_from(">h", resp, idx)[0]; idx += 2; return v

    def gi32():
        nonlocal idx
        v = struct.unpack_from(">i", resp, idx)[0]; idx += 4; return v

    def gf32():
        nonlocal idx
        v = struct.unpack_from(">f", resp, idx)[0]; idx += 4; return v

    def gu8():
        nonlocal idx
        v = resp[idx]; idx += 1; return v

    mos_temp       = gi16() / 10.0
    motor_temp     = gi16() / 10.0
    output_current = gi32() / 100.0
    input_current  = gi32() / 100.0
    id_current     = gi32() / 100.0
    iq_current     = gi32() / 100.0
    duty           = gi16() / 1000.0
    rpm            = gi32()
    input_voltage  = gi16() / 10.0
    idx += 24
    error_code     = gu8()
    outer_loop_pos = gf32()
    motor_id       = gu8()
    idx += 6
    vd_voltage     = gi32() / 1000.0
    vq_voltage     = gi32() / 1000.0
    control_mode   = gi32()
    encoder_angle  = gf32()
    outer_enc_angle= gf32()

    return {
        "mos_temp":             mos_temp,
        "motor_temp":           motor_temp,
        "output_current":       output_current,
        "input_current":        input_current,
        "id_current":           id_current,
        "iq_current":           iq_current,
        "duty":                 duty,
        "rpm":                  rpm,
        "input_voltage":        input_voltage,
        "outer_loop_position":  outer_loop_pos,
        "motor_id":             motor_id,
        "error_code":           error_code,
        "error_name":           AKFault.name(error_code),
        "vd_voltage":           vd_voltage,
        "vq_voltage":           vq_voltage,
        "control_mode":         control_mode,
        "encoder_angle":        encoder_angle,
        "outer_encoder_angle":  outer_enc_angle,
    }


def decode_position(resp: bytes) -> float | None:
    """Decode a COMM_ROTOR_POSITION response payload → degrees."""
    if len(resp) < 4:
        return None
    return struct.unpack_from(">f", resp, 0)[0]


# ===========================================================================
# MIT helpers (shared by serial and CAN)
# ===========================================================================

def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Map float x ∈ [x_min, x_max] → unsigned int over `bits` bits (§4.2)."""
    span = x_max - x_min
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * float(1 << bits) / span)


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """Inverse of float_to_uint (§4.2)."""
    return x_int * (x_max - x_min) / ((1 << bits) - 1) + x_min


def encode_mit_can(p_des: float, v_des: float,
                   kp: float, kd: float, t_ff: float,
                   p_min: float = _MIT_P_MIN, p_max: float = _MIT_P_MAX,
                   v_min: float = _MIT_V_MIN, v_max: float = _MIT_V_MAX,
                   t_min: float = _MIT_T_MIN, t_max: float = _MIT_T_MAX) -> bytes:
    """
    Pack a MIT command into 8 CAN data bytes (§4.2 V3.2.0).

    V3.2.0 layout (kp, kd, pos, vel, torque):
      buffer[0]  kp_int[11:4]
      buffer[1]  kp_int[3:0] | kd_int[11:8]
      buffer[2]  kd_int[7:0]
      buffer[3]  p_int[15:8]
      buffer[4]  p_int[7:0]
      buffer[5]  v_int[11:4]
      buffer[6]  v_int[3:0] | t_int[11:8]
      buffer[7]  t_int[7:0]
    """
    p_int  = float_to_uint(p_des, p_min,      p_max,      16)
    v_int  = float_to_uint(v_des, v_min,      v_max,      12)
    kp_int = float_to_uint(kp,    _MIT_KP_MIN, _MIT_KP_MAX, 12)
    kd_int = float_to_uint(kd,    _MIT_KD_MIN, _MIT_KD_MAX, 12)
    t_int  = float_to_uint(t_ff,  t_min,      t_max,      12)

    data = bytearray(8)
    data[0] =  kp_int >> 4
    data[1] = ((kp_int & 0xF) << 4) | (kd_int >> 8)
    data[2] =  kd_int & 0xFF
    data[3] =  p_int  >> 8
    data[4] =  p_int  & 0xFF
    data[5] =  v_int  >> 4
    data[6] = ((v_int  & 0xF) << 4) | (t_int >> 8)
    data[7] =  t_int  & 0xFF
    return bytes(data)


def decode_mit_can(data: bytes,
                   p_min: float = _MIT_P_MIN, p_max: float = _MIT_P_MAX,
                   v_min: float = _MIT_V_MIN, v_max: float = _MIT_V_MAX,
                   t_min: float = _MIT_T_MIN, t_max: float = _MIT_T_MAX) -> dict:
    """
    Parse the 8-byte MIT CAN reply frame from the motor (§4.2).

    Reply layout:
      DATA[0]      Drive ID
      DATA[1:2]    p_int (16-bit position)
      DATA[3:4]    v_int (12-bit velocity, upper nibble of DATA[4])
      DATA[4:5]    t_int (12-bit torque, lower nibble of DATA[4])
      DATA[6]      temperature raw (−40 offset → °C)
      DATA[7]      error code
    """
    motor_id = data[0]
    p_int    = (data[1] << 8) | data[2]
    v_int    = (data[3] << 4) | (data[4] >> 4)
    t_int    = ((data[4] & 0xF) << 8) | data[5]
    temp_raw = data[6]
    error    = data[7]

    return {
        "motor_id":    motor_id,
        "position":    uint_to_float(p_int, p_min, p_max, 16),
        "velocity":    uint_to_float(v_int, v_min, v_max, 12),
        "torque":      uint_to_float(t_int, t_min, t_max, 12),
        "temperature": temp_raw - 40,
        "error_code":  error,
        "error_name":  AKFault.name(error),
    }


# Alias kept for backwards compatibility with old src.py callers
decode_feedback = decode_mit_can
