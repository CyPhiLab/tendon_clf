#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ak_motor/motor.py
=================
AKMotor  – single motor over serial (Servo Mode + MIT Mode).
AKModGroup – group of serial motors (one port each).

Analogous to utils/Mod.py (Dynamixel) for the spirob 3-cable robot.
"""

import struct
import time
import serial

try:
    from .protocol import (
        COMM_GET_VALUES, COMM_ROTOR_POSITION, COMM_GET_VALUES_SETUP,
        COMM_MIT,
        build_packet, crc16,
        encode_duty, encode_current, encode_current_brake, encode_rpm,
        encode_position, encode_position_velocity, encode_origin,
        encode_get_values, encode_get_position, encode_get_params,
        encode_mit,
        decode_state, decode_position,
        AKFault,
    )
except ImportError:
    # Fallback for direct script execution from this folder (no package context)
    from protocol import (
        COMM_GET_VALUES, COMM_ROTOR_POSITION, COMM_GET_VALUES_SETUP,
        COMM_MIT,
        build_packet, crc16,
        encode_duty, encode_current, encode_current_brake, encode_rpm,
        encode_position, encode_position_velocity, encode_origin,
        encode_get_values, encode_get_position, encode_get_params,
        encode_mit,
        decode_state, decode_position,
        AKFault,
    )


class AKMotor:
    """
    Serial driver for one CubeMars AK Series motor (V3.2.0 protocol).
    Covers both Servo Mode and MIT Mode over UART via the R-LINK adapter.

    Typical usage:
        motor = AKMotor('/dev/ttyUSB0')
        motor.open()

        motor.set_origin()
        motor.set_position(90.0)            # degrees
        motor.set_current(3.0)              # Amps
        motor.set_rpm(2000)                 # ERPM

        # MIT via serial (V3.2 only):
        motor.send_mit_cmd(p_des=0.5, v_des=0.0, kp=100.0, kd=2.0, t_ff=0.0)

        state = motor.get_state()
        pos   = motor.get_position()        # degrees (float)
        motor.stop()
        motor.close()
    """

    DEFAULT_BAUD = 921600

    def __init__(self, port: str, baud: int = DEFAULT_BAUD,
                 read_timeout: float = 0.1):
        self.port         = port
        self.baud         = baud
        self.read_timeout = read_timeout
        self._serial: serial.Serial | None = None

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    def open(self):
        self._serial = serial.Serial(self.port, baudrate=self.baud,
                                     timeout=self.read_timeout)
        self._serial.reset_input_buffer()
        print(f"[STATUS] AKMotor connected on {self.port} @ {self.baud} baud")

    def close(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            print(f"[STATUS] AKMotor {self.port} closed")

    def __enter__(self): self.open(); return self
    def __exit__(self, *_): self.close()

    # -----------------------------------------------------------------------
    # Low-level framing
    # -----------------------------------------------------------------------

    def _send_raw(self, packet: bytes):
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial port is not open – call open() first.")
        self._serial.write(packet)

    def _send(self, frame_id: int, payload: bytes = b""):
        self._send_raw(build_packet(frame_id, payload))

    def _recv(self, expected_frame_id: int | None = None,
              timeout: float | None = None) -> bytes | None:
        """
        Read one response frame. Scans for 0xAA header, validates CRC-16
        and tail (0xBB). Returns payload bytes after frame_id, or None.
        """
        ser      = self._serial
        deadline = time.monotonic() + (timeout or self.read_timeout)

        while time.monotonic() < deadline:
            hdr = ser.read(1)
            if not hdr:
                continue
            if hdr[0] == 0xAA:
                break
        else:
            print("[WARN] AKMotor: no 0xAA header received")
            return None

        len_byte = ser.read(1)
        if not len_byte:
            print("[WARN] AKMotor: missing length byte")
            return None
        data_len = len_byte[0]

        body = ser.read(data_len + 3)
        if len(body) < data_len + 3:
            print(f"[WARN] AKMotor: short packet ({len(body)} bytes, need {data_len+3})")
            return None

        data     = body[:data_len]
        crc_recv = (body[data_len] << 8) | body[data_len + 1]
        tail     = body[data_len + 2]

        if tail != 0xBB:
            print(f"[WARN] AKMotor: bad tail 0x{tail:02X}")
            return None
        if crc16(data) != crc_recv:
            print("[WARN] AKMotor: CRC mismatch")
            return None

        resp_fid = data[0]
        payload  = bytes(data[1:])
        if expected_frame_id is not None and resp_fid != expected_frame_id:
            print(f"[WARN] AKMotor: unexpected frame_id 0x{resp_fid:02X}")
            return None
        return payload

    def _send_recv(self, frame_id: int, payload: bytes = b"",
                   resp_frame_id: int | None = None) -> bytes | None:
        self._send(frame_id, payload)
        return self._recv(expected_frame_id=resp_frame_id)

    # -----------------------------------------------------------------------
    # Read commands
    # -----------------------------------------------------------------------

    def get_state(self) -> dict | None:
        """Request all motor operating parameters (COMM_GET_VALUES)."""
        resp = self._send_recv(COMM_GET_VALUES, resp_frame_id=COMM_GET_VALUES)
        if resp is None:
            print("[WARN] get_state: no response")
            return None
        result = decode_state(resp)
        if result is None:
            print(f"[WARN] get_state: short response ({len(resp)} bytes, need ≥55)")
        return result

    def get_position(self) -> float | None:
        """One-shot rotor position request (COMM_ROTOR_POSITION). Returns degrees."""
        resp = self._send_recv(COMM_ROTOR_POSITION,
                               resp_frame_id=COMM_ROTOR_POSITION)
        if resp is None:
            print("[WARN] get_position: no response")
            return None
        result = decode_position(resp)
        if result is None:
            print("[WARN] get_position: short response")
        return result

    def get_params(self, bitmask: int) -> bytes | None:
        """Request selected motor parameters (COMM_GET_VALUES_SETUP)."""
        payload = struct.pack(">I", bitmask)
        return self._send_recv(COMM_GET_VALUES_SETUP, payload,
                               resp_frame_id=COMM_GET_VALUES_SETUP)

    # -----------------------------------------------------------------------
    # Servo Mode control commands
    # -----------------------------------------------------------------------

    def set_duty(self, duty: float):
        """Duty cycle mode. duty ∈ ±0.95."""
        self._send_raw(encode_duty(duty))

    def set_current(self, current_a: float):
        """Current (torque) mode. current_a ∈ ±60 A."""
        self._send_raw(encode_current(current_a))

    def set_current_brake(self, current_a: float):
        """Braking current mode. current_a ∈ 0–60 A."""
        self._send_raw(encode_current_brake(current_a))

    def set_rpm(self, rpm: float):
        """Velocity loop mode. rpm ∈ ±100 000 ERPM."""
        self._send_raw(encode_rpm(rpm))

    def set_position(self, degrees: float):
        """Position loop mode. degrees ∈ ±36 000°."""
        self._send_raw(encode_position(degrees))

    def set_position_velocity(self, degrees: float,
                              speed_erpm: int = 5000,
                              accel_erpm_s: int = 30000):
        """Position + velocity loop."""
        self._send_raw(encode_position_velocity(degrees, speed_erpm, accel_erpm_s))

    def set_origin(self, permanent: bool = False):
        """Zero the position encoder."""
        self._send_raw(encode_origin(permanent))
        print(f"[STATUS] AKMotor ({self.port}): origin set "
              f"({'permanent' if permanent else 'temporary'})")

    def stop(self):
        """Soft stop: zero current so motor freewheels."""
        self.set_current(0.0)

    def brake(self, current_a: float = 5.0):
        """Apply a holding brake current."""
        self.set_current_brake(current_a)

    def disable(self):
        """Disable motor output (zero current)."""
        self.set_current(0.0)
        print(f"[STATUS] AKMotor ({self.port}): disabled")

    # -----------------------------------------------------------------------
    # MIT Mode over serial (V3.2.0 NEW – COMM_MIT = 0x60)
    # -----------------------------------------------------------------------

    def send_mit_cmd(self, p_des: float, v_des: float,
                     kp: float, kd: float, t_ff: float):
        """
        MIT force-control command over serial.
        τ = t_ff + kp × (p_des − p) + kd × (v_des − v)
        """
        self._send_raw(encode_mit(p_des, v_des, kp, kd, t_ff))

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def print_state(self):
        state = self.get_state()
        if state is None:
            print(f"[ERROR] AKMotor ({self.port}): could not read state.")
            return
        print(f"---- AK Motor State ({self.port}) ----")
        print(f"  Motor ID             : {state['motor_id']}")
        print(f"  Position             : {state['outer_loop_position']:.4f} °")
        print(f"  Speed                : {state['rpm']} ERPM")
        print(f"  Duty cycle           : {state['duty']:.4f}")
        print(f"  Output current       : {state['output_current']:.3f} A")
        print(f"  Input current        : {state['input_current']:.3f} A")
        print(f"  Iq / Id current      : {state['iq_current']:.3f} A / {state['id_current']:.3f} A")
        print(f"  Input voltage        : {state['input_voltage']:.2f} V")
        print(f"  MOS temp             : {state['mos_temp']:.1f} °C")
        print(f"  Motor temp           : {state['motor_temp']:.1f} °C")
        print(f"  Vd / Vq              : {state['vd_voltage']:.3f} V / {state['vq_voltage']:.3f} V")
        print(f"  Control mode         : {state['control_mode']}")
        print(f"  Encoder angle        : {state['encoder_angle']:.4f}")
        print(f"  Outer encoder angle  : {state['outer_encoder_angle']:.4f}")
        print(f"  Error                : {state['error_name']}")
        print("--------------------------------------")


# ===========================================================================
# AKModGroup – group of serial motors (one port each)
# ===========================================================================

class AKModGroup:
    """
    Manages a group of CubeMars AK Series motors, one per serial port.
    Analogous to utils/Mod.py (Dynamixel) for the spirob 3-cable robot.

    Typical usage:
        mod = AKModGroup(['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2'])
        mod.open()
        mod.set_origin()
        mod.set_position([90.0, 0.0, -45.0])
        positions = mod.get_position()
        mod.stop()
        mod.close()
    """

    def __init__(self, ports: list[str], baud: int = AKMotor.DEFAULT_BAUD):
        self.motors: list[AKMotor] = [AKMotor(p, baud) for p in ports]
        print(f"[STATUS] AKModGroup: {len(self.motors)} motor(s) on {ports}")

    def open(self):
        for m in self.motors: m.open()

    def close(self):
        for m in self.motors: m.close()

    def __enter__(self): self.open(); return self
    def __exit__(self, *_): self.close()

    def get_state(self) -> list[dict | None]:
        return [m.get_state() for m in self.motors]

    def get_position(self) -> list[float | None]:
        return [m.get_position() for m in self.motors]

    def set_duty(self, duties: list[float]):
        for m, v in zip(self.motors, self._norm(duties)): m.set_duty(v)

    def set_current(self, currents: list[float]):
        for m, v in zip(self.motors, self._norm(currents)): m.set_current(v)

    def set_current_brake(self, currents: list[float]):
        for m, v in zip(self.motors, self._norm(currents)): m.set_current_brake(v)

    def set_rpm(self, rpms: list[float]):
        for m, v in zip(self.motors, self._norm(rpms)): m.set_rpm(v)

    def set_position(self, positions_deg: list[float]):
        for m, v in zip(self.motors, self._norm(positions_deg)): m.set_position(v)

    def set_position_velocity(self, positions_deg: list[float],
                              speeds_erpm: list[int],
                              accels_erpm_s: list[int]):
        for m, p, s, a in zip(self.motors,
                               self._norm(positions_deg),
                               self._norm(speeds_erpm),
                               self._norm(accels_erpm_s)):
            m.set_position_velocity(p, s, a)

    def set_origin(self, permanent: bool = False):
        for m in self.motors: m.set_origin(permanent)

    def send_mit_cmd(self,
                     p_des: list[float], v_des: list[float],
                     kp: list[float], kd: list[float], t_ff: list[float]):
        for m, p, v, k_p, k_d, t in zip(self.motors,
                                          self._norm(p_des),
                                          self._norm(v_des),
                                          self._norm(kp),
                                          self._norm(kd),
                                          self._norm(t_ff)):
            m.send_mit_cmd(p, v, k_p, k_d, t)

    def stop(self):
        for m in self.motors: m.stop()

    def brake(self, current_a: float = 5.0):
        for m in self.motors: m.brake(current_a)

    def print_state(self):
        for i, m in enumerate(self.motors):
            print(f"\n=== Motor {i} ===")
            m.print_state()

    def _norm(self, values: list) -> list:
        """Pad/trim list to match motor count."""
        if not values:
            raise ValueError("AKModGroup: empty values list")
        values = list(values)
        n = len(self.motors)
        if len(values) < n:
            values += [values[-1]] * (n - len(values))
        return values[:n]
