#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ak_motor/motor_can.py
=====================
AKMotorCAN   – Servo Mode over CAN bus (one motor).
AKMotorMIT   – MIT Force-Control Mode over CAN bus (one motor).
AKController – Unified CAN bus wrapper for a group of motors (Servo or MIT).

All three share a single python-can Bus instance when used via AKController.
"""

import struct
import time

try:
    from .protocol import (
        CAN_PKT_SET_DUTY,
        CAN_PKT_SET_CURRENT,
        CAN_PKT_SET_CURRENT_BRAKE,
        CAN_PKT_SET_RPM,
        CAN_PKT_SET_POS,
        CAN_PKT_SET_ORIGIN_HERE,
        CAN_PKT_SET_POS_SPD,
        CAN_PKT_SET_MIT,
        CAN_PKT_SET_DISABLE,
        CAN_PKT_SET_FRAME_CONFIG,
        MIT_MODEL_PARAMS,
        _MIT_P_MIN, _MIT_P_MAX, _MIT_V_MIN, _MIT_V_MAX, _MIT_T_MIN, _MIT_T_MAX,
        encode_can_duty, encode_can_current, encode_can_current_brake,
        encode_can_rpm, encode_can_pos, encode_can_pos_spd,
        encode_mit_can, decode_mit_can, decode_feedback,
        AKFault, can_servo_fault_name,
    )
except ImportError:
    # Fallback for direct script execution from this folder (no package context)
    from protocol import (
        CAN_PKT_SET_DUTY,
        CAN_PKT_SET_CURRENT,
        CAN_PKT_SET_CURRENT_BRAKE,
        CAN_PKT_SET_RPM,
        CAN_PKT_SET_POS,
        CAN_PKT_SET_ORIGIN_HERE,
        CAN_PKT_SET_POS_SPD,
        CAN_PKT_SET_MIT,
        CAN_PKT_SET_DISABLE,
        CAN_PKT_SET_FRAME_CONFIG,
        MIT_MODEL_PARAMS,
        _MIT_P_MIN, _MIT_P_MAX, _MIT_V_MIN, _MIT_V_MAX, _MIT_T_MIN, _MIT_T_MAX,
        encode_can_duty, encode_can_current, encode_can_current_brake,
        encode_can_rpm, encode_can_pos, encode_can_pos_spd,
        encode_mit_can, decode_mit_can, decode_feedback,
        AKFault, can_servo_fault_name,
    )

try:
    import can
    _CAN_AVAILABLE = True
except ImportError:
    _CAN_AVAILABLE = False


def _require_can():
    if not _CAN_AVAILABLE:
        raise ImportError("python-can is not installed. Run: pip install python-can")


# ===========================================================================
# AKMotorCAN – Servo Mode over CAN bus
# ===========================================================================

class AKMotorCAN:
    """
    CAN bus driver for one CubeMars AK Series motor (Servo Mode, V3.2.0).

    CAN frame type: Extended (29-bit).
    CAN-ID encoding: motor_id | (CAN_PACKET_ID << 8).
    Bus rate: 1 Mbit/s.

    Typical usage:
        motor = AKMotorCAN(motor_id=104)
        motor.open()
        motor.set_origin()
        motor.set_position(90.0)
        feedback = motor.receive_feedback()
        motor.disable()
        motor.close()
    """

    def __init__(self, motor_id: int,
                 interface: str = 'pcan',
                 channel: str = 'PCAN_USBBUS1',
                 bitrate: int = 1_000_000,
                 receive_timeout: float = 0.05,
                 bus: "can.BusABC | None" = None):
        _require_can()
        self.motor_id     = motor_id
        self.interface    = interface
        self.channel      = channel
        self.bitrate      = bitrate
        self.receive_timeout = receive_timeout
        self._bus         = bus
        self._owns_bus    = bus is None

    def open(self):
        if self._bus is None:
            self._bus = can.interface.Bus(interface=self.interface,
                                          channel=self.channel,
                                          bitrate=self.bitrate)
            print(f"[STATUS] AKMotorCAN id={self.motor_id} opened "
                  f"{self.interface}:{self.channel} @ {self.bitrate} bit/s")

    def close(self):
        if self._owns_bus and self._bus is not None:
            self._bus.shutdown(); self._bus = None
            print(f"[STATUS] AKMotorCAN id={self.motor_id} bus closed")

    def __enter__(self): self.open(); return self
    def __exit__(self, *_): self.close()

    def _send(self, pkt_id: int, data: bytes):
        can_id = self.motor_id | (pkt_id << 8)
        self._bus.send(can.Message(arbitration_id=can_id,
                                   data=data, is_extended_id=True))

    def receive_feedback(self, timeout: float | None = None) -> dict | None:
        """
        Wait for a Servo Mode feedback frame (func 0x29).
        Returns dict: position (°), speed (ERPM), current (A),
                      temperature (°C), error_code, error_name.
        """
        deadline = time.monotonic() + (timeout or self.receive_timeout)
        while time.monotonic() < deadline:
            msg = self._bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None: break
            if not msg.is_extended_id: continue
            func_id = (msg.arbitration_id >> 8) & 0x1FFFFF
            src_id  =  msg.arbitration_id       & 0xFF
            if src_id != self.motor_id or func_id != 0x29: continue
            if len(msg.data) < 8: continue
            d = msg.data
            pos_int = struct.unpack(">h", bytes(d[0:2]))[0]
            spd_int = struct.unpack(">h", bytes(d[2:4]))[0]
            cur_int = struct.unpack(">h", bytes(d[4:6]))[0]
            return {
                "position":    pos_int * 0.1,
                "speed":       spd_int * 10.0,
                "current":     cur_int * 0.01,
                "temperature": d[6],
                "error_code":  d[7],
                "error_name":  can_servo_fault_name(d[7]),  # §4.3.1 CAN enum
            }
        return None

    # Servo Mode commands
    def set_duty(self, duty: float):
        self._send(CAN_PKT_SET_DUTY, encode_can_duty(duty))

    def set_current(self, current_a: float):
        self._send(CAN_PKT_SET_CURRENT, encode_can_current(current_a))

    def set_current_brake(self, current_a: float):
        self._send(CAN_PKT_SET_CURRENT_BRAKE, encode_can_current_brake(current_a))

    def set_rpm(self, rpm: float):
        self._send(CAN_PKT_SET_RPM, encode_can_rpm(rpm))

    def set_position(self, degrees: float):
        self._send(CAN_PKT_SET_POS, encode_can_pos(degrees))

    def set_origin(self, permanent: bool = False):
        self._send(CAN_PKT_SET_ORIGIN_HERE, bytes([1 if permanent else 0]))
        print(f"[STATUS] AKMotorCAN id={self.motor_id}: origin set")

    def set_position_velocity(self, degrees: float,
                              speed_erpm: int = 5000,
                              accel_erpm_s: int = 30000):
        self._send(CAN_PKT_SET_POS_SPD, encode_can_pos_spd(degrees, speed_erpm, accel_erpm_s))

    def disable(self):
        """Disable motor output (CAN_PKT_SET_DISABLE = 15, NEW in V3.2)."""
        self._send(CAN_PKT_SET_DISABLE, bytes(8))
        print(f"[STATUS] AKMotorCAN id={self.motor_id}: disable command sent")

    def configure_feedback(self, frame_flags: int):
        """Configure optional feedback frames (CAN_PKT_SET_FRAME_CONFIG = 16)."""
        payload = bytearray(8)
        payload[6] = (frame_flags >> 8) & 0xFF
        payload[7] =  frame_flags       & 0xFF
        self._send(CAN_PKT_SET_FRAME_CONFIG, bytes(payload))

    def stop(self):
        self.set_current(0.0)

    def brake(self, current_a: float = 5.0):
        self.set_current_brake(current_a)

    def print_state(self):
        fb = self.receive_feedback()
        if fb is None:
            print(f"[ERROR] AKMotorCAN id={self.motor_id}: no feedback received")
            return
        print(f"---- AK Motor CAN State (id={self.motor_id}) ----")
        print(f"  Position    : {fb['position']:.2f} °")
        print(f"  Speed       : {fb['speed']:.1f} ERPM")
        print(f"  Current     : {fb['current']:.3f} A")
        print(f"  Temperature : {fb['temperature']} °C")
        print(f"  Error       : {fb['error_name']}")
        print("-------------------------------------------------")


# ===========================================================================
# AKMotorMIT – MIT Force-Control Mode over CAN bus
# ===========================================================================

class AKMotorMIT:
    """
    MIT (impedance / force-control) mode CAN driver for one AK Series motor.

    Motor executes: τ = t_ff + kp × (p_des − p) + kd × (v_des − v)

    Typical usage:
        motor = AKMotorMIT(motor_id=1, model='AK80-9')
        motor.open()
        motor.send_mit_cmd(p_des=0.5, v_des=0.0, kp=100.0, kd=2.0, t_ff=0.0)
        reply = motor.receive_mit_reply()
        motor.stop()
        motor.close()
    """

    def __init__(self, motor_id: int,
                 interface: str = 'pcan',
                 channel: str = 'PCAN_USBBUS1',
                 bitrate: int = 1_000_000,
                 receive_timeout: float = 0.05,
                 bus: "can.BusABC | None" = None,
                 model: str | None = None,
                 p_min: float = _MIT_P_MIN, p_max: float = _MIT_P_MAX,
                 v_min: float = _MIT_V_MIN, v_max: float = _MIT_V_MAX,
                 t_min: float = _MIT_T_MIN, t_max: float = _MIT_T_MAX):
        _require_can()
        self.motor_id     = motor_id
        self.interface    = interface
        self.channel      = channel
        self.bitrate      = bitrate
        self.receive_timeout = receive_timeout
        self._bus         = bus
        self._owns_bus    = bus is None
        self.p_min = p_min; self.p_max = p_max

        if model and model in MIT_MODEL_PARAMS:
            kt, v_lim, t_lim = MIT_MODEL_PARAMS[model]
            self.kt    = kt
            self.v_min = -v_lim; self.v_max = v_lim
            self.t_min = -t_lim; self.t_max = t_lim
            print(f"[STATUS] AKMotorMIT id={motor_id}: model={model}, "
                  f"Kt={kt} N·m/A, v=±{v_lim} rad/s, t=±{t_lim} N·m")
        else:
            self.kt    = None
            self.v_min = v_min; self.v_max = v_max
            self.t_min = t_min; self.t_max = t_max

    def open(self):
        if self._bus is None:
            self._bus = can.interface.Bus(interface=self.interface,
                                          channel=self.channel,
                                          bitrate=self.bitrate)
            print(f"[STATUS] AKMotorMIT id={self.motor_id} opened "
                  f"{self.interface}:{self.channel}")

    def close(self):
        if self._owns_bus and self._bus is not None:
            self._bus.shutdown(); self._bus = None
            print(f"[STATUS] AKMotorMIT id={self.motor_id} bus closed")

    def __enter__(self): self.open(); return self
    def __exit__(self, *_): self.close()

    def send_mit_cmd(self, p_des: float, v_des: float,
                     kp: float, kd: float, t_ff: float):
        data   = encode_mit_can(p_des, v_des, kp, kd, t_ff,
                                self.p_min, self.p_max,
                                self.v_min, self.v_max,
                                self.t_min, self.t_max)
        can_id = self.motor_id | (CAN_PKT_SET_MIT << 8)
        self._bus.send(can.Message(arbitration_id=can_id,
                                   data=data, is_extended_id=True))

    def receive_mit_reply(self, timeout: float | None = None) -> dict | None:
        """Wait for one MIT feedback frame. Returns None on timeout."""
        cmd_id   = self.motor_id | (CAN_PKT_SET_MIT << 8)
        deadline = time.monotonic() + (timeout or self.receive_timeout)
        while time.monotonic() < deadline:
            msg = self._bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None: break
            if not msg.is_extended_id: continue
            if msg.arbitration_id != cmd_id: continue
            if len(msg.data) < 8: continue
            return decode_mit_can(bytes(msg.data),
                                  self.p_min, self.p_max,
                                  self.v_min, self.v_max,
                                  self.t_min, self.t_max)
        return None

    def torque_from_current(self, iq_a: float) -> float | None:
        """Convert Iq current (A) → torque (N·m) using T = Iq × Kt."""
        if self.kt is None:
            print("[WARN] AKMotorMIT: no Kt – pass model= to constructor")
            return None
        return iq_a * self.kt

    def stop(self):
        self.send_mit_cmd(0.0, 0.0, 0.0, 0.0, 0.0)

    def brake(self, kd: float = 2.0):
        self.send_mit_cmd(0.0, 0.0, 0.0, kd, 0.0)

    def print_state(self):
        self.send_mit_cmd(0.0, 0.0, 0.0, 0.0, 0.0)
        state = self.receive_mit_reply()
        if state is None:
            print(f"[ERROR] AKMotorMIT id={self.motor_id}: no response")
            return
        print(f"---- AK Motor MIT State (id={self.motor_id}) ----")
        print(f"  Position    : {state['position']:.5f} rad  "
              f"({state['position'] * 180.0 / 3.14159:.2f}°)")
        print(f"  Velocity    : {state['velocity']:.4f} rad/s")
        print(f"  Torque      : {state['torque']:.4f} N·m")
        print(f"  Temperature : {state['temperature']} °C")
        print(f"  Error       : {state['error_name']}")
        print("-------------------------------------------------")


# ===========================================================================
# AKController – unified CAN bus wrapper for a group of motors
# ===========================================================================

class AKController:
    """
    Manages a group of CubeMars AK motors on a shared CAN bus.

    Use the factory classmethods to create Servo or MIT groups:

        # MIT mode – 3-motor spirob
        ctrl = AKController.make_mit(
            motor_ids=[1, 2, 3], model='AK80-9',
        )
        ctrl.open()
        ctrl.send_mit_cmd(p_des=[0.5, -0.3, 0.1], v_des=0.0,
                          kp=100.0, kd=2.0, t_ff=0.0)
        replies = ctrl.receive_mit_replies()
        ctrl.stop()
        ctrl.close()

        # Servo mode
        ctrl = AKController.make_servo(
            motor_ids=[1, 2, 3],
        )
        ctrl.open()
        ctrl.set_position([90.0, 0.0, -45.0])
        ctrl.stop()
        ctrl.close()
    """

    def __init__(self, motors: "list[AKMotorCAN | AKMotorMIT]"):
        self.motors = motors
        self._shared_bus = None

    @classmethod
    def make_servo(cls, motor_ids: list[int],
                   interface: str = 'pcan',
                   channel: str = 'PCAN_USBBUS1',
                   bitrate: int = 1_000_000) -> "AKController":
        """Create a Servo Mode group sharing one CAN bus."""
        _require_can()
        bus = can.interface.Bus(interface=interface, channel=channel,
                                bitrate=bitrate)
        motors = [AKMotorCAN(motor_id, bus=bus) for motor_id in motor_ids]
        for m in motors: m._owns_bus = False
        obj = cls(motors); obj._shared_bus = bus
        return obj

    @classmethod
    def make_mit(cls, motor_ids: list[int],
                 interface: str = 'pcan',
                 channel: str = 'PCAN_USBBUS1',
                 bitrate: int = 1_000_000,
                 model: str | None = None,
                 p_min: float = _MIT_P_MIN, p_max: float = _MIT_P_MAX,
                 v_min: float = _MIT_V_MIN, v_max: float = _MIT_V_MAX,
                 t_min: float = _MIT_T_MIN, t_max: float = _MIT_T_MAX) -> "AKController":
        """Create a MIT Mode group sharing one CAN bus."""
        _require_can()
        bus = can.interface.Bus(interface=interface, channel=channel,
                                bitrate=bitrate)
        motors = [AKMotorMIT(mid, bus=bus, model=model,
                             p_min=p_min, p_max=p_max,
                             v_min=v_min, v_max=v_max,
                             t_min=t_min, t_max=t_max)
                  for mid in motor_ids]
        for m in motors: m._owns_bus = False
        obj = cls(motors); obj._shared_bus = bus
        return obj

    def open(self):
        if self._shared_bus is not None:
            print(f"[STATUS] AKController: {len(self.motors)} motor(s) sharing bus")
        else:
            for m in self.motors: m.open()

    def close(self):
        if self._shared_bus is not None:
            self._shared_bus.shutdown(); self._shared_bus = None
            print("[STATUS] AKController: shared bus closed")
        else:
            for m in self.motors: m.close()

    def __enter__(self): self.open(); return self
    def __exit__(self, *_): self.close()

    # MIT commands
    def send_mit_cmd(self,
                     p_des: "list[float] | float",
                     v_des: "list[float] | float",
                     kp:    "list[float] | float",
                     kd:    "list[float] | float",
                     t_ff:  "list[float] | float"):
        n = len(self.motors)
        for m, p, v, k_p, k_d, t in zip(self.motors,
                                          self._bc(p_des, n),
                                          self._bc(v_des, n),
                                          self._bc(kp,    n),
                                          self._bc(kd,    n),
                                          self._bc(t_ff,  n)):
            if isinstance(m, AKMotorMIT):
                m.send_mit_cmd(p, v, k_p, k_d, t)

    def receive_mit_replies(self, timeout: float = 0.05) -> "list[dict | None]":
        return [m.receive_mit_reply(timeout=timeout)
                if isinstance(m, AKMotorMIT) else None
                for m in self.motors]

    # Servo commands
    def set_current(self, currents: "list[float] | float"):
        for m, v in zip(self.motors, self._bc(currents, len(self.motors))):
            if isinstance(m, AKMotorCAN): m.set_current(v)

    def set_rpm(self, rpms: "list[float] | float"):
        for m, v in zip(self.motors, self._bc(rpms, len(self.motors))):
            if isinstance(m, AKMotorCAN): m.set_rpm(v)

    def set_position(self, positions_deg: "list[float] | float"):
        for m, v in zip(self.motors, self._bc(positions_deg, len(self.motors))):
            if isinstance(m, AKMotorCAN): m.set_position(v)

    def set_origin(self, permanent: bool = False):
        for m in self.motors:
            if isinstance(m, AKMotorCAN): m.set_origin(permanent)

    def disable(self, motors):
        for m in motors:
            if isinstance(m, AKMotorCAN): m.disable()

    def receive_feedbacks(self, timeout: float = 0.05) -> "list[dict | None]":
        return [m.receive_feedback(timeout=timeout)
                if isinstance(m, AKMotorCAN) else None
                for m in self.motors]

    def get_feedback_all_motors(self, timeout: float = 0.05) -> "list[dict | None]":
        """
        Efficiently read one 0x29 status frame from every AKMotorCAN in one pass.

        Instead of waiting up to `timeout` seconds per motor sequentially,
        this reads all frames arriving on the shared bus within a single
        `timeout` window and bins them by src_id.

        Returns a list aligned to self.motors order.
        Each entry is a decoded feedback dict, or None if that motor did not
        respond within the timeout.
        """
        # Build a lookup: motor_id → list index
        id_to_idx = {m.motor_id: i
                     for i, m in enumerate(self.motors)
                     if isinstance(m, AKMotorCAN)}

        results   = [None] * len(self.motors)
        remaining = set(id_to_idx.keys())          # motor IDs we still need
        bus       = (self._shared_bus
                     or next((m._bus for m in self.motors
                               if isinstance(m, AKMotorCAN) and m._bus), None))
        if bus is None:
            return results

        deadline = time.monotonic() + timeout
        while remaining and time.monotonic() < deadline:
            msg = bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None:
                break
            if not msg.is_extended_id:
                continue
            func_id = (msg.arbitration_id >> 8) & 0x1FFFFF
            src_id  =  msg.arbitration_id       & 0xFF
            if func_id != 0x29 or src_id not in remaining:
                continue
            if len(msg.data) < 8:
                continue
            d = msg.data
            pos_int = struct.unpack(">h", bytes(d[0:2]))[0]
            spd_int = struct.unpack(">h", bytes(d[2:4]))[0]
            cur_int = struct.unpack(">h", bytes(d[4:6]))[0]
            results[id_to_idx[src_id]] = {
                "position":    pos_int * 0.1,
                "speed":       spd_int * 10.0,
                "current":     cur_int * 0.01,
                "temperature": d[6],
                "error_code":  d[7],
                "error_name":  can_servo_fault_name(d[7]),
            }
            remaining.discard(src_id)

        return results

    # Shared
    def stop(self):
        for m in self.motors: m.stop()

    def print_state(self):
        for i, m in enumerate(self.motors):
            print(f"\n=== Motor {i} (id={m.motor_id}) ===")
            m.print_state()

    # @staticmethod
    # def _bc(val, n: int) -> list:
    #     if isinstance(val, (int, float)):
    #         return [val] * n
    #     val = list(val)
    #     if len(val) < n:
    #         val += [val[-1]] * (n - len(val))
    #     return val[:n]
