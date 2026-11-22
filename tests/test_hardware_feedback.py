"""Feedback freshness checks without a CAN adapter or motor commands."""

import sys
import threading
import struct
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.nodes.hardware_node import HardwareNode  # noqa: E402
from motor_control.ak_motor import motor_can  # noqa: E402
from test_current_profile import profile  # noqa: E402, F401


def test_missing_motor_is_not_republished_with_other_motors():
    node = HardwareNode.__new__(HardwareNode)
    node.motor_ids = [10, 1, 2]
    node._fb_lock = threading.Lock()
    node._latest_fb = [None] * 3
    node.commanded_current = [-0.2] * 3
    node.pole_pairs, node.gear = 14, 10
    clock = [10.]
    node.now = lambda: clock[0]
    messages = []
    node.state_pub = SimpleNamespace(publish=messages.append)
    fb = dict(position=0., speed=0., current=-0.2, error_code=0, received_at=10.)
    batches = [[dict(fb), dict(fb), dict(fb)], [dict(fb, received_at=11.), None, None]]

    def receive(timeout):
        return batches.pop(0)

    node.controller = SimpleNamespace(get_feedback_all_motors=receive)
    for expected in ([10, 1, 2], [10]):
        calls = iter([True, False])
        node.ok = lambda: next(calls)
        node._feedback_loop()
        clock[0] += 1.
        messages.clear()
        node._publish_feedback()
        assert [m['motor_id'] for m in messages] == expected
        assert all(m['stamp'] == clock[0] - 1. for m in messages)
        messages.clear()
        node._publish_feedback()
        assert not messages


def test_driver_timestamps_only_received_frames(monkeypatch):
    controller = motor_can.AKController.__new__(motor_can.AKController)
    controller.motors = []
    for motor_id in (10, 1, 2):
        motor = motor_can.AKMotorCAN.__new__(motor_can.AKMotorCAN)
        motor.motor_id = motor_id
        controller.motors.append(motor)
    frames = iter([
        SimpleNamespace(is_extended_id=True, arbitration_id=(0x29 << 8) | 10,
                        data=struct.pack('>hhhBB', 100, 2, -20, 25, 0), timestamp=123.),
        None,
    ])
    controller._shared_bus = SimpleNamespace(recv=lambda timeout: next(frames))
    monkeypatch.setattr(motor_can.time, 'time', lambda: 1000.)
    result = controller.get_feedback_all_motors()
    assert result[0]['received_at'] == 1000.
    assert result[0]['position'] == 10.
    assert result[0]['current'] == -0.2
    assert result[1:] == [None, None]


def test_hardware_to_profile_missing_motor_timeout(profile):
    node, messages, clock = profile()
    hardware = HardwareNode.__new__(HardwareNode)
    hardware.motor_ids = node.motor_ids
    hardware._fb_lock = threading.Lock()
    hardware._latest_fb = [None] * 3
    hardware.commanded_current = [-0.2] * 3
    hardware.pole_pairs, hardware.gear = 14, 10
    hardware.now = lambda: clock[0]
    hardware.state_pub = SimpleNamespace(publish=node._on_motor_state)
    for step in range(60):
        clock[0] = step * node.dt
        fb = dict(position=0., speed=0., current=-0.2, error_code=0, received_at=clock[0])
        batch = [dict(fb) if step == 0 else None, dict(fb), dict(fb)]
        hardware.controller = SimpleNamespace(get_feedback_all_motors=lambda timeout: batch)
        calls = iter([True, False])
        hardware.ok = lambda: next(calls)
        hardware._feedback_loop()
        hardware._publish_feedback()
        node._tick()
        if node.finished:
            break
    assert node.finished
    assert clock[0] <= node.feedback_timeout_s + 2 * node.dt
    assert messages[-1]['event']['aborted']
    assert all(value == node.floor for value in messages[-1]['u'])


def test_dry_run_feedback_still_uses_simulated_positions():
    node = HardwareNode.__new__(HardwareNode)
    node.motor_ids = [10, 1]
    node.dry_run = True
    node.sim_position = [0.1, -0.2]
    node.now = lambda: 12.
    messages = []
    node.state_pub = SimpleNamespace(publish=messages.append)
    node._send([-0.2, -0.3])
    assert [m['position'] for m in messages] == node.sim_position
    assert [m['stamp'] for m in messages] == [12., 12.]
    assert [m['commanded_current'] for m in messages] == [-0.2, -0.3]
