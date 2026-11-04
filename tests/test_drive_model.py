"""Checks of drive_model='motor' (spirob_zmq.drive): MJCF export, current mode, stick-slip, ratchet.

    python tests/test_drive_model.py        (or pytest tests/)
"""

import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq import drive  # noqa: E402
from spirob_zmq.common import ProfileParams, VoltageLimit, load_model, rest_state, step_current  # noqa: E402

PARAMS = ProfileParams(drive_model='motor')
G = drive.GEAR


def _model():
    model = load_model(PARAMS)
    data = mujoco.MjData(model)
    rest_state(PARAMS, model, data)
    return model, data


def _ramp(model, data, motor, start, stop, rate=0.02, hold=0.5):
    """Ramp one motor's current (others at the -0.1 A floor); return (currents, motor angles) per step."""
    vlimit = VoltageLimit(PARAMS)
    n = int(abs(stop - start) / rate / model.opt.timestep)
    u = np.full(model.nu, -0.1)
    cur, ang = [], []
    for k in range(n + int(hold / model.opt.timestep)):
        u[motor] = start + (stop - start) * min(1.0, k / n)
        step_current(model, data, u, 1, vlimit)
        cur.append(u[motor])
        ang.append(drive.motor_position(model, data)[motor])
    return np.array(cur), np.array(ang)


def test_export_is_plain_mjcf():
    """The exported XML compiles to the same model and steps identically, without this module."""
    model, data = _model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'spirob_motor.xml')
        drive.main(['--export', path])
        exported = mujoco.MjModel.from_xml_path(path)
    assert (exported.nq, exported.nv, exported.nu, exported.na) == (model.nq, model.nv, model.nu, model.na)
    for field in ('tendon_lengthspring', 'jnt_stiffness', 'qpos_spring', 'dof_frictionloss', 'dof_solref',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm', 'actuator_ctrlrange'):
        assert np.allclose(getattr(exported, field), getattr(model, field)), field
    d1, d2 = mujoco.MjData(model), mujoco.MjData(exported)
    d1.qpos[:] = d2.qpos[:] = model.key_qpos[0]
    for d, m in ((d1, model), (d2, exported)):
        d.ctrl[:] = [-0.5, -0.1, -0.1]
        mujoco.mj_step(m, d, 1000)
    assert np.abs(d1.qpos - d2.qpos).max() < 1e-5


def test_layout_and_rest_tension():
    model, data = _model()
    assert drive.is_motor_model(model)
    assert drive.robot_nq(model) == 48 and model.nq == model.nv == 52 and model.na == 3
    assert np.allclose(drive.rest_tension(model), drive.MOTOR_DRIVE['drive_rest_tension'])
    # A new start pose: the drives placed for the rest tensions
    q = data.qpos[:48] + 0.02
    data.qpos[:] = drive.full_qpos(model, q)
    mujoco.mj_forward(model, data)
    assert np.allclose(drive.tendon_tension(model, data), drive.rest_tension(model), atol=1e-6)
    # The current model is unchanged
    current = load_model(ProfileParams())
    assert not drive.is_motor_model(current) and current.nq == 48


def test_current_mode_and_bus_limit():
    """ctrl is the motor current: G * i on the tendon at rest, and the 24 V bus caps the speed."""
    model, data = _model()
    data.ctrl[:] = [-0.2, -0.1, -0.1]
    data.qvel[:] = 0.0
    data.act[:] = 0.0  # no bristle deflection: no friction force
    mujoco.mj_forward(model, data)
    assert np.allclose(data.actuator_force[0], -0.2, atol=1e-9)  # actuator units: amps
    assert np.allclose(data.qfrc_actuator[drive.motor_qpos_ids(model)[0]], -0.2 * G)
    # Paying out with the tendon slack the motor runs free until the back-EMF reaches the bus voltage
    model, data = _model()
    data.ctrl[:] = [0.55, -0.1, -0.1]
    mujoco.mj_step(model, data, 500)
    emf = data.actuator_velocity[0]   # actuator velocity = Kt * motor speed = back-EMF [V]
    assert drive.tendon_tension(model, data)[0] == 0.0
    assert 20.0 < emf <= 24.0 - 2.2 * 0.0, emf


def test_stuck_drive_slips_and_ratchets():
    """Wind-in: the drive holds, breaks away, slips, and nothing springs back when unloaded."""
    model, data = _model()
    static = np.broadcast_to(drive.MOTOR_DRIVE['drive_static_friction'], (3,))[0]
    tension = drive.tendon_tension(model, data)[0]
    cur, ang = _ramp(model, data, 0, -0.1, -0.6)
    moved = np.flatnonzero(np.abs(ang - ang[0]) > 0.03)
    assert moved.size, 'motor never slipped'
    onset = cur[moved[0]]
    # Taut: static friction plus the tension, plus what the robot resists while the slip starts
    assert -(static + tension) / G - 0.08 < onset < -(static + tension) / G + 0.03, onset
    # Below breakaway the stuck drive barely turns
    held = np.abs(cur) < 0.8 * (static + tension) / G
    assert np.abs(ang[held] - ang[0]).max() < 0.01
    assert ang[-1] - ang[0] < -0.05
    # Unload back to the floor current: no spring-back
    _, ang2 = _ramp(model, data, 0, cur[-1], -0.1, rate=0.1)
    # Only the stuck drive's own elastic give (LuGre bristle, 16 N / 4e4 N/m = 0.4 mm = 0.01 rad here,
    # plus some bristle creep): the hardware returns ~0.005 rad after +-0.24 A cycles. A drive that
    # unwound would return about the whole slip (> 1 rad).
    assert abs(ang2[-1] - ang[-1]) < 0.03


def test_slack_wind_in_breaks_away_at_static_friction():
    """With the tendon slack the drive's own static friction alone sets the breakaway."""
    model, data = _model()
    static = np.broadcast_to(drive.MOTOR_DRIVE['drive_static_friction'], (3,))[1]
    step_current(model, data, np.array([-0.1, 0.5, -0.1]), 300, VoltageLimit(PARAMS))   # pay out
    step_current(model, data, np.array([-0.1, 0.0, -0.1]), 500, VoltageLimit(PARAMS))
    assert drive.tendon_tension(model, data)[1] == 0.0
    cur, ang = _ramp(model, data, 1, 0.0, -0.45)
    moved = np.flatnonzero(np.abs(ang - ang[0]) > 0.03)
    assert moved.size
    assert abs(cur[moved[0]] + static / G) < 0.03, (cur[moved[0]], -static / G)


def test_slack_tendon_pays_out_freely():
    """Positive current pays the tendon out; the tendon goes slack instead of pushing the robot."""
    model, data = _model()
    start = drive.motor_position(model, data)[0]
    step_current(model, data, np.array([0.5, -0.1, -0.1]), 1000, VoltageLimit(PARAMS))
    assert drive.motor_position(model, data)[0] - start > 0.1
    assert drive.tendon_tension(model, data)[0] < 1e-9


if __name__ == '__main__':
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as exc:
                failed += 1
                print(f'FAIL {name}: {exc}')
    sys.exit(1 if failed else 0)
