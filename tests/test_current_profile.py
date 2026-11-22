"""Model configuration and offline safety checks for current-profile experiments."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.common import ProfileParams, load_model  # noqa: E402
from spirob_zmq.core import Logger, Node  # noqa: E402
from spirob_zmq.launch import LAUNCHES, MODE_PARAMS  # noqa: E402
from spirob_zmq.nodes.current_profile_node import CurrentProfileNode  # noqa: E402


def test_scale_parameters_preserve_nominal_geometry_and_mass():
    nominal = load_model(ProfileParams())
    scaled = load_model(ProfileParams(joint_stiffness_scale=1.7, joint_damping_scale=0.6))
    np.testing.assert_allclose(scaled.jnt_stiffness, nominal.jnt_stiffness * 1.7)
    np.testing.assert_allclose(scaled.dof_damping, nominal.dof_damping * 0.6)
    for name in ('body_mass', 'body_inertia', 'dof_armature', 'tendon_armature',
                 'tendon_damping', 'tendon_frictionloss', 'qpos_spring'):
        np.testing.assert_array_equal(getattr(scaled, name), getattr(nominal, name))


@pytest.mark.parametrize('name', ['joint_stiffness_scale', 'joint_damping_scale'])
@pytest.mark.parametrize('value', [0., -1., np.nan])
def test_invalid_scales(name, value):
    with pytest.raises(ValueError, match=name):
        load_model(ProfileParams(**{name: value}))


def test_model_timestep_is_explicit_and_validated():
    assert load_model(ProfileParams(model_timestep=0.0005)).opt.timestep == 0.0005
    with pytest.raises(ValueError, match='model_timestep'):
        load_model(ProfileParams(model_timestep=0.))


@pytest.fixture
def profile(monkeypatch):
    def init(self, name, params=None):
        self._params, self._profile = params or {}, {}
        self._logger = Logger(name)

    messages, clock = [], [0.]
    monkeypatch.setattr(Node, '__init__', init)
    monkeypatch.setattr(Node, 'create_subscription', lambda *a, **kw: None)
    monkeypatch.setattr(Node, 'create_timer', lambda *a: None)
    monkeypatch.setattr(Node, 'create_publisher',
                        lambda *a: SimpleNamespace(publish=lambda msg: messages.append(msg)))

    def make(**params):
        node = CurrentProfileNode({
            'tests': ['mechanics'], 'mechanics_levels': [-0.2, -0.3],
            'mechanics_ramp_s': [0.04, 0.08], 'hold_s': 0.02, 'lead_s': 0.,
            **params,
        })
        node.now = lambda: clock[0]
        node._on_initial_state({})
        return node, messages, clock

    return make


def test_mechanics_profile_is_negative_bounded_multispeed_and_finishes(profile):
    node, messages, clock = profile()
    for _ in range(500):
        for motor in node.motor_ids:
            node._on_motor_state({'motor_id': motor, 'position': 0.})
        node._tick()
        if node.finished:
            break
        clock[0] += node.dt
    assert node.finished
    currents = np.array([msg['u'] for msg in messages])
    assert np.all(currents >= -0.3) and np.all(currents <= -0.1)
    np.testing.assert_array_equal(currents[-1], np.full(3, -0.1))
    assert np.all(np.count_nonzero(currents < -0.10001, axis=1) <= 1)
    assert np.max(np.abs(np.diff(currents, axis=0))) <= 0.05 + 1e-12
    phases = [msg['phase'] for msg in messages]
    assert any('0.04s' in p for p in phases) and any('0.08s' in p for p in phases)
    for motor in range(3):
        assert any(f'mechanics {motor} -0.3A' in p for p in phases)


def test_staged_mechanics_holds_prior_motors_then_releases_all(profile):
    node, messages, clock = profile(
        mechanics_sequence='staged', floor_current=-0.2,
        mechanics_levels=[-0.4], mechanics_ramp_s=[4.], hold_s=2.,
    )
    for step in range(2600):
        clock[0] = step * node.dt
        for motor in node.motor_ids:
            node._on_motor_state({'motor_id': motor, 'position': 0., 'stamp': clock[0]})
        node._tick()
        if node.finished:
            break
    assert node.finished
    currents = np.array([m['u'] for m in messages])
    assert np.all(currents >= -0.4) and np.all(currents <= -0.2)
    assert np.max(np.abs(np.diff(currents, axis=0))) <= 0.0005 + 1e-12
    last_hold_end = -1.
    for i, motor in enumerate(node.motor_ids):
        load = [m for m in messages if f'motor {motor} load' in m['phase']]
        hold = [m for m in messages if f'motor {motor} hold' in m['phase']]
        assert load and hold
        assert load[0]['stamp'] > last_hold_end
        assert abs(len(load) * node.dt - 4.) <= 2 * node.dt
        assert abs(len(hold) * node.dt - 2.) <= 2 * node.dt
        last_hold_end = hold[-1]['stamp']
        for msg in load + hold:
            np.testing.assert_allclose(msg['u'][:i], -0.4)
            np.testing.assert_allclose(msg['u'][i + 1:], -0.2)
        for msg in hold:
            assert msg['u'][i] == pytest.approx(-0.4)
    unload = [m for m in messages if m['phase'].endswith('unload_all')]
    assert abs(len(unload) * node.dt - 4.) <= 2 * node.dt
    for msg in unload:
        np.testing.assert_allclose(msg['u'], np.full(3, msg['u'][0]))
    np.testing.assert_allclose(messages[-1]['u'], [-0.2] * 3)


def test_staged_mechanics_feedback_loss_releases_all_held_motors(profile):
    node, messages, clock = profile(
        mechanics_sequence='staged', floor_current=-0.2,
        mechanics_levels=[-0.4], mechanics_ramp_s=[0.04], hold_s=0.02,
    )
    for step in range(100):
        clock[0] = step * node.dt
        for motor in node.motor_ids:
            node._on_motor_state({'motor_id': motor, 'position': 0., 'stamp': clock[0]})
        node._tick()
        if 'motor 1 hold' in messages[-1]['phase']:
            break
    np.testing.assert_allclose(messages[-1]['u'], [-0.4, -0.4, -0.2])
    clock[0] += node.feedback_timeout_s + node.dt
    for motor in node.motor_ids[1:]:
        node._on_motor_state({'motor_id': motor, 'position': 0., 'stamp': clock[0]})
    node._tick()
    assert node.finished and messages[-1]['event']['aborted']
    np.testing.assert_allclose(messages[-1]['u'], [-0.2] * 3)


@pytest.mark.parametrize('params', [
    {'mechanics_sequence': 'unknown'},
    {'mechanics_sequence': 'staged', 'mechanics_motion_limited': True},
])
def test_invalid_staged_mechanics_configuration(profile, params):
    with pytest.raises(ValueError):
        profile(**params)


@pytest.mark.parametrize('reason', ['stale', 'invalid'])
def test_mechanics_aborts_to_floor(profile, reason):
    node, messages, clock = profile()
    for motor in node.motor_ids:
        node._on_motor_state({'motor_id': motor, 'position': 0.})
    node._tick()
    clock[0] = 1.
    if reason != 'stale':
        for i, motor in enumerate(node.motor_ids):
            node._on_motor_state({'motor_id': motor,
                                 'position': np.nan if i == 0 else 0.})
    node._tick()
    assert node.finished
    np.testing.assert_array_equal(messages[-1]['u'], np.full(3, node.floor))
    assert messages[-1]['event']['aborted']


def test_missing_motor_aborts_even_while_others_keep_reporting(profile):
    node, messages, clock = profile()
    for motor in node.motor_ids:
        node._on_motor_state({'motor_id': motor, 'position': 0., 'stamp': 0.})
    node._tick()
    for step in range(1, 60):
        clock[0] = step * node.dt
        for motor in node.motor_ids[1:]:
            node._on_motor_state({'motor_id': motor, 'position': 0., 'stamp': clock[0]})
        # A delayed/replayed reading must not refresh the missing motor.
        node._on_motor_state({'motor_id': node.motor_ids[0], 'position': 0., 'stamp': 0.})
        node._tick()
        if node.finished:
            break
    assert node.finished and clock[0] <= node.feedback_timeout_s + 2 * node.dt
    assert messages[-1]['event']['aborted']
    np.testing.assert_array_equal(messages[-1]['u'], np.full(3, node.floor))


def test_delayed_feedback_does_not_allow_excitation_at_startup(profile):
    node, messages, clock = profile()
    clock[0] = 0.1
    for motor in node.motor_ids:
        node._on_motor_state({'motor_id': motor, 'position': 0., 'stamp': -1.})
    node._tick()
    assert messages[-1]['phase'] == 'wait'
    assert node.start_angle is None
    clock[0] = 0.6
    node._tick()
    assert node.finished and messages[-1]['event']['aborted']


@pytest.mark.parametrize('bad_stamp', [float('nan'), float('inf'), 100.])
def test_invalid_feedback_timestamp_is_not_fresh(profile, bad_stamp):
    node, _, _ = profile()
    node._on_motor_state({'motor_id': node.motor_ids[0], 'position': 1., 'stamp': bad_stamp})
    assert node.feedback_time[0] == -np.inf
    assert np.isnan(node.angle[0])


@pytest.mark.parametrize('params', [
    {}, {'mechanics_post_onset_s': 0.}, {'mechanics_post_onset_s': np.inf},
    {'mechanics_post_onset_s': 0.1, 'onset_threshold': 0.},
])
def test_motion_limited_requires_explicit_valid_limits(profile, params):
    with pytest.raises(ValueError):
        profile(mechanics_motion_limited=True, **params)


@pytest.mark.parametrize('trigger', ['time', 'no_motion'])
def test_motion_limited_pull_releases_to_floor_same_tick(profile, trigger):
    node, messages, clock = profile(
        mechanics_motion_limited=True,
        mechanics_post_onset_s=0.03, mechanics_levels=[-0.3],
        mechanics_ramp_s=[0.1], hold_s=0.05,
    )
    positions = np.zeros(3)
    onset_current = None
    released = None
    for step in range(40):
        clock[0] = step * node.dt
        if step >= 2 and trigger != 'no_motion':
            positions[0] = -0.01
        if step >= 3:
            positions[1] = 3.0
            if trigger == 'time':
                positions[0] = -3.0
        for i, motor in enumerate(node.motor_ids):
            node._on_motor_state({'motor_id': motor, 'position': positions[i], 'stamp': clock[0]})
        node._tick()
        msg = messages[-1]
        event = msg.get('event', {})
        if event.get('kind') == 'mechanics_onset':
            onset_current = msg['u'][0]
        if 'post_onset' in msg['phase']:
            assert msg['u'][0] == onset_current
        if event.get('kind') == 'mechanics_release':
            released = msg
            break
    assert released is not None
    expected = 'post_onset_time' if trigger == 'time' else 'no_motion'
    assert released['event']['reason'] == expected
    np.testing.assert_array_equal(released['u'], np.full(3, node.floor))
    assert not node.finished
    assert 'recover' in released['phase']
    if trigger == 'time':
        assert clock[0] <= 0.02 + node.mechanics_post_onset_s + node.dt
    # Large motion during recovery does not trigger an angle-based abort.
    clock[0] += node.dt
    for i, motor in enumerate(node.motor_ids):
        node._on_motor_state({'motor_id': motor, 'position': -6. if i == 0 else 0.,
                              'stamp': clock[0]})
    node._tick()
    assert not node.finished
    np.testing.assert_array_equal(messages[-1]['u'], np.full(3, node.floor))


@pytest.mark.parametrize('tests', [['mechanics'], ['together']])
def test_large_motor_travel_does_not_abort_profile(profile, tests):
    node, messages, clock = profile(tests=tests)
    for step in range(10):
        clock[0] = step * node.dt
        for i, motor in enumerate(node.motor_ids):
            node._on_motor_state({'motor_id': motor, 'position': step * (i + 1.)})
        node._tick()
        assert not node.finished
        assert not messages[-1].get('event', {}).get('aborted', False)
        assert np.all(messages[-1]['u'] >= node.in_max)
        assert np.all(messages[-1]['u'] <= node.out_max)


def test_removed_travel_settings_emit_warning(profile, capsys):
    profile(max_travel=0.01, max_total_travel=0.02, mechanics_pull_travel=0.005)
    output = capsys.readouterr().err
    for name in ('max_travel', 'max_total_travel', 'mechanics_pull_travel'):
        assert f'{name} is no longer used' in output


@pytest.mark.parametrize('params', [
    {'mechanics_levels': [0.]}, {'mechanics_levels': [-1.]},
    {'mechanics_ramp_s': [0.]}, {'mechanics_ramp_s': [np.nan]},
    {'feedback_timeout_s': -1.}, {'floor_current': 0.},
    {'rate_hz': 0.}, {'rate_hz': np.nan},
])
def test_invalid_mechanics_configuration(profile, params):
    with pytest.raises(ValueError):
        profile(**params)


def test_new_modes_do_not_open_positive_current_range():
    for mode in ('mechanics_sysid', 'real_mechanics_sysid'):
        assert 'current_profile_node' in LAUNCHES[mode]
        assert MODE_PARAMS[mode]['current_profile_node'] == ['tests=["mechanics"]']
        assert all('actuator_ctrlrange' not in item
                   for params in MODE_PARAMS[mode].values() for item in params)


