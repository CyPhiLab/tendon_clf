"""Prediction and host-current feedback checks. No CAN bus or hardware is opened."""

import copy
import struct

import mujoco
import numpy as np
import pytest

from test_tracking_log import make_controller  # noqa: F401
from test_state_estimation_node import make_node  # noqa: F401
from motor_control.ak_motor.protocol import encode_can_current
from spirob_zmq.common import MOTOR_COMMAND, ProfileParams, VoltageLimit, load_model, rest_state, step_current
from spirob_zmq.nodes.hardware_node import HardwareNode
from spirob_zmq.velocity_feedback import (
    DIAGNOSTICS, TendonPredictor, VelocityFeedback, control_mode, restore_snapshot, state_snapshot,
)
from spirob_zmq import velocity_feedback


@pytest.fixture
def setup_feedback():
    params = ProfileParams()
    model = load_model(params, drive_model='current')
    config = VelocityFeedback(params, model)
    return params, model, config


def test_mapping_units_sign_and_limits(setup_feedback):
    _, model, config = setup_feedback
    assert config.motor_ids == [10, 1, 2]
    assert config.tendon_names == ['tendon1', 'tendon2', 'tendon3']
    np.testing.assert_allclose(config.kt, 1.27)
    np.testing.assert_allclose(model.actuator_gear[:, 0], config.kt / config.radius)
    speed, raw = config.speed_reference([-.02, .02, .08])
    np.testing.assert_allclose(raw, [-.5, .5, 2.])
    np.testing.assert_allclose(speed, [-.5, .5, 1.])
    np.testing.assert_array_equal(config.ordered([2, 10, 1], [30, 10, 20]), [10, 20, 30])
    for ids, values in (([10, 1], [1, 2]), ([10, 10, 2], [1, 2, 3]),
                        ([10, 1, 2], [1, np.nan, 3])):
        with pytest.raises(ValueError):
            config.ordered(ids, values)


@pytest.mark.parametrize('change', [
    {'tendon_name': 'tendon2'}, {'radius_m': 0}, {'sign': -1},
    {'sign': 0}, {'kt_output_nm_a': .127}, {'kd_nm_s_rad': -1},
    {'kd_nm_s_rad': np.nan}, {'speed_limit_rad_s': 0}, {'radius_m': [.04]},
])
def test_bad_configuration_rejected(setup_feedback, change):
    params, model, _ = setup_feedback
    rows = copy.deepcopy(params.declare_parameter('velocity_feedback_motors'))
    rows[0].update(change)
    with pytest.raises(ValueError):
        VelocityFeedback(ProfileParams(velocity_feedback_motors=rows), model)


def test_explicit_reordered_configuration(setup_feedback):
    params, model, _ = setup_feedback
    rows = list(reversed(params.declare_parameter('velocity_feedback_motors')))
    config = VelocityFeedback(ProfileParams(velocity_feedback_motors=rows), model)
    assert config.tendon_names == ['tendon1', 'tendon2', 'tendon3']


def test_native_and_motor_model_are_rejected(setup_feedback):
    _, model, _ = setup_feedback
    with pytest.raises(ValueError, match='MIT'):
        control_mode(ProfileParams(motor_control_mode='qp_velocity_mit'))
    with pytest.raises(ValueError, match='drive_model=current'):
        VelocityFeedback(ProfileParams(drive_model='motor'), model)


def test_zero_gain_matching_speed_and_both_directions(setup_feedback):
    _, _, config = setup_feedback
    qp = np.array([-.5014, -.7531, -1.1033])
    lower, upper = np.full(3, -1.9), np.full(3, -.1)
    sent, diag = config.command(qp, np.ones(3), np.zeros(3), lower, upper)
    np.testing.assert_allclose(sent, qp, atol=.001)
    np.testing.assert_array_equal(diag['feedback_torque_nm'], 0)
    config.kd[:] = .2
    sent, diag = config.command(qp, np.ones(3), np.ones(3), lower, upper)
    np.testing.assert_allclose(sent, qp, atol=.001)
    np.testing.assert_array_equal(diag['feedback_torque_nm'], 0)
    _, diag = config.command(qp, [-.5, .5, 0], [0, 0, 0], lower, upper)
    np.testing.assert_allclose(diag['feedback_torque_nm'], [-.1, .1, 0])
    assert diag['requested_current_a'][0] < qp[0]
    assert diag['requested_current_a'][1] > qp[1]


def test_combined_asymmetric_limits_and_wire_quantization(setup_feedback):
    _, _, config = setup_feedback
    config.kd[:] = 10.
    lower, upper = np.array([-1.9, -.7504, -.8]), np.array([-.1, -.1004, .2])
    sent, diag = config.command([-.5]*3, [-1, 1, 1], [0]*3, lower, upper)
    encoded = np.array([struct.unpack('>i', encode_can_current(v))[0] / 1000 for v in sent])
    assert np.all(encoded >= lower) and np.all(encoded <= upper)
    assert diag['saturated'].all()
    np.testing.assert_array_equal(encoded, diag['sent_current_a'])
    np.testing.assert_allclose(diag['requested_torque_nm'],
                               diag['feedforward_torque_nm'] + diag['feedback_torque_nm'])


@pytest.mark.parametrize('period, steps', [(.005, 5), (.0001, 1), (.0016, 2)])
def test_final_prediction_and_source_immutability(setup_feedback, period, steps):
    params, model, config = setup_feedback
    source = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, source, 0)
    source.qvel[:] = .02
    source.ctrl[:] = -.2
    source.qacc_warmstart[:] = .3
    mujoco.mj_forward(model, source)
    before = state_snapshot(model, source)['state'].copy()
    derived = source.ten_velocity.copy()
    predictor = TendonPredictor(model, config, period, VoltageLimit(params))
    current = np.array([-.4, -.7, -.2])
    reference, diag = predictor.predict(source, current)
    assert predictor.steps == steps
    assert diag['prediction_duration_s'] == pytest.approx(steps * .001)
    assert model.opt.timestep == .001
    assert predictor.data.time - source.time == pytest.approx(diag['prediction_duration_s'])
    np.testing.assert_array_equal(state_snapshot(model, source)['state'], before)
    np.testing.assert_array_equal(source.ten_velocity, derived)
    expected = mujoco.MjData(model)
    mujoco.mj_copyData(expected, model, source)
    mujoco.mj_forward(model, expected)
    step_current(model, expected, current, steps, VoltageLimit(params))
    stale = expected.ten_velocity.copy()
    mujoco.mj_forward(model, expected)
    assert np.max(np.abs(stale - expected.ten_velocity)) > 1e-9
    np.testing.assert_allclose(diag['tendon_velocity_m_s'], expected.ten_velocity[config.tendon_ids])
    np.testing.assert_allclose(reference, config.speed_reference(expected.ten_velocity)[0])


def test_prediction_rejects_mujoco_automatic_reset(setup_feedback, monkeypatch):
    params, model, config = setup_feedback
    source = mujoco.MjData(model)
    source.time = 1.
    predictor = TendonPredictor(model, config, .005, VoltageLimit(params))
    monkeypatch.setattr(velocity_feedback, 'step_current',
                        lambda model, data, *args: mujoco.mj_resetData(model, data))
    with pytest.raises(ValueError, match='automatic reset'):
        predictor.predict(source, np.zeros(3))
    assert source.time == 1.


def test_full_snapshot_includes_activations_and_inputs():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint name="j" type="slide"/>
    <geom size=".1" mass="1"/></body><body mocap="true"/>
    </worldbody><actuator><general joint="j" dyntype="filter" dynprm=".01"/>
    </actuator></mujoco>""")
    source, destination = mujoco.MjData(model), mujoco.MjData(model)
    source.qpos[:] = .2
    source.qvel[:] = .4
    source.act[:] = .7
    source.ctrl[:] = .9
    source.qacc_warmstart[:] = .3
    source.qfrc_applied[:] = .8
    source.xfrc_applied[:] = .6
    source.mocap_pos[:] = .5
    source.time = 1.2
    before = state_snapshot(model, source)
    restore_snapshot(model, destination, before)
    np.testing.assert_array_equal(state_snapshot(model, destination)['state'], before['state'])
    mujoco.mj_step(model, destination)
    np.testing.assert_array_equal(state_snapshot(model, source)['state'], before['state'])
    invalid = dict(before, dimensions=[0]*7)
    with pytest.raises(ValueError):
        restore_snapshot(model, destination, invalid)


def test_estimator_publishes_activation_snapshot_without_prediction_mutation(make_node):
    node = make_node(activation=True, motor_control_mode='qp_velocity_host',
                     velocity_reference_source='qp_rollout')
    node.now = lambda: 100.
    node.measurement_wall_time = 99.98
    node.x[-1] = .7
    messages = []
    node.state_pub.publish = messages.append
    node.publish(.1)
    msg = messages[-1]
    before = state_snapshot(node.model, node.data)['state'].copy()
    x_before, p_before = node.x.copy(), node.P.copy()
    predicted = mujoco.MjData(node.model)
    restore_snapshot(node.model, predicted, msg['prediction_state'])
    assert predicted.act[0] == pytest.approx(.7)
    assert msg['measurement_age_s'] == pytest.approx(.02)
    mujoco.mj_step(node.model, predicted, 3)
    np.testing.assert_array_equal(node.x, x_before)
    np.testing.assert_array_equal(node.P, p_before)
    np.testing.assert_array_equal(state_snapshot(node.model, node.data)['state'], before)


def test_failed_estimator_update_does_not_refresh_feedback_state(make_node, monkeypatch):
    node = make_node(motor_control_mode='qp_velocity_host', velocity_reference_source='qp_rollout')
    node.initialized = True
    node.have_estimate = True
    node.now = lambda: 100.
    node._on_command({'motor_id': 1, 'stamp': 100., 'control_mode': 'qp_velocity_host',
                       'qp_current_a': -.2, 'app_ctrl': -.4, 'fault_code': 0})
    messages = []
    node.state_pub.publish = messages.append
    monkeypatch.setattr(node, 'predict', lambda n: None)

    def fail_update(*args):
        raise ValueError('test correction failure')

    monkeypatch.setattr(node, 'update', fail_update)
    node._on_measurement({'stamp': .1, 'published_at': 100., 'data': [.1, .2, .3]})
    assert not messages


@pytest.fixture
def hardware(make_controller):
    _, messages = make_controller()
    node = HardwareNode({'motor_control_mode': 'qp_velocity_host', 'dry_run': True})
    clock = [100.]
    node.now = lambda: clock[0]
    node.velocity_feedback.kd[:] = .2
    node._on_true_state({
        'motor_ids': [2, 10, 1], 'motor_position': [0, 0, 0],
        'motor_velocity': [0, 0, 0], 'published_at': clock[0],
    })
    return node, messages, clock


def command(stamp=100.):
    return {
        'stamp': stamp, 'state_published_at': stamp, 'measurement_age_s': 0.,
        'motor_ids': [2, 10, 1], 'u': [-.6, -.4, -.5],
        'control_mode': 'qp_velocity_host', 'velocity_reference': [0, -.5, .5],
        'prediction': {'tendon_velocity_m_s': [0, -.02, .02],
                       'prediction_duration_s': .005, 'prediction_compute_s': .001,
                       'outer_compute_s': .002, 'reference_source': 'ekf_prior',
                       'reference_stamp': .1, 'reference_wall_time': stamp,
                       'reference_extract_compute_s': .0001, 'prediction_steps': 5,
                       'prediction_input_current_a': [-.3, -.2, -.4],
                       'prediction_input_kind': 'qp_nominal',
                       'tendon_names': ['tendon3', 'tendon1', 'tendon2']},
    }


def test_hardware_feedback_order_and_diagnostics(hardware):
    node, messages, _ = hardware
    node._on_command(command())
    assert node._velocity_fault is None
    diag = [m for topic, m in messages if topic == DIAGNOSTICS][-1]
    assert diag['motor_ids'] == [10, 1, 2]
    np.testing.assert_allclose(diag['qp_current_a'], [-.4, -.5, -.6])
    np.testing.assert_allclose(diag['feedback_torque_nm'], [-.1, .1, 0.])
    for key in ('measured_current_a', 'measurement_age_s', 'state_age_s',
                'prediction_duration_s', 'host_feedback_compute_s', 'saturated'):
        assert key in diag


@pytest.mark.parametrize('fault', ['encoder', 'state', 'command', 'drive', 'communication', 'reference'])
def test_watchdog_stops_and_latches(hardware, fault):
    node, _, clock = hardware
    node._on_command(command())
    if fault == 'encoder':
        node._velocity_states[1]['stamp'] -= 1.
    elif fault == 'state':
        node._state_published_at -= 1.
    elif fault == 'command':
        node._last_velocity_command -= 1.
    elif fault == 'drive':
        node._velocity_states[1]['fault_code'] = 5
    elif fault == 'reference':
        node._reference_wall_time -= 1.
    else:
        node._feedback_error = 'CAN disconnected'
    node._velocity_watchdog()
    assert node._velocity_fault is not None
    np.testing.assert_array_equal(node.commanded_current, 0.)
    node._on_command(command(clock[0]))
    np.testing.assert_array_equal(node.commanded_current, 0.)


@pytest.mark.parametrize('field,value', [('velocity_reference', [np.nan, 0, 0]),
                                        ('motor_ids', [10, 10, 2]),
                                        ('control_mode', 'qp_current'),
                                        ('stamp', 99.)])
def test_bad_command_stops(hardware, field, value):
    node, _, _ = hardware
    msg = command()
    msg[field] = value
    node._on_command(msg)
    assert node._velocity_fault
    np.testing.assert_array_equal(node.commanded_current, 0.)


def test_dry_run_does_not_refresh_stale_encoder(hardware):
    node, _, clock = hardware
    clock[0] += .2
    node._send(np.array([-.3]*3))
    assert node._velocity_states[10]['stamp'] == 100.
    node._on_command(command(clock[0]))
    assert 'stale encoder' in node._velocity_fault


def test_real_erpm_conversion_once(hardware):
    node, messages, _ = hardware
    node._latest_fb = [dict(position=90., speed=1400., current=-.2,
                           commanded_current=-.2, qp_current_a=-.1,
                           error_code=0, received_at=100.)] * 3
    node._publish_feedback()
    assert node._velocity_states[10]['velocity'] == pytest.approx(2*np.pi/6)
    assert node._velocity_states[10]['position'] == pytest.approx(np.pi/2)


def test_controller_prediction_state_and_staleness(make_controller):
    node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.,
                                    velocity_reference_source='qp_rollout')
    clock = [100.]
    node.now = lambda: clock[0]
    data = mujoco.MjData(node.model)
    mujoco.mj_resetDataKeyframe(node.model, data, 0)
    msg = {'stamp': 0., 'published_at': 100., 'measurement_age_s': 0.,
           'prediction_state': state_snapshot(node.model, data),
           'q': data.qpos.copy(), 'dq': data.qvel.copy(), 'is_valid': True}
    node._on_state(msg)
    np.testing.assert_array_equal(node.data.qpos, data.qpos)
    # Relax only the clock used for the budget check, not model time or production configuration.
    node.prediction_budget_s = 1.
    node._tick()
    commands = [m for topic, m in messages if topic == MOTOR_COMMAND]
    assert len(commands) == 1
    assert commands[0]['prediction']['prediction_duration_s'] == .005
    np.testing.assert_array_equal(node.data.qpos, data.qpos)
    np.testing.assert_array_equal(node.data.qvel, data.qvel)
    clock[0] += .2
    node._tick()
    assert len([m for topic, m in messages if topic == MOTOR_COMMAND]) == 1
    node._on_state(dict(msg, is_valid=False))
    assert not node.have_state


def test_controller_budget_rejects_command(make_controller):
    node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.,
                                    prediction_budget_s=1e-12, velocity_reference_source='qp_rollout')
    node.now = lambda: 100.
    source = mujoco.MjData(node.model)
    mujoco.mj_resetDataKeyframe(node.model, source, 0)
    node._on_state({'stamp': 0., 'published_at': 100., 'measurement_age_s': 0.,
                    'prediction_state': state_snapshot(node.model, source),
                    'q': source.qpos, 'dq': source.qvel, 'is_valid': True})
    previous = node.last_u.copy()
    node._tick()
    assert not any(topic == MOTOR_COMMAND for topic, _ in messages)
    np.testing.assert_array_equal(node.last_u, previous)


def test_prediction_timing_budget(make_controller):
    node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.,
                                    rate_hz=50., velocity_reference_source='qp_rollout')
    node.now = lambda: 100.
    source = mujoco.MjData(node.model)
    rest_state(node, node.model, source)
    for i in range(120):
        source.qvel[:] = .01 * np.sin(i)
        node._on_state({'stamp': i * .02, 'published_at': 100., 'measurement_age_s': 0.,
                        'prediction_state': state_snapshot(node.model, source),
                        'q': source.qpos, 'dq': source.qvel, 'is_valid': True})
        node._tick()
    records = [m['prediction'] if t == MOTOR_COMMAND else m for t, m in messages
               if t == MOTOR_COMMAND or t == DIAGNOSTICS]
    assert len(records) == 120
    outer = np.array([m['outer_compute_s'] for m in records])
    prediction = np.array([m['prediction_compute_s'] for m in records])
    print(f'\n120 QP + prediction updates: prediction p95={np.percentile(prediction, 95)*1e3:.3f} ms; '
          f'outer p50/p95/max={np.percentile(outer, 50)*1e3:.3f}/'
          f'{np.percentile(outer, 95)*1e3:.3f}/{outer.max()*1e3:.3f} ms; '
          f'budget={node.prediction_budget_s*1e3:.3f} ms; '
          f'rejected={np.count_nonzero(outer > node.prediction_budget_s)}')
    assert np.percentile(outer, 95) < node.prediction_budget_s
    assert all(m['prediction']['outer_compute_s'] <= node.prediction_budget_s
               for t, m in messages if t == MOTOR_COMMAND)
