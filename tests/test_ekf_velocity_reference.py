"""Reuse the EKF prior without extra integration or relabeling corrected velocities."""

import copy
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_state_estimation_node import make_node  # noqa: F401
from test_tracking_log import make_controller  # noqa: F401
from test_velocity_feedback import hardware, command  # noqa: F401
from spirob_zmq.common import MOTOR_COMMAND, MOTOR_STATE, rest_state
from spirob_zmq.nodes import state_estimation_node as estimation
from spirob_zmq.nodes.virtual_measurement_node import VirtualMeasurementNode
from spirob_zmq.velocity_feedback import DIAGNOSTICS, reference_source, restore_snapshot, state_snapshot


@pytest.fixture
def prior_estimator(make_node, monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <option timestep=".001" gravity="0 0 0"/>
      <worldbody>
        <site name="anchor"/>
        <body pos="0 .3 0">
          <joint name="j" type="slide" axis="0 1 0" stiffness="10" damping=".1"/>
          <geom size=".02" mass="1"/>
          <site name="ee"/>
        </body>
      </worldbody>
      <tendon><spatial name="tendon1"><site site="anchor"/><site site="ee"/></spatial></tendon>
      <actuator><general tendon="tendon1" gear="31.75" dyntype="filter" dynprm=".02"
                        ctrlrange="-1.9 -.1"/></actuator>
      <keyframe><key name="init" qpos="0"/></keyframe>
    </mujoco>""")
    monkeypatch.setattr(estimation, 'load_model', lambda node: model)
    node = estimation.StateEstimationNode({
        'site_names': ['ee'], 'motor_ids': [10], 'jacobian_thread': False,
        'motor_control_mode': 'qp_velocity_host', 'init_from': 'keyframe',
        'velocity_feedback_motors': [
            {'motor_id': 10, 'tendon_name': 'tendon1', 'radius_m': .04, 'sign': 1,
             'kt_output_nm_a': 1.27, 'kd_nm_s_rad': .1, 'speed_limit_rad_s': 1.},
        ],
    })
    node.now = lambda: 100.
    node.last_stamp = 0.
    node.measurement_wall_time = 100.
    node.x[:] = [0., .3, -.4]
    node._on_command({
        'motor_id': 10, 'stamp': 100., 'control_mode': 'qp_velocity_host',
        'qp_current_a': -.7, 'app_ctrl': -1.1, 'current': -1.08, 'fault_code': 0,
    })
    node.n_updates = 1
    messages = []
    node.state_pub.publish = messages.append
    return node, messages


def measurement(stamp=.01):
    return {'stamp': stamp, 'published_at': 100., 'data': [0., .3, 0.]}


def test_ekf_excludes_friction_compensation_from_prediction(prior_estimator):
    node, _ = prior_estimator
    node._on_command({
        'motor_id': 10, 'stamp': 100., 'control_mode': 'qp_velocity_host',
        'qp_current_a': -.7, 'app_ctrl': -1.1, 'current': -1.08, 'fault_code': 0,
    })
    np.testing.assert_array_equal(node.app_u, [-.7])
    node.predict(10)
    node._capture_velocity_prediction(10, .01, .001)
    np.testing.assert_array_equal(node.velocity_prediction['prediction_input_current_a'], [-.7])


def test_changed_compensation_does_not_change_prediction_or_covariance(prior_estimator, monkeypatch):
    node, _ = prior_estimator
    initial_state = state_snapshot(node.model, node.data)
    x, p = node.x.copy(), node.P.copy()
    linearization_inputs = []
    linearize = node.discrete_jacobian

    def observe_linearization(*args):
        linearization_inputs.append(args[1].copy())
        return linearize(*args)

    monkeypatch.setattr(node, 'discrete_jacobian', observe_linearization)
    results = []
    for total in (-.8, -1.4):
        restore_snapshot(node.model, node.data, initial_state)
        node.x, node.P = x.copy(), p.copy()
        node._on_command({
            'motor_id': 10, 'stamp': 100., 'control_mode': 'qp_velocity_host',
            'qp_current_a': -.7, 'app_ctrl': total, 'current': total + .02, 'fault_code': 0,
        })
        node.request_jacobian()
        node.predict(10)
        node._capture_velocity_prediction(10, .01, .001)
        results.append((node.x.copy(), node.P.copy(),
                        node.velocity_prediction['tendon_velocity_m_s'].copy()))
    for first, second in zip(*results):
        np.testing.assert_array_equal(first, second)
    for current in linearization_inputs:
        np.testing.assert_array_equal(current, [-.7])


def test_feedback_mode_uses_nominal_input_with_qp_rollout(make_node):
    node = make_node(motor_control_mode='qp_velocity_host', velocity_reference_source='qp_rollout')
    node.now = lambda: 100.
    node._on_command({'motor_id': 1, 'stamp': 100., 'control_mode': 'qp_velocity_host',
                       'qp_current_a': -.3, 'app_ctrl': -.8, 'fault_code': 0})
    np.testing.assert_array_equal(node.app_u, [-.3])


def test_nominal_controller_keeps_existing_applied_input(make_node):
    node = make_node()
    node._on_command({'motor_id': 1, 'app_ctrl': -.8, 'qp_current_a': -.3})
    np.testing.assert_array_equal(node.app_u, [-.8])


@pytest.mark.parametrize('change', [
    {'qp_current_a': None}, {'qp_current_a': float('nan')}, {'stamp': 99.},
    {'fault_code': 5}, {'control_mode': 'qp_current'},
])
def test_bad_nominal_metadata_blocks_estimates(prior_estimator, change):
    node, messages = prior_estimator
    msg = {'motor_id': 10, 'stamp': 100., 'control_mode': 'qp_velocity_host',
           'qp_current_a': -.7, 'app_ctrl': -1.1, 'fault_code': 0}
    msg.update(change)
    node._on_command(msg)
    node._on_measurement(measurement())
    assert not messages
    np.testing.assert_array_equal(node.app_u, [-.7])
    assert np.isnan(node._nominal_input_stamp[0])


def test_missing_nominal_field_never_falls_back_to_total(prior_estimator):
    node, messages = prior_estimator
    node._on_command({'motor_id': 10, 'stamp': 100., 'control_mode': 'qp_velocity_host',
                       'app_ctrl': -1.1, 'fault_code': 0})
    node._on_measurement(measurement())
    assert not messages
    np.testing.assert_array_equal(node.app_u, [-.7])


def test_stale_nominal_input_blocks_prediction(prior_estimator):
    node, messages = prior_estimator
    node.now = lambda: 100.2
    node._on_measurement(dict(measurement(), published_at=100.2))
    assert not messages


def test_capture_prior_before_correction_and_no_extra_steps(prior_estimator, monkeypatch):
    node, messages = prior_estimator
    calls = []
    step = estimation.step_current

    def counted_step(*args):
        calls.append(args[3])
        step(*args)

    monkeypatch.setattr(estimation, 'step_current', counted_step)
    prior = {}

    def correct(*args):
        prior.update(copy.deepcopy(node.velocity_prediction))
        np.testing.assert_allclose(prior['tendon_velocity_m_s'], node.x[1:2])
        node.x[1] = -.9
        node.x[2] = .2
        node.have_estimate = True

    monkeypatch.setattr(node, 'update', correct)
    node._on_measurement(measurement())
    assert sum(calls) == 10
    assert len(messages) == 1
    msg = messages[0]
    assert msg['dq'][0] == -.9
    assert msg['velocity_prediction']['tendon_velocity_m_s'][0] != -.9
    assert prior['reference_source'] == 'ekf_prior'
    assert prior['reference_stamp'] == msg['stamp'] == .01
    assert prior['reference_wall_time'] == 100.
    assert prior['prediction_duration_s'] == pytest.approx(.01)
    np.testing.assert_array_equal(prior['prediction_input_current_a'], [-.7])
    # Later correction/input updates cannot mutate the stored reference.
    node.app_u[:] = -.1
    node.data.qvel[:] = 2.
    np.testing.assert_array_equal(msg['velocity_prediction']['tendon_velocity_m_s'],
                                  prior['tendon_velocity_m_s'])
    np.testing.assert_array_equal(msg['velocity_prediction']['prediction_input_current_a'], [-.7])


def test_capture_refreshes_final_velocity_without_mutation(prior_estimator, monkeypatch):
    node, _ = prior_estimator
    node.predict(10)
    before = state_snapshot(node.model, node.data)['state'].copy()
    x, p = node.x.copy(), node.P.copy()
    expected = mujoco.MjData(node.model)
    mujoco.mj_copyData(expected, node.model, node.data)
    stale = expected.ten_velocity.copy()
    mujoco.mj_forward(node.model, expected)
    assert not np.allclose(stale, expected.ten_velocity)

    def forbid_step(*args):
        pytest.fail('Extracting EKF reference must not integrate')

    monkeypatch.setattr(mujoco, 'mj_step', forbid_step)
    node._capture_velocity_prediction(10, .01, .001)
    np.testing.assert_allclose(node.velocity_prediction['tendon_velocity_m_s'], expected.ten_velocity)
    np.testing.assert_array_equal(state_snapshot(node.model, node.data)['state'], before)
    np.testing.assert_array_equal(node.x, x)
    np.testing.assert_array_equal(node.P, p)


def test_zero_step_startup_has_no_velocity_reference(prior_estimator, monkeypatch):
    node, messages = prior_estimator
    node.last_stamp = None
    node.velocity_prediction = {'old': 'must not be reused'}
    monkeypatch.setattr(node, 'update', lambda *args: setattr(node, 'have_estimate', True))
    node._on_measurement(measurement())
    assert messages[0]['velocity_prediction'] is None
    assert messages[0]['is_valid']


def test_numerical_reset_does_not_publish_prior(prior_estimator, monkeypatch):
    node, messages = prior_estimator
    monkeypatch.setattr(estimation, 'step_current',
                        lambda model, data, *args: mujoco.mj_resetData(model, data))
    node._on_measurement(measurement())
    assert not messages
    assert node.velocity_prediction is None


def test_failed_correction_does_not_publish_captured_prior(prior_estimator, monkeypatch):
    node, messages = prior_estimator

    def fail_update(*args):
        assert node.velocity_prediction is not None
        raise ValueError('test correction failure')

    monkeypatch.setattr(node, 'update', fail_update)
    node._on_measurement(measurement())
    assert not messages


def state_message(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    prior = command()['prediction']
    prior['motor_ids'] = [2, 10, 1]
    return {
        'stamp': .1, 'published_at': 100., 'measurement_age_s': 0.,
        'prediction_state': state_snapshot(model, data),
        'q': data.qpos.copy(), 'dq': data.qvel.copy(), 'is_valid': True,
        'velocity_prediction': prior,
    }


def test_controller_reuses_prior_without_rollout(make_controller, monkeypatch):
    node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.)
    clock = [100.]
    node.now = lambda: clock[0]
    node.prediction_budget_s = 1.
    assert node.velocity_reference_source == 'ekf_prior'
    assert node.predictor is None
    msg = state_message(node.model)

    def forbid_step(*args):
        pytest.fail('Controller must not run an additional prediction')

    monkeypatch.setattr(mujoco, 'mj_step', forbid_step)
    node._on_state(msg)
    node._tick()
    clock[0] += .01
    node._tick()
    commands = [m for t, m in messages if t == MOTOR_COMMAND]
    assert len(commands) == 2
    for cmd in commands:
        np.testing.assert_allclose(cmd['velocity_reference'], [-.5, .5, 0.])
        np.testing.assert_allclose(cmd['prediction']['prediction_input_current_a'], [-.2, -.4, -.3])
        assert cmd['prediction']['reference_stamp'] == .1
        assert cmd['prediction']['reference_wall_time'] == 100.
        assert cmd['prediction']['reference_source'] == 'ekf_prior'
    assert commands[1]['stamp'] > commands[0]['stamp']
    clock[0] += .2
    node._tick()
    assert len([m for t, m in messages if t == MOTOR_COMMAND]) == 2


@pytest.mark.parametrize('change', [
    {'reference_wall_time': 99.}, {'reference_stamp': .09}, {'prediction_steps': 0},
    {'prediction_steps': float('inf')},
    {'reference_source': 'qp_rollout'}, {'tendon_velocity_m_s': [0., np.nan, 0.]},
    {'prediction_input_kind': 'applied_total'},
    {'tendon_names': ['tendon1', 'tendon2', 'tendon3']},
])
def test_controller_rejects_invalid_reference(make_controller, change):
    node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.)
    node.now = lambda: 100.
    msg = state_message(node.model)
    msg['velocity_prediction'].update(change)
    node._on_state(msg)
    node._tick()
    assert not node.have_state
    assert not any(t == MOTOR_COMMAND for t, _ in messages)


def test_controller_waits_for_first_prior(make_controller):
    node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.)
    node.now = lambda: 100.
    msg = state_message(node.model)
    msg['velocity_prediction'] = None
    node._on_state(msg)
    node._tick()
    assert not node.have_state
    assert not any(t == MOTOR_COMMAND for t, _ in messages)


@pytest.mark.parametrize('field,value', [
    ('reference_source', 'qp_rollout'), ('reference_wall_time', 99.),
    ('prediction_steps', 0), ('prediction_duration_s', -1.),
])
def test_hardware_rejects_bad_reference_with_fresh_command(hardware, field, value):
    node, _, _ = hardware
    msg = command()
    msg['prediction'][field] = value
    node._on_command(msg)
    assert node._velocity_fault
    np.testing.assert_array_equal(node.commanded_current, 0.)


def test_hardware_logs_reference_input_and_age(hardware):
    node, messages, clock = hardware
    clock[0] += .01
    node._on_command(command())
    diag = [m for t, m in messages if t == DIAGNOSTICS][-1]
    assert diag['reference_source'] == 'ekf_prior'
    assert diag['reference_age_s'] == pytest.approx(.01)
    np.testing.assert_array_equal(diag['prediction_input_current_a'], [-.2, -.4, -.3])
    assert diag['prediction_input_kind'] == 'qp_nominal'


def test_hardware_reports_nominal_but_plant_receives_total(hardware):
    node, messages, _ = hardware
    node.velocity_feedback.kd[:] = 10.
    node._on_command(command())
    states = [m for t, m in messages if t == MOTOR_STATE][-3:]
    np.testing.assert_array_equal([m['qp_current_a'] for m in states], [-.4, -.5, -.6])
    np.testing.assert_array_equal([m['app_ctrl'] for m in states], [-1.9, -.1, -.6])
    plant = VirtualMeasurementNode.__new__(VirtualMeasurementNode)
    plant.motor_ids = [2, 10, 1]
    plant.ctrl_u = np.zeros(3)
    for state in states:
        plant._on_motor_state(state)
    np.testing.assert_array_equal(plant.ctrl_u, [-.6, -1.9, -.1])
    node._stop_velocity('test stop')
    stopped = [m for t, m in messages if t == MOTOR_STATE][-3:]
    assert all(m['qp_current_a'] == 0. and m['app_ctrl'] == 0. for m in stopped)


def test_can_feedback_preserves_nominal_current_at_capture(hardware):
    node, messages, _ = hardware
    node.qp_current[:] = [-.4, -.5, -.6]
    node.commanded_current[:] = [-.8, -.9, -1.]
    feedback = dict(position=0., speed=0., current=-.79, error_code=0, received_at=100.)
    node.controller = SimpleNamespace(get_feedback_all_motors=lambda **kw: [feedback]*3)
    calls = iter([True, False])
    node.ok = lambda: next(calls)
    node._feedback_loop()
    node.qp_current[:] = -.2
    node._publish_feedback()
    states = [m for t, m in messages if t == MOTOR_STATE][-3:]
    np.testing.assert_array_equal([m['qp_current_a'] for m in states], [-.4, -.5, -.6])
    np.testing.assert_array_equal([m['app_ctrl'] for m in states], [-.8, -.9, -1.])
    assert all(m['current'] == -.79 for m in states)


def test_rollout_source_still_reaches_hardware(hardware, make_controller):
    motor_node, messages, _ = hardware
    motor_node.velocity_reference_source = 'qp_rollout'
    controller, _ = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.,
                                    velocity_reference_source='qp_rollout')
    controller.now = lambda: 100.
    controller.prediction_budget_s = 1.
    controller._on_state(state_message(controller.model))
    controller._tick()
    msg = [m for t, m in messages if t == MOTOR_COMMAND][-1]
    motor_node._on_command(msg)
    assert motor_node._velocity_fault is None
    diag = [m for t, m in messages if t == DIAGNOSTICS][-1]
    assert diag['reference_source'] == 'qp_rollout'
    np.testing.assert_array_equal(diag['prediction_input_current_a'], msg['u'])


def test_reference_source_validation():
    with pytest.raises(ValueError, match='velocity_reference_source'):
        reference_source(type('Params', (), {'declare_parameter': lambda *args: 'invalid'})())


def test_controller_timing_comparison(make_controller):
    results = {}
    for source in ('ekf_prior', 'qp_rollout'):
        node, messages = make_controller(motor_control_mode='qp_velocity_host', start_delay_s=0.,
                                        velocity_reference_source=source, rate_hz=200.)
        node.now = lambda: 100.
        data = mujoco.MjData(node.model)
        rest_state(node, node.model, data)
        messages.clear()
        for i in range(60):
            data.qvel[:] = .01 * np.sin(i)
            msg = state_message(node.model)
            msg.update(prediction_state=state_snapshot(node.model, data),
                       q=data.qpos.copy(), dq=data.qvel.copy())
            node._on_state(msg)
            node._tick()
        records = [m['prediction'] if t == MOTOR_COMMAND else m for t, m in messages
                   if t in (MOTOR_COMMAND, DIAGNOSTICS)]
        assert len(records) == 60
        durations = np.array([r['outer_compute_s'] for r in records])
        results[source] = np.percentile(durations, [50, 95, 100]) * 1000
        if source == 'ekf_prior':
            assert np.percentile(durations, 95) < 1. / node.rate_hz
        assert all(m['prediction']['outer_compute_s'] <= 1. / node.rate_hz
                   for t, m in messages if t == MOTOR_COMMAND)
    print('\nController ms p50/p95/max:', results)
    assert results['ekf_prior'][0] < results['qp_rollout'][0]


def test_extraction_cost(prior_estimator):
    node, _ = prior_estimator
    node.predict(10)
    timings = []
    for _ in range(100):
        started = time.perf_counter()
        node._capture_velocity_prediction(10, .01, .001)
        timings.append(time.perf_counter() - started)
    print('\nSmall-model reference extraction p95 ms:', np.percentile(timings, 95) * 1000)
