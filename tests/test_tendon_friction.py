"""Offline capstan geometry, dynamics, and consumer integration checks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.common import (  # noqa: E402
    INITIAL_STATE, MOTOR_COMMAND, MOTOR_STATE, ProfileParams, VoltageLimit, actuator_moment, forward_actuation,
    load_model, load_tendon_friction, rest_state, step_current,
)
from spirob_zmq.core import Logger, Node  # noqa: E402
from spirob_zmq.nodes import state_estimation_node as estimation  # noqa: E402
from spirob_zmq.nodes import get_state_node, virtual_measurement_node  # noqa: E402
from spirob_zmq import replay_hw  # noqa: E402
from spirob_zmq.tendon_friction import TendonFriction  # noqa: E402
from test_tracking_log import make_controller  # noqa: E402, F401


@pytest.fixture
def routed_model():
    return mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
          <worldbody>
            <site name="base" pos="0 0 0"/>
            <site name="guide1" pos="1 0 0"/>
            <site name="guide2" pos="1 1 0"/>
            <body pos="2 1 0">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.01" mass="1"/>
              <site name="ee"/>
            </body>
          </worldbody>
          <tendon><spatial name="t">
            <site site="base"/><site site="guide1"/><site site="guide2"/><site site="ee"/>
          </spatial></tendon>
          <actuator><motor tendon="t" gear="2" ctrlrange="-5 0"/></actuator>
          <keyframe><key name="init" qpos="0"/></keyframe>
        </mujoco>
    """)


@pytest.mark.parametrize('velocity', [-0.1, 0.0, 0.1])
def test_two_guide_analytic_ratios_and_force(routed_model, velocity):
    m = routed_model
    d = mujoco.MjData(m)
    d.qvel[:] = velocity
    d.ctrl[:] = -0.5
    mujoco.mj_forward(m, d)
    friction = TendonFriction(m, 0.2)
    jac, angles, slip = friction.span_kinematics(d, 0)
    np.testing.assert_allclose(jac, [[0.], [0.], [1.]])
    np.testing.assert_allclose(angles, [np.pi / 2, np.pi / 2])
    np.testing.assert_allclose(slip, [velocity, velocity])
    expected = np.exp(0.2 * np.array([0., np.pi / 2, np.pi]) * np.sign(velocity))
    np.testing.assert_allclose(friction.tension_ratios(angles, slip, 0), expected)
    effective = forward_actuation(m, d, friction)
    np.testing.assert_allclose(effective, [[2 * expected[-1]]])
    np.testing.assert_allclose(d.qacc, [-expected[-1]])
    np.testing.assert_array_equal(d.qfrc_applied, [0.])


def test_straight_routing_has_no_attenuation(routed_model):
    m = routed_model
    m.site_pos[m.site('guide2').id] = [1.5, 0, 0]
    m.body_pos[m.site('ee').bodyid] = [2, 0, 0]
    d = mujoco.MjData(m)
    d.qvel[:] = -0.1
    mujoco.mj_forward(m, d)
    f = TendonFriction(m, 0.3)
    jac, angles, slip = f.span_kinematics(d, 0)
    np.testing.assert_array_equal(angles, 0.)
    np.testing.assert_array_equal(f.tension_ratios(angles, slip, 0), np.ones(len(jac)))
    np.testing.assert_allclose(f.moment(d, actuator_moment(m, d)), actuator_moment(m, d))


def test_sliding_correction_is_dissipative_on_real_routing():
    m = load_model(ProfileParams())
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    friction = TendonFriction(m, [0.1, 0.15, 0.2])
    rng = np.random.default_rng(12)
    for _ in range(5):
        d.qvel[:] = rng.normal(0, 0.2, m.nv)
        d.ctrl[:] = [-0.3, -0.4, -0.5]
        mujoco.mj_forward(m, d)
        nominal = actuator_moment(m, d)
        effective = friction.moment(d, nominal)
        assert friction.correction(d, nominal, effective) @ d.qvel <= 1e-12
        for i in range(m.nu):
            jac, _, _ = friction.span_kinematics(d, i)
            np.testing.assert_allclose(m.actuator_gear[i, 0] * jac.sum(axis=0),
                                       nominal[i], atol=1e-13)
    # Independently differentiate every free-span length, not just the total tendon length.
    jac, _, _ = friction.span_kinematics(d, 0)
    q = d.qpos.copy()
    sites = friction.routes[0]
    eps = 1e-6
    fd = np.empty_like(jac)
    for j in range(m.nv):
        lengths = []
        for sign in (-1, 1):
            d.qpos[:] = q
            d.qpos[j] += sign * eps
            mujoco.mj_kinematics(m, d)
            lengths.append(np.linalg.norm(np.diff(d.site_xpos[sites], axis=0), axis=1))
        fd[:, j] = (lengths[1] - lengths[0]) / (2 * eps)
    np.testing.assert_allclose(jac, fd, atol=1e-9)


def test_local_reversals_and_smoothing(routed_model):
    friction = TendonFriction(routed_model, 0.2, velocity_eps=0.01)
    angles = np.full(2, np.pi / 2)
    ratios = friction.tension_ratios(angles, np.array([0.5, -0.5]), 0)
    np.testing.assert_allclose(ratios, [1, np.exp(0.2 * np.pi / 2), 1])
    near_zero = friction.tension_ratios(angles, np.array([1e-9, -1e-9]), 0)
    np.testing.assert_allclose(near_zero, 1., atol=4e-8)


def test_per_actuator_coefficient_leaves_other_moments_unchanged():
    m = load_model(ProfileParams())
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = -0.1
    mujoco.mj_forward(m, d)
    nominal = actuator_moment(m, d)
    effective = TendonFriction(m, [0.1, 0., 0.]).moment(d, nominal)
    assert np.linalg.norm(effective[0] - nominal[0]) > 1e-5
    np.testing.assert_array_equal(effective[1:], nominal[1:])


@pytest.mark.parametrize('mu', [0.0, [0.0]])
def test_zero_coefficient_preserves_original_stepping(routed_model, mu):
    m = routed_model
    params = ProfileParams(tendon_friction_mu=mu)
    assert load_tendon_friction(params, m) is None
    a, b = mujoco.MjData(m), mujoco.MjData(m)
    a.qvel[:] = b.qvel[:] = 0.01
    current = np.array([-0.3])
    limit = VoltageLimit(params)
    for _ in range(4):
        b.ctrl[:] = limit.clip(current, b.actuator_velocity)
        mujoco.mj_step(m, b)
    step_current(m, a, current, 4, limit, load_tendon_friction(params, m))
    np.testing.assert_array_equal(a.qpos, b.qpos)
    np.testing.assert_array_equal(a.qvel, b.qvel)
    nominal = actuator_moment(m, a)
    np.testing.assert_array_equal(TendonFriction(m, mu).moment(a, nominal), nominal)


def test_step_matches_forward_force_and_preserves_external_force(routed_model):
    m = routed_model
    d = mujoco.MjData(m)
    d.qvel[:] = -0.1
    d.ctrl[:] = -0.5
    d.qfrc_applied[:] = 0.1
    f = TendonFriction(m, 0.2)
    forward_actuation(m, d, f)
    acceleration = d.qacc.copy()
    velocity = d.qvel.copy()
    position = d.qpos.copy()
    for _ in range(3):
        forward_actuation(m, d, f)
        np.testing.assert_allclose(d.qacc, acceleration)
        np.testing.assert_array_equal(d.qfrc_applied, [0.1])
    step_current(m, d, np.array([-0.5]), 1, VoltageLimit(ProfileParams()), f)
    np.testing.assert_allclose(d.qvel, velocity + m.opt.timestep * acceleration)
    np.testing.assert_allclose(d.qpos, position + m.opt.timestep * d.qvel)
    np.testing.assert_array_equal(d.qfrc_applied, [0.1])


@pytest.mark.parametrize('mu', [-0.1, np.nan, np.inf, [0.1, 0.2], [[0.1]]])
def test_invalid_coefficients(routed_model, mu):
    with pytest.raises(ValueError, match='tendon_friction_mu'):
        TendonFriction(routed_model, mu)


@pytest.mark.parametrize('eps', [0., -1., np.nan, np.inf])
def test_invalid_velocity_epsilon(routed_model, eps):
    with pytest.raises(ValueError, match='velocity_eps'):
        TendonFriction(routed_model, 0.1, eps)


def test_invalid_geometry_and_motor_modes_fail_explicitly(routed_model):
    m = routed_model
    f = TendonFriction(m, 0.1)
    d = mujoco.MjData(m)
    m.site_pos[m.site('guide1').id] = m.site_pos[m.site('base').id]
    mujoco.mj_forward(m, d)
    with pytest.raises(ValueError, match='Degenerate'):
        f.moment(d, actuator_moment(m, d))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    with pytest.raises(ValueError, match='RK4'):
        TendonFriction(m, 0.1)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    m.wrap_type[0] = mujoco.mjtWrap.mjWRAP_PULLEY
    with pytest.raises(ValueError, match='site-only'):
        TendonFriction(m, 0.1)
    m.actuator_gear[0, 0] = -1
    with pytest.raises(ValueError, match='positive-gear'):
        TendonFriction(m, 0.1)


def test_positive_force_and_overflow_fail_without_changing_external_force(routed_model):
    m = routed_model
    m.actuator_ctrlrange[0, 1] = 5
    d = mujoco.MjData(m)
    d.qfrc_applied[:] = 0.1
    f = TendonFriction(m, 0.1)
    with pytest.raises(ValueError, match='nonpositive'):
        step_current(m, d, np.array([0.1]), 1, VoltageLimit(ProfileParams()), f)
    np.testing.assert_array_equal(d.qfrc_applied, [0.1])
    f = TendonFriction(m, 1e4)
    with pytest.raises(ValueError, match='overflow'):
        f.tension_ratios(np.array([np.pi]), np.array([1.]), 0)


@pytest.mark.parametrize('centered', [False, True])
def test_ekf_linearizes_capstan_transition(routed_model, monkeypatch, centered):
    def init(self, name, params=None):
        self._params, self._profile = params or {}, {}
        self._logger = Logger(name)

    monkeypatch.setattr(Node, '__init__', init)
    monkeypatch.setattr(Node, 'create_subscription', lambda *args, **kwargs: None)
    monkeypatch.setattr(Node, 'create_publisher',
                        lambda *args: SimpleNamespace(publish=lambda msg: None))
    monkeypatch.setattr(estimation, 'load_model', lambda node: routed_model)
    node = estimation.StateEstimationNode({
        'site_names': ['ee'], 'motor_ids': [1], 'jacobian_thread': False,
        'tendon_friction_mu': 0.2, 'fd_centered': centered,
        'tendon_friction_velocity_eps': 0.01, 'settle_time': 0.,
    })
    x = np.array([0., -0.005])
    u = np.array([-0.5])
    A, B = node.discrete_jacobian(x, u)
    assert A.shape == (2, 2) and B.shape == (2, 1)
    eps = 1e-7
    combined = np.r_[x, u]
    expected = np.empty((2, 3))
    for j in range(3):
        states = []
        for sign in (-1, 1):
            values = combined.copy()
            values[j] += sign * eps
            d = mujoco.MjData(node.model)
            node._set_state(d, values[:2])
            step_current(node.model, d, values[2:], 1, node.vlimit, node.tendon_friction)
            states.append(node._get_state(d))
        expected[:, j] = (states[1] - states[0]) / (2 * eps)
    np.testing.assert_allclose(np.column_stack((A, B)), expected, rtol=1e-4, atol=1e-7)
    # At a fixed upper current bound, linearization must include current clipping.
    _, limited_B = node.discrete_jacobian(x, np.array([0.2]))
    np.testing.assert_array_equal(limited_B, 0.)
    node.x = x.copy()
    node.app_u[:] = u
    monkeypatch.setattr(node, 'request_jacobian', lambda: None)
    d = mujoco.MjData(node.model)
    node._set_state(d, x)
    step_current(node.model, d, u, 2, node.vlimit, node.tendon_friction)
    node.predict(2)
    np.testing.assert_allclose(node.x, node._get_state(d), atol=1e-14)
    node.vlimit.v_bus = 0.22
    _, voltage_limited_B = node.discrete_jacobian(x, u)
    np.testing.assert_array_equal(voltage_limited_B, 0.)


def test_controller_uses_distributed_moment_and_keeps_current_limits(make_controller):
    node, messages = make_controller(tendon_friction_mu=0.15, start_delay_s=0.,
                                     site_name='ee')
    node.now = lambda: 10.
    node._on_state({'stamp': 3., 'q': node.model.key_qpos[0].copy(),
                    'dq': np.full(node.model.nv, -0.05), 'is_valid': True})
    node._tick()
    nominal = actuator_moment(node.model, node.data)
    effective = node.tendon_friction.moment(node.data, nominal)
    assert np.linalg.norm(effective - nominal) > 1e-5
    pinv = np.linalg.pinv(effective.T * node.law.k_v, rcond=node.pinv_rcond)
    np.testing.assert_allclose(node._A[1:1 + node.model.nu, :node.model.nv],
                               pinv @ node._M)
    command = [msg for topic, msg in messages if topic == MOTOR_COMMAND][-1]
    u = np.array(command['u'])
    assert np.all(u <= 0.)
    assert np.all(u >= node.law.u_min)


def test_rest_state_uses_capstan_step(routed_model):
    params = ProfileParams(settle_time=0.005, tendon_friction_mu=0.2)
    m = routed_model
    m.actuator_ctrlrange[0, 1] = -0.1
    a, b = mujoco.MjData(m), mujoco.MjData(m)
    rest_state(params, m, a)
    step_current(m, b, np.zeros(1), 5, VoltageLimit(params), load_tendon_friction(params, m))
    np.testing.assert_allclose(a.qpos, b.qpos)
    np.testing.assert_allclose(a.qvel, b.qvel)
    assert a.time == 0.


@pytest.mark.parametrize('virtual', [False, True])
def test_plant_nodes_use_capstan_step(routed_model, monkeypatch, virtual):
    def init(self, name, params=None):
        self._params, self._profile = params or {}, {}
        self._logger = Logger(name)

    monkeypatch.setattr(Node, '__init__', init)
    monkeypatch.setattr(Node, 'create_subscription', lambda *args, **kwargs: None)
    monkeypatch.setattr(Node, 'create_timer', lambda *args: None)
    monkeypatch.setattr(Node, 'create_publisher',
                        lambda *args: SimpleNamespace(publish=lambda msg: None))
    module = virtual_measurement_node if virtual else get_state_node
    monkeypatch.setattr(module, 'load_model', lambda node: routed_model)
    if virtual:
        monkeypatch.setattr(
            virtual_measurement_node, 'MarkerModel',
            lambda *args: SimpleNamespace(initialize=lambda points: SimpleNamespace(
                q=np.zeros(1), rms=0., labels={'ee': 0})),
        )
    cls = module.VirtualMeasurementNode if virtual else module.GetStateNode
    node = cls({'site_names': ['ee'], 'motor_ids': [1], 'rate_hz': 1000.,
                'tendon_friction_mu': 0.2, 'noise_std': 0., 'seed': 1})
    node.data.qvel[:] = -0.1
    node.ctrl_u[:] = -0.5
    expected = mujoco.MjData(node.model)
    mujoco.mj_copyData(expected, node.model, node.data)
    step_current(node.model, expected, node.ctrl_u, 1, node.vlimit, node.tendon_friction)
    node._tick()
    np.testing.assert_allclose(node.data.qpos, expected.qpos)
    np.testing.assert_allclose(node.data.qvel, expected.qvel)


def test_replay_uses_capstan_step(routed_model, monkeypatch, tmp_path):
    monkeypatch.setattr(replay_hw, 'load_model', lambda params: routed_model)
    records = [
        {'topic': MOTOR_COMMAND, 'msg': {'stamp': 0., 'motor_ids': [1], 'u': [-0.5]}},
        {'topic': INITIAL_STATE, 'msg': {'stamp': 0., 'q': [0.]}},
    ]
    for t in (0., 0.01):
        records.append({'topic': MOTOR_STATE, 'msg': {
            'stamp': t, 'motor_id': 1, 'position': 0., 'current': -0.5,
        }})
    path = tmp_path / 'pull.jsonl'
    path.write_text(''.join(json.dumps(dict(record, t=record['msg']['stamp'])) + '\n'
                            for record in records))
    params = {'motor_ids': [1], 'r_spool': 0.04, 'tendon_friction_mu': 0.2}
    result = replay_hw.replay(path, overrides=params)
    baseline = replay_hw.replay(path, overrides=dict(params, tendon_friction_mu=0.))
    d = mujoco.MjData(routed_model)
    mujoco.mj_forward(routed_model, d)
    length0 = d.ten_length[0]
    step_current(routed_model, d, np.array([-0.5]), 10, VoltageLimit(ProfileParams()),
                 TendonFriction(routed_model, 0.2))
    assert result['sim'].shape == (10, 1)
    assert result['sim'][-1, 0] == pytest.approx((d.ten_length[0] - length0) / 0.04)
    assert abs(result['sim'][-1, 0]) < abs(baseline['sim'][-1, 0])
