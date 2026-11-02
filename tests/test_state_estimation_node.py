"""Offline EKF covariance checks with small MuJoCo models."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.common import step_current  # noqa: E402
from spirob_zmq.core import Logger, Node  # noqa: E402
from spirob_zmq.nodes import state_estimation_node as estimation  # noqa: E402


@pytest.fixture
def make_node(monkeypatch):
    def init_node(self, name, params=None):
        self.name = name
        self._params = params or {}
        self._profile = {}
        self._logger = Logger(name)

    monkeypatch.setattr(Node, '__init__', init_node)
    monkeypatch.setattr(Node, 'create_subscription', lambda *args: None)
    monkeypatch.setattr(
        Node, 'create_publisher',
        lambda *args: SimpleNamespace(publish=lambda msg: None),
    )

    def make(activation=False, **params):
        actuator = (
            '<general joint="j1" dyntype="filter" dynprm="0.05"/>'
            if activation else '<motor joint="j1"/>'
        )
        model = mujoco.MjModel.from_xml_string(f"""
            <mujoco>
              <option timestep="0.01" integrator="implicitfast" gravity="0 0 0"/>
              <worldbody>
                <body>
                  <joint name="j1" stiffness="3" damping="0.1"/>
                  <geom type="capsule" size="0.02" fromto="0 0 0 0.4 0 0" mass="1"/>
                  <body pos="0.4 0 0">
                    <joint name="j2" stiffness="2" damping="0.1"/>
                    <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" mass="1"/>
                    <site name="ee" pos="0.3 0 0"/>
                  </body>
                </body>
              </worldbody>
              <actuator>{actuator}</actuator>
              <keyframe><key name="init" qpos="0.2 -0.1"/></keyframe>
            </mujoco>
        """)
        monkeypatch.setattr(estimation, 'load_model', lambda node: model)
        return estimation.StateEstimationNode({
            'site_names': ['ee'], 'motor_ids': [1], 'jacobian_thread': False,
            **params,
        })

    return make


@pytest.mark.parametrize('init_from', ['keyframe', 'measurement'])
def test_initial_and_reset_covariance(make_node, init_from):
    node = make_node(init_from=init_from, init_q_std=0.2, init_dq_std=0.003)
    expected = np.diag([0.2 ** 2] * 2 + [0.003 ** 2] * 2)
    np.testing.assert_array_equal(node.P, expected)

    if init_from == 'measurement':
        node.P[:] = 99
        node._on_initial_state({'q': [0.1, -0.2]})
        assert node.initialized
        np.testing.assert_array_equal(node.P, expected)

    node.P[:] = np.inf
    node._check_covariance('test')
    np.testing.assert_array_equal(node.P, expected)


@pytest.mark.parametrize('std', [0.0, 0.05, 0.1])
def test_input_covariance(make_node, std):
    node = make_node(process_current_std=std)
    B = np.array([[1., 2.], [-3., 1.], [2., 4.], [0., -1.]])
    expected = B @ (std ** 2 * np.eye(2)) @ B.T
    Q = node._process_covariance(B)
    assert Q.shape == (4, 4)
    np.testing.assert_allclose(Q, expected, rtol=1e-14, atol=1e-17)
    np.testing.assert_array_equal(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() >= -1e-15


@pytest.mark.parametrize('activation', [False, True])
def test_input_jacobian_matches_finite_difference(make_node, activation):
    node = make_node(activation=activation)
    u = np.array([0.1])
    A, B = node.discrete_jacobian(node.x, u)
    assert A.shape == (node.nx, node.nx)
    assert B.shape == (node.nx, 1)
    states = []
    eps = 1e-5
    for sign in (-1, 1):
        data = mujoco.MjData(node.model)
        node._set_state(data, node.x)
        data.ctrl[:] = u + sign * eps
        mujoco.mj_forward(node.model, data)
        mujoco.mj_step(node.model, data)
        states.append(node._get_state(data))
    np.testing.assert_allclose(B[:, 0], (states[1] - states[0]) / (2 * eps),
                               rtol=1e-5, atol=1e-8)


def test_activation_block_is_preserved(make_node):
    node = make_node(activation=True, init_act_std=0.07)
    assert node.na == 1
    assert node.P.shape == (5, 5)
    assert node.P[-1, -1] == 0.07 ** 2
    _, B = node._step_jacobians
    Q = node._process_covariance(B)
    assert Q[-1, -1] > 0
    node.Q[:] = 42
    node._reset_covariance()
    assert node.P[-1, -1] == 0.07 ** 2


@pytest.mark.parametrize('n', [1, 4])
def test_prediction_accumulates_held_input_noise_and_preserves_state(make_node, monkeypatch, n):
    node = make_node()
    node.app_u[:] = 0.1
    F = np.eye(node.nx)
    F[:node.nq, node.nq:] = node.h * np.eye(node.nv)
    B = np.arange(1., node.nx + 1).reshape(-1, 1)
    node._step_jacobians = F, B
    monkeypatch.setattr(node, 'request_jacobian', lambda: None)

    batched_data = mujoco.MjData(node.model)
    mujoco.mj_copyData(batched_data, node.model, node.data)
    step_current(node.model, batched_data, node.app_u, n, node.vlimit)

    Fn = np.linalg.matrix_power(F, n)
    G = sum(np.linalg.matrix_power(F, k) @ B for k in range(n))
    expected_Q = G @ (0.05 ** 2 * np.eye(1)) @ G.T
    expected_P = Fn @ node.P @ Fn.T + expected_Q

    node.predict(n)
    np.testing.assert_allclose(node.P, expected_P, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(node.Q, expected_Q, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(node.x, node._get_state(batched_data))
    np.testing.assert_array_equal(node.P, node.P.T)
    assert np.linalg.eigvalsh(node.P).min() >= -1e-18
    assert node.prediction_diagnostics['n_steps'] == n


def test_zero_steps_do_not_change_state_or_covariance(make_node):
    node = make_node()
    x, P, Q = node.x.copy(), node.P.copy(), node.Q.copy()
    node.predict(0)
    np.testing.assert_array_equal(node.x, x)
    np.testing.assert_array_equal(node.P, P)
    np.testing.assert_array_equal(node.Q, Q)
    assert node.prediction_diagnostics == {'n_steps': 0}


@pytest.mark.parametrize('name', [
    'process_current_std', 'init_act_std', 'init_q_std', 'init_dq_std',
    'max_joint_q_correction',
])
@pytest.mark.parametrize('value', [-1.0, float('nan'), float('inf')])
def test_invalid_noise_parameters_fail_explicitly(make_node, name, value):
    with pytest.raises(ValueError, match=name):
        make_node(**{name: value})


@pytest.mark.parametrize('raw_q', [[0.6, -0.4], [0.1, -0.3], [0.2, -0.2], [0., 0.]])
def test_per_joint_cap_and_joseph_covariance(make_node, monkeypatch, capsys, raw_q):
    node = make_node()
    x, P = node.x.copy(), node.P.copy()
    mujoco.mj_kinematics(node.model, node.data)
    mujoco.mj_comPos(node.model, node.data)
    z = node.data.site_xpos[node.site_ids].reshape(-1).copy()
    y = np.array([1., 0., 0.])
    K = np.zeros((node.nx, 3))
    K[:, 0] = [*raw_q, 0.4, -0.5]
    monkeypatch.setattr(np.linalg, 'solve', lambda S, rhs: K.T.copy())
    jac = np.zeros((3, node.nv))
    mujoco.mj_jacSite(node.model, node.data, jac, None, node.site_ids[0])
    H = np.zeros((3, node.nx))
    H[:, :node.nq] = jac
    applied_K = K.copy()
    applied_K[:node.nq, 0] = np.clip(raw_q, -0.2, 0.2)
    IKH = np.eye(node.nx) - applied_K @ H
    expected_P = IKH @ P @ IKH.T + applied_K @ node.R @ applied_K.T

    node.update(z + y, node.site_ids)
    np.testing.assert_allclose(node.x - x, applied_K @ y, atol=1e-16)
    np.testing.assert_allclose(node.P, expected_P, atol=1e-16)
    np.testing.assert_array_equal(node.P, node.P.T)
    count = int(np.count_nonzero(np.abs(raw_q) > 0.2))
    assert node.last_diagnostics['correction_q_capped_count'] == count
    assert (
        f'EKF update 1: joint corrections capped for {count}/2 joints (limit 0.2 rad)'
        in capsys.readouterr().err
    )
    published = []
    node.diagnostics_pub.publish = published.append
    node.publish(123.)
    assert published[0]['stamp'] == 123.
    assert published[0]['correction_q_capped_count'] == count
    node.update(z + y, node.site_ids)
    assert 'EKF update 2: joint corrections capped for' in capsys.readouterr().err
