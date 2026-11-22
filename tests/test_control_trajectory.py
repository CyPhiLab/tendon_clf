"""Circular, four-point and polygon reference checks with no motor communication."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_tracking_log import make_controller  # noqa: F401
from spirob_zmq.common import MOTOR_COMMAND, TARGET_STATE, TRACKING_STATE, actuator_moment
from spirob_zmq.ekf_report import export_tracking_csv, plot_tracking
from spirob_zmq.nodes import control_node


@pytest.mark.parametrize('speed', [0.002, 0.005, 0.01])
@pytest.mark.parametrize('sample_start', [0., 1790000000.])
def test_polygon_hold_two_laps_and_derivatives(make_controller, speed, sample_start):
    node, _ = make_controller(tracking='polygon', speed=speed, target_pos=None)
    points = np.array([[0., .25, .125], [.05, .225, .15],
                       [0., .2, .175], [-.05, .225, .15]])
    np.testing.assert_array_equal(node._reference(sample_start)[0], points[0])
    node.trajectory_start_stamp = sample_start
    np.testing.assert_array_equal(node._reference(sample_start - 1.)[0], points[0])
    elapsed = 0.
    epsilon = 1e-3
    for index in range(8):
        point = points[index % 4]
        edge = points[(index + 1) % 4] - point
        edge_time = np.linalg.norm(edge) / speed
        for offset in (epsilon, 2.5, 5. - epsilon):
            position, velocity, acceleration = node._reference(sample_start + elapsed + offset)
            np.testing.assert_array_equal(position, point)
            np.testing.assert_array_equal(velocity, np.zeros(3))
            np.testing.assert_array_equal(acceleration, np.zeros(3))
        for fraction in (0.001, .25, .5, .75, .999):
            stamp = sample_start + elapsed + 5. + fraction * edge_time
            position, velocity, acceleration = node._reference(stamp)
            np.testing.assert_allclose(position, point + fraction * edge, atol=1e-8)
            np.testing.assert_allclose(velocity, edge / edge_time, atol=1e-14)
            assert np.linalg.norm(velocity) == pytest.approx(speed)
            assert position[1] + position[2] == pytest.approx(.375)
            np.testing.assert_array_equal(acceleration, np.zeros(3))
            before, velocity_before, _ = node._reference(stamp - epsilon)
            after, velocity_after, _ = node._reference(stamp + epsilon)
            np.testing.assert_allclose((after - before) / (2. * epsilon), velocity, atol=1e-6)
            np.testing.assert_array_equal(velocity_after, velocity_before)
        elapsed += 5. + edge_time
        before = node._reference(sample_start + elapsed - epsilon)[0]
        after = node._reference(sample_start + elapsed + epsilon)[0]
        np.testing.assert_allclose(before, after, atol=speed * epsilon + 1e-8)
    for offset in (epsilon, 5., 1000.):
        position, velocity, acceleration = node._reference(sample_start + elapsed + offset)
        np.testing.assert_array_equal(position, points[0])
        np.testing.assert_array_equal(velocity, np.zeros(3))
        np.testing.assert_array_equal(acceleration, np.zeros(3))


@pytest.mark.parametrize('sample_start', [0., 1790000000.])
def test_polygon_active_hold_and_timestamped_targets(make_controller, monkeypatch, sample_start):
    node, messages = make_controller(tracking='polygon', start_delay_s=5.)
    assert node.speed == .005
    edge_time = np.sqrt(.05**2 + .025**2 + .025**2) / node.speed
    clock = [10.]
    node.now = lambda: clock[0]
    monkeypatch.setattr(control_node.daqp, 'solve',
                        lambda *args: (np.zeros(node.model.nv + 1), 0., 1, {}))
    node._on_state({'stamp': sample_start, 'q': node.model.key_qpos[0].copy(),
                    'dq': np.zeros(node.model.nv), 'is_valid': True})
    node._tick()
    assert not any(topic == MOTOR_COMMAND for topic, _ in messages)
    assert node.trajectory_start_stamp == sample_start + 5.
    for elapsed in (5., 7.5, 9.999, 5. + 8. * (5. + edge_time) + .01):
        clock[0] = 10. + elapsed
        messages.clear()
        node._tick()
        assert any(topic == MOTOR_COMMAND for topic, _ in messages)
        target = next(message for topic, message in messages if topic == TARGET_STATE)
        np.testing.assert_array_equal(target['position_m'], [0., .25, .125])
        np.testing.assert_array_equal(target['velocity_m_s'], np.zeros(3))
        np.testing.assert_array_equal(target['acceleration_m_s2'], np.zeros(3))
    clock[0] = 10. + 10. + edge_time / 2.
    messages.clear()
    node._tick()
    target = next(message for topic, message in messages if topic == TARGET_STATE)
    np.testing.assert_allclose(target['position_m'], [.025, .2375, .1375], atol=1e-8)
    np.testing.assert_allclose(target['velocity_m_s'], np.array([.05, -.025, .025]) / edge_time)
    node._publish_tracking(target['stamp'], np.array(target['position_m']), 'estimated')
    assert messages[-1][1]['error_norm_m'] == pytest.approx(0.)


def test_trajectory_points_and_derivatives(make_controller):
    node, _ = make_controller(tracking='trajectory')
    assert node.omega == 1.
    node.trajectory_start_stamp = 100.
    expected = [[0., .225, .125], [0., .25, .1],
                [0., .275, .125], [0., .25, .15], [0., .225, .125]]
    for t, point in zip(np.arange(5) * np.pi / 2, expected):
        p, v, a = node._reference(100. + t)
        np.testing.assert_allclose(p, point, atol=1e-14)
        assert p[0] == v[0] == a[0] == 0.
        assert ((p[1] - .25) / .025) ** 2 + ((p[2] - .125) / .025) ** 2 == pytest.approx(1.)
        eps = 1e-5
        # Use interior points so finite differences do not cross the start boundary.
        p1, v1, _ = node._reference(100. + t + .1 + eps)
        p0, v0, _ = node._reference(100. + t + .1 - eps)
        _, vm, am = node._reference(100. + t + .1)
        np.testing.assert_allclose((p1 - p0) / (2 * eps), vm, atol=1e-9)
        np.testing.assert_allclose((v1 - v0) / (2 * eps), am, atol=1e-9)


@pytest.mark.parametrize('omega', [0., -1., 2.])
def test_angular_speed(make_controller, omega):
    node, _ = make_controller(tracking='trajectory', omega=omega)
    node.trajectory_start_stamp = 0.
    p, v, a = node._reference(1.)
    np.testing.assert_allclose(p, [0., .25 - .025*np.cos(omega), .125 - .025*np.sin(omega)])
    np.testing.assert_allclose(v, omega * np.array([0., .025*np.sin(omega), -.025*np.cos(omega)]))
    np.testing.assert_allclose(a, omega**2 * np.array([0., .025*np.cos(omega), .025*np.sin(omega)]))
    eps = 1e-4
    p1, v1, _ = node._reference(1. + eps)
    p0, v0, _ = node._reference(1. - eps)
    np.testing.assert_allclose((p1 - p0) / (2 * eps), v, atol=1e-9)
    np.testing.assert_allclose((v1 - v0) / (2 * eps), a, atol=1e-9)
    np.testing.assert_allclose((p1 - 2 * p + p0) / eps**2, a, atol=1e-8)


@pytest.mark.parametrize('params', [
    {'tracking': 'unknown'}, {'omega': float('nan')}, {'omega': float('inf')},
    {'tracking': 'trajectory', 'task_dim': 6},
    {'tracking': 'four_point', 'task_dim': 6},
        {'tracking': 'polygon', 'task_dim': 6}, {'tracking': 'ellipse'},
        *[{'tracking': 'polygon', 'speed': value}
            for value in (0., -1., float('nan'), float('inf'))],
    *[{'tracking': 'four_point', 'four_point_hold_s': value}
      for value in (0., -1., float('nan'), float('inf'))],
    {'target_pos': None}, {'target_pos': [0., float('nan'), 0.]},
])
def test_invalid_trajectory_configuration(make_controller, params):
    with pytest.raises(ValueError):
        make_controller(**params)


def test_setpoint_reference_unchanged(make_controller):
    node, _ = make_controller()
    for stamp in (0., 1000.):
        p, v, a = node._reference(stamp)
        np.testing.assert_array_equal(p, [.1, .2, .3])
        np.testing.assert_array_equal(v, np.zeros(3))
        np.testing.assert_array_equal(a, np.zeros(3))


@pytest.mark.parametrize('tracking', ['trajectory', 'four_point'])
def test_trajectory_does_not_require_setpoint(make_controller, tracking):
    node, _ = make_controller(tracking=tracking, target_pos=None)
    np.testing.assert_allclose(node._reference(0.)[0], [0., .225, .125])


@pytest.mark.parametrize('start', [0., 1790000000.])
@pytest.mark.parametrize('hold_s', [10., 2.5])
@pytest.mark.parametrize('omega', [0., -1., 2.])
def test_four_point_holds_boundaries_and_repeats(make_controller, start, hold_s, omega):
    node, _ = make_controller(tracking='four_point', four_point_hold_s=hold_s, omega=omega)
    points = [[0., .225, .125], [0., .25, .1], [0., .275, .125], [0., .25, .15]]
    np.testing.assert_allclose(node._reference(start)[0], points[0])
    node.trajectory_start_stamp = start
    np.testing.assert_allclose(node._reference(start - 1.)[0], points[0])
    for index in range(12):
        for offset in (0., hold_s / 2, hold_s - 1e-4):
            p, v, a = node._reference(start + index * hold_s + offset)
            np.testing.assert_allclose(p, points[index % 4], atol=1e-14)
            np.testing.assert_array_equal(v, np.zeros(3))
            np.testing.assert_array_equal(a, np.zeros(3))


@pytest.mark.parametrize('sample_start', [0., 1790000000.])
@pytest.mark.parametrize('tracking', ['trajectory', 'four_point'])
def test_delay_phase_and_timestamped_tracking(make_controller, monkeypatch, tmp_path, sample_start, tracking):
    node, messages = make_controller(tracking=tracking, start_delay_s=2.)
    clock = [10.]
    node.now = lambda: clock[0]
    monkeypatch.setattr(control_node.daqp, 'solve',
                        lambda *args: (np.zeros(node.model.nv + 1), 0., 1, {}))
    node._tick()
    assert not messages
    q = node.model.key_qpos[0].copy()
    node._on_state({'stamp': sample_start, 'q': q, 'dq': np.zeros(node.model.nv), 'is_valid': True})
    node._tick()
    assert node.trajectory_start_stamp == sample_start + 2.
    assert not any(t == MOTOR_COMMAND for t, _ in messages)
    target = [m for t, m in messages if t == TARGET_STATE][-1]
    np.testing.assert_allclose(target['position_m'], [0., .225, .125])
    np.testing.assert_array_equal(target['velocity_m_s'], [0., 0., 0.])
    # Freeze the epoch before activation, even if a measurement is delivered
    # before the first active timer tick and its estimate arrives afterward.
    point_at_start = node._reference(sample_start + 2.01)[0]
    node._publish_tracking(sample_start + 2.01, point_at_start, 'vicon')
    early = messages[-1][1]
    clock[0] = 12.
    node._tick()
    assert node.trajectory_start_stamp == sample_start + 2.
    assert any(t == MOTOR_COMMAND for t, _ in messages)
    node._publish_tracking(sample_start + 2.01, point_at_start, 'estimated')
    assert messages[-1][1]['target_m'] == early['target_m']
    advance = node.four_point_hold_s if tracking == 'four_point' else .5
    clock[0] += advance
    node._tick()
    target = [m for t, m in messages if t == TARGET_STATE][-1]
    np.testing.assert_allclose(target['position_m'], node._reference(sample_start + 2. + advance)[0])
    if tracking == 'four_point':
        np.testing.assert_allclose(target['position_m'], [0., .25, .1])
        np.testing.assert_array_equal(target['velocity_m_s'], np.zeros(3))
        np.testing.assert_array_equal(target['acceleration_m_s2'], np.zeros(3))
    # Arriving late must not compare a sample with the target at reception time.
    stamp = sample_start + 2.25
    point = node._reference(stamp)[0]
    for source in ('vicon', 'estimated'):
        node._publish_tracking(stamp, point, source)
    samples = [m for t, m in messages if t == TRACKING_STATE][-2:]
    for m in samples:
        assert m['error_norm_m'] == pytest.approx(0.)
        np.testing.assert_allclose(m['target_m'], point)
    path = tmp_path / 'trajectory.jsonl'
    path.write_text(''.join(json.dumps({'t': i, 'topic': t, 'msg': m}) + '\n'
                            for i, (t, m) in enumerate(messages)))
    csv_path = export_tracking_csv(path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[-1]['error_norm_m']) == pytest.approx(0.)
    assert float(rows[-1]['target_y_m']) == pytest.approx(point[1])
    plot_tracking(path, tmp_path / 'trajectory.png')
    assert (tmp_path / 'trajectory.png').stat().st_size > 0


@pytest.mark.parametrize('mode', ['setpoint', 'trajectory'])
def test_qp_task_cost_and_clf_reference(make_controller, monkeypatch, mode):
    node, _ = make_controller(tracking=mode, start_delay_s=0.)
    node.now = lambda: 10.
    node._on_state({'stamp': 3., 'q': node.model.key_qpos[0].copy(),
                    'dq': np.full(node.model.nv, .01), 'is_valid': True})
    node.trajectory_start_stamp = 2.
    captured = {}

    def solve(H, g, A, b, *args):
        captured.update(H=H.copy(), g=g.copy(), A=A.copy(), b=b.copy())
        return np.zeros(node.model.nv + 1), 0., 1, {}

    monkeypatch.setattr(control_node.daqp, 'solve', solve)
    node._tick()
    m, d = node.model, node.data
    nv = m.nv
    J, Jdot, M = node._jac6[:3], node._jdot6[:3], node._M
    target, vel, acc = node._reference(3.)
    error = d.site_xpos[node.site_id] - target
    velocity_error = J @ d.qvel - vel
    task_const = Jdot @ d.qvel - acc + node.K * error + node.Kd * velocity_error
    B = actuator_moment(m, d).T * node.law.k_v
    pinv = np.linalg.pinv(B, rcond=node.pinv_rcond)
    bias = d.qfrc_bias - d.qfrc_passive
    if node.include_constraint_forces:
        bias -= d.qfrc_constraint
    P, p = pinv @ M, pinv @ bias
    N = np.eye(nv) - J.T @ np.linalg.solve(J @ J.T, J)
    # Finite difference of the intended objective independently checks its linear term.
    def objective(qdd):
        return (node.task_weight * np.linalg.norm(J @ qdd + task_const)**2
                + node.reg_u * np.linalg.norm(P @ qdd + p)**2
                + node.reg_null * np.linalg.norm(N @ qdd + node.null_gain * N @ d.qvel)**2
                + node.reg_qdd * np.linalg.norm(qdd)**2)
    gradient = []
    eps = 1e-4
    for i in range(nv):
        v = np.zeros(nv)
        v[i] = eps
        gradient.append((objective(v) - objective(-v)) / (2 * eps))
    np.testing.assert_allclose(captured['g'][:nv], gradient, rtol=1e-4, atol=1e-4)
    eta = np.concatenate([error, velocity_error])
    V = eta @ node.Pe @ eta
    expected_b = -V/node.e - eta @ node.FTPe_PeF @ eta - 2*eta @ node.PeG @ (Jdot @ d.qvel - acc)
    assert captured['b'][0] == pytest.approx(expected_b)


@pytest.mark.parametrize('mode', ['setpoint', 'trajectory', 'four_point', 'polygon'])
def test_real_solver_outputs_bounded_current(make_controller, mode):
    node, messages = make_controller(tracking=mode, start_delay_s=0.)
    node.now = lambda: 10.
    node._on_state({'stamp': 3., 'q': node.model.key_qpos[0].copy(),
                    'dq': np.zeros(node.model.nv), 'is_valid': True})
    if mode == 'polygon':
        node.trajectory_start_stamp = -3.
    node._tick()
    commands = [m for topic, m in messages if topic == MOTOR_COMMAND]
    assert len(commands) == 1
    current = np.asarray(commands[0]['u'])
    assert np.isfinite(current).all()
    assert np.all(current >= node.law.u_min - 1e-8)
    assert np.all(current <= node.law.u_max + 1e-8)


@pytest.mark.parametrize('tracking, flag, value', [
    ('trajectory', '--omega', '0.5'), ('four_point', '--omega', '0.5'),
    ('polygon', '--speed', '0.005'),
])
def test_launcher_tracking_arguments(monkeypatch, tracking, flag, value):
    from spirob_zmq import launch

    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 0, wait=lambda **kwargs: 0)

    monkeypatch.setattr(launch.subprocess, 'Popen', spawn)
    assert launch.main(['--mode', 'no_ekf', '--no-broker', '--no-record', '--headless',
                        '--tracking', tracking, flag, value,
                        '-p', f'control_node.{flag[2:]}=0.02']) == 0
    controller = next(command for command in commands if 'spirob_zmq.nodes.control_node' in command)
    assert f'tracking={tracking}' in controller
    assert controller[-2:] == ['-p', f'{flag[2:]}={value}']
    other = next(command for command in commands if 'spirob_zmq.nodes.get_state_node' in command)
    assert f'tracking={tracking}' not in other and f'{flag[2:]}={value}' not in other


@pytest.mark.parametrize('args', [
    ['--mode', 'teleop', '--tracking', 'trajectory'],
    ['--mode', 'teleop', '--tracking', 'four_point'],
    ['--mode', 'teleop', '--tracking', 'polygon'],
    ['--mode', 'teleop', '--speed', '0.005'],
    ['--tracking', 'ellipse'],
    *[['--speed', value] for value in ('0', '-1', 'nan', 'inf')],
    ['--omega', 'nan'], ['--omega', 'inf'], ['--tracking', 'unknown'],
])
def test_launcher_rejects_invalid_tracking_arguments(monkeypatch, args):
    from spirob_zmq import launch

    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('launched a process'))
    with pytest.raises(SystemExit) as error:
        launch.main(args)
    assert error.value.code == 2
