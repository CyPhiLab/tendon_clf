"""Startup and marker-tracking checks on synthetic Vicon data from the robot model.

    python tests/test_startup.py        (or pytest tests/)
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.common import ProfileParams, VoltageLimit, load_model, rest_state, step_current  # noqa: E402
from spirob_zmq.marker_tracking import MarkerTracker  # noqa: E402
from spirob_zmq.pose_init import MarkerModel, stationary_clusters, transform  # noqa: E402

PARAMS = ProfileParams()
NAMES = PARAMS.declare_parameter('site_names')
EXTRA = [f'marker_{k}_seg_{s}' for s in (11, 16) for k in (1, 2, 3)]   # physically present, not measured


def _model():
    model = load_model(PARAMS)
    data = mujoco.MjData(model)
    rest_state(PARAMS, model, data)
    return model, data


def _sites(model, data, names):
    mujoco.mj_kinematics(model, data)
    return np.array([data.site(n).xpos.copy() for n in names])


def _curl(model, data, current=(-0.7, 0.0, 0.0), duration=0.6, rate=100.0):
    """Drive one tendon from rest; marker trajectories (incl. distractors) at the Vicon rate."""
    vlimit = VoltageLimit(PARAMS)
    n_sub = int(round(1.0 / rate / model.opt.timestep))
    traj = []
    for _ in range(int(duration * rate)):
        step_current(model, data, np.array(current), n_sub, vlimit)
        traj.append(_sites(model, data, NAMES + EXTRA))
    return np.array(traj)


def _snapshot(true_pts, rng, noise=3e-4, ghosts=2):
    pts = true_pts + rng.normal(0, noise, true_pts.shape)
    pts = np.vstack([pts, rng.uniform(-0.3, 0.3, (ghosts, 3)) + [0, 0.2, 0.2]])
    perm = rng.permutation(len(pts))
    return pts[perm], perm


def _labels_ok(result, perm):
    return all(perm[result.labels[n]] == NAMES.index(n) for n in NAMES)


def test_marker_model_geometry():
    model, _ = _model()
    mm = MarkerModel(model, NAMES, model.key_qpos[0])
    assert mm.segments == ['2', '5', '8']
    sides = [1e3 * mm.ref_sides[s].mean() for s in mm.segments]
    assert np.allclose(sides, [60.6, 51.1, 43.6], atol=0.2), sides
    assert mm.anchor is not None


def test_initialize_from_rest_and_curled_pose():
    rng = np.random.default_rng(1)
    model, data = _model()
    mm = MarkerModel(model, NAMES, model.key_qpos[0])
    rest = _sites(model, data, NAMES + EXTRA)
    curled = _curl(model, data)[-1]
    far = np.linalg.norm(curled[:9] - rest[:9], axis=1).max()
    assert far > 0.05, f'curl too small for a meaningful test ({1e3 * far:.0f} mm)'
    for pts in (rest, curled):
        shuffled, perm = _snapshot(pts, rng)
        r = mm.initialize(shuffled)
        assert r is not None
        assert _labels_ok(r, perm)
        assert r.rms < 1.5e-3, r.rms


def test_calibrate_frame_at_rest_recovers_frame():
    rng = np.random.default_rng(2)
    model, data = _model()
    mm = MarkerModel(model, NAMES, model.key_qpos[0])
    q_rest = data.qpos.copy()
    rest = _sites(model, data, NAMES + EXTRA)
    angle = np.radians(8.0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_true = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    t_true = np.array([0.04, -0.03, 0.02])
    vicon = (rest - t_true) @ R_true                      # model = R_true @ vicon + t_true
    shuffled, perm = _snapshot(vicon, rng)
    R, t, res, labels = mm.calibrate_frame_at_rest(shuffled, q_rest)
    assert all(perm[labels[n]] == NAMES.index(n) for n in NAMES)
    err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert err < 1.0 and np.linalg.norm(t - t_true) < 3e-3, (err, t - t_true)
    # Pose init after the frame is applied
    r = mm.initialize(transform(shuffled, (R, t)))
    assert _labels_ok(r, perm) and r.rms < 2e-3


def test_stationary_clusters():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.2, 0.2, (9, 3))
    still = [pts + rng.normal(0, 2e-4, pts.shape) for _ in range(60)]
    centres, is_still = stationary_clusters(still)
    assert is_still and len(centres) == 9
    moving = [pts + [0.0, 1e-3 * k, 0.0] for k in range(60)]
    _, is_still = stationary_clusters(moving)
    assert not is_still


def test_tracker_fast_curl_with_dropouts_and_distractors():
    """Never mislabel; with the estimator's prediction (the node always has it) keep losses low."""
    model, data = _model()
    rest = _sites(model, data, NAMES)
    traj = _curl(model, data, current=(-1.2, 0.0, 0.0), duration=0.8)
    speed = np.linalg.norm(np.diff(traj[:, :9], axis=0), axis=2).max() * 100
    assert speed > 2.0, f'peak marker speed only {speed:.2f} m/s'
    for use_prediction in (False, True):
        rng = np.random.default_rng(4)
        tracker = MarkerTracker(NAMES, rest, 0.0)
        wrong = lost = visible_total = 0
        for k, frame in enumerate(traj):
            truth = frame[:9]
            visible = rng.random(len(frame)) > 0.1
            pts, _ = _snapshot(frame[visible], rng)
            pred = truth + rng.normal(0, 3e-3, truth.shape) if use_prediction else None
            out = tracker.update(pts, (k + 1) / 100.0, predicted=pred)
            ok = ~np.isnan(out[:, 0])
            wrong += np.sum(np.linalg.norm(out[ok] - truth[ok], axis=1) > 3e-3)
            lost += np.sum(~ok & visible[:9])
            visible_total += np.sum(visible[:9])
        assert wrong == 0, f'{wrong} mislabelled markers (prediction={use_prediction})'
        if use_prediction:
            assert lost / visible_total < 0.08, f'lost {lost} of {visible_total} visible markers'


def test_tracker_reacquires_with_prediction():
    rng = np.random.default_rng(5)
    model, data = _model()
    rest = _sites(model, data, NAMES)
    traj = _curl(model, data, current=(-0.7, 0.0, 0.0), duration=0.6)
    tracker = MarkerTracker(NAMES, rest, 0.0)
    hidden = NAMES.index('marker_2_seg_8')
    for k, frame in enumerate(traj):
        truth = frame[:9]
        keep = np.ones(len(frame), bool)
        if 10 <= k < 40:
            keep[hidden] = False
        pts, _ = _snapshot(frame[keep], rng, ghosts=0)
        out = tracker.update(pts, (k + 1) / 100.0, predicted=truth + rng.normal(0, 3e-3, truth.shape))
        if 10 <= k < 40:
            assert np.isnan(out[hidden, 0])
    assert np.linalg.norm(out[hidden] - truth[hidden]) < 3e-3


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
