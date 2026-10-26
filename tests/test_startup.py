"""Startup and marker-tracking checks on synthetic Vicon data from the robot model.

    python tests/test_startup.py        (or pytest tests/)
"""

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.common import (ProfileParams, VoltageLimit, load_model, marker_sites, rest_state,  # noqa: E402
                               step_current)
from spirob_zmq.marker_tracking import MarkerTracker  # noqa: E402
from spirob_zmq.pose_init import MarkerModel, stationary_clusters, transform  # noqa: E402

PARAMS = ProfileParams()
NAMES = list(PARAMS.declare_parameter('site_names'))
N = len(NAMES)


def _unmeasured_markers(min_gap=0.03):
    """Unmeasured marker sites: those more than min_gap from every measured marker at rest, and the rest.

    The far ones are distractors in the synthetic Vicon data. The near ones (seg 10 next to 11, 14
    and 18 next to 16: 17-24 mm apart, nearly the same triangle) cannot be told apart from the
    measured markers unless they are tracked too (robots.py unpublished_sites).
    """
    model = load_model(PARAMS)
    data = mujoco.MjData(model)
    rest_state(PARAMS, model, data)
    names = [model.site(i).name for i in range(model.nsite)]
    unmeasured = [n for n in names if re.fullmatch(r'marker_[123]_seg_\d+', n) and n not in NAMES]
    measured = np.array([data.site(n).xpos for n in NAMES])
    far = [n for n in unmeasured if np.linalg.norm(measured - data.site(n).xpos, axis=1).min() > min_gap]
    return far, [n for n in unmeasured if n not in far]


EXTRA, ADJACENT = _unmeasured_markers()
# Tracked markers (measured first), and the markers on the robot, in the configured setup and with the
# adjacent markers on the robot and tracked
TRACKED = marker_sites(PARAMS)
SETUPS = {'configured': (TRACKED, TRACKED + [n for n in EXTRA if n not in TRACKED]),
          'adjacent': (NAMES + ADJACENT, NAMES + ADJACENT + EXTRA)}


def _model():
    model = load_model(PARAMS)
    data = mujoco.MjData(model)
    rest_state(PARAMS, model, data)
    return model, data


def _sites(model, data, names):
    mujoco.mj_kinematics(model, data)
    return np.array([data.site(n).xpos.copy() for n in names])


def _curl(model, data, current=(-0.7, 0.0, 0.0), duration=0.6, rate=100.0, names=None):
    """Drive the tendons from rest; marker trajectories (default: measured, then distractors) at the Vicon rate."""
    vlimit = VoltageLimit(PARAMS)
    n_sub = int(round(1.0 / rate / model.opt.timestep))
    traj = []
    for _ in range(int(duration * rate)):
        step_current(model, data, np.array(current), n_sub, vlimit)
        traj.append(_sites(model, data, NAMES + EXTRA if names is None else names))
    return np.array(traj)


def _tracker(model, names, positions):
    mm = MarkerModel(model, names, model.key_qpos[0])
    return MarkerTracker(names, positions, 0.0, chain=mm.segments, chord_max=mm.chord_max, anchor=mm.anchor)


def _snapshot(true_pts, rng, noise=3e-4, ghosts=2):
    pts = true_pts + rng.normal(0, noise, true_pts.shape)
    pts = np.vstack([pts, rng.uniform(-0.3, 0.3, (ghosts, 3)) + [0, 0.2, 0.2]])
    perm = rng.permutation(len(pts))
    return pts[perm], perm


def _labels_ok(result, perm, names=NAMES):
    return all(perm[result.labels[n]] == names.index(n) for n in names)


def test_marker_model_geometry():
    model, _ = _model()
    mm = MarkerModel(model, NAMES, model.key_qpos[0])
    assert mm.segments == sorted({n.split('_seg_')[1] for n in NAMES}, key=int)
    sides = [mm.ref_sides[s].mean() for s in mm.segments]
    assert all(a > b for a, b in zip(sides, sides[1:])), 'triangles should shrink along the chain'
    if '2' in mm.segments:
        assert abs(1e3 * mm.ref_sides['2'].mean() - 60.6) < 0.2
    assert mm.anchor is not None


def test_initialize_from_rest_and_curled_pose():
    rng = np.random.default_rng(1)
    for setup, (tracked, present) in SETUPS.items():
        model, data = _model()
        mm = MarkerModel(model, tracked, model.key_qpos[0])
        rest = _sites(model, data, present)
        curled = _curl(model, data, names=present)[-1]
        far = np.linalg.norm(curled[:N] - rest[:N], axis=1).max()
        assert far > 0.05, f'curl too small for a meaningful test ({1e3 * far:.0f} mm)'
        for pts in (rest, curled):
            shuffled, perm = _snapshot(pts, rng)
            r = mm.initialize(shuffled)
            assert r is not None, setup
            assert _labels_ok(r, perm, tracked), setup
            assert r.rms < 1.5e-3, (setup, r.rms)


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
    """Never mislabel, with or without the estimator's prediction, and keep losses low, tip included.

    The tip segments (11, 16) reach about 4 m/s in this curl; with the 15 mm gate alone they lost
    15-20 % of their frames, and with untracked adjacent markers on the robot they get mislabelled.
    """
    for setup, (tracked, present) in SETUPS.items():
        model, data = _model()
        rest = _sites(model, data, tracked)
        traj = _curl(model, data, current=(-1.2, 0.0, 0.0), duration=0.8, names=present)
        n = len(tracked)
        speed = np.linalg.norm(np.diff(traj[:, :N], axis=0), axis=2).max() * 100
        assert speed > 2.0, f'peak marker speed only {speed:.2f} m/s'
        seg = np.array([int(name.split('_seg_')[1]) for name in tracked])
        groups = {'segments <= 8': seg <= 8, 'segments 11, 16': np.isin(seg, (11, 16))}
        for use_prediction in (False, True):
            wrong, lost, visible_total = 0, dict.fromkeys(groups, 0), dict.fromkeys(groups, 0)
            for seed in range(4):
                rng = np.random.default_rng(4 + seed)
                tracker = _tracker(model, tracked, rest)
                for k, frame in enumerate(traj):
                    truth = frame[:n]
                    visible = rng.random(len(frame)) > 0.1
                    pts, _ = _snapshot(frame[visible], rng)
                    pred = truth + rng.normal(0, 3e-3, truth.shape) if use_prediction else None
                    out = tracker.update(pts, (k + 1) / 100.0, predicted=pred)
                    ok = ~np.isnan(out[:, 0])
                    wrong += np.sum(np.linalg.norm(out[ok] - truth[ok], axis=1) > 3e-3)
                    for g, sel in groups.items():
                        lost[g] += np.sum(~ok & visible[:n] & sel)
                        visible_total[g] += np.sum(visible[:n] & sel)
            assert wrong == 0, f'{wrong} mislabelled markers ({setup}, prediction={use_prediction})'
            for g in groups:
                frac = lost[g] / visible_total[g]
                assert frac < 0.05, f'{setup}, prediction={use_prediction}: lost {100 * frac:.1f} % on {g}'


def test_tracker_reacquires_lost_tip_segments_without_estimator():
    """All six tip markers vanish at speed (as in run 20261022_140828) and come back as whole segments."""
    for setup, (tracked, present) in SETUPS.items():
        model, data = _model()
        rest = _sites(model, data, tracked)
        traj = _curl(model, data, current=(-1.9, 0.0, 0.0), duration=0.5, names=present)
        n = len(tracked)
        tip = np.array([name.split('_seg_')[1] in ('11', '16') for name in tracked])
        gone = range(15, 25)                        # tip speed 3-5 m/s
        for seed in range(4):
            rng = np.random.default_rng(10 + seed)
            tracker = _tracker(model, tracked, rest)
            back = None
            for k, frame in enumerate(traj):
                truth = frame[:n]
                visible = rng.random(len(frame)) > 0.05
                if k in gone:
                    visible[:n][tip] = False
                pts, _ = _snapshot(frame[visible], rng)
                out = tracker.update(pts, (k + 1) / 100.0)
                ok = ~np.isnan(out[:, 0])
                assert np.all(np.linalg.norm(out[ok] - truth[ok], axis=1) < 3e-3), f'mislabelled at frame {k} ({setup})'
                if k >= gone[-1] + 1 and back is None and ok[tip].all():
                    back = k - gone[-1] - 1
            assert back is not None and back <= 3, f'tip segments back after {back} frames ({setup}, seed {seed})'


def test_tracker_reacquires_with_prediction():
    rng = np.random.default_rng(5)
    model, data = _model()
    rest = _sites(model, data, NAMES)
    traj = _curl(model, data, current=(-0.7, 0.0, 0.0), duration=0.6)
    tracker = MarkerTracker(NAMES, rest, 0.0)
    hidden = NAMES.index('marker_2_seg_8')
    for k, frame in enumerate(traj):
        truth = frame[:N]
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
