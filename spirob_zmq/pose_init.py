"""Start the estimator from any robot pose.

From one still Vicon snapshot of unlabelled markers this finds the Vicon -> MuJoCo frame, works out
which marker is which, and fits the joint angles:

1. stationary_clusters: DBSCAN over warmup frames; also tells whether the robot was still.
2. Frame: register_base_markers (static markers on the mount, any robot pose) or
   calibrate_frame_at_rest (robot at rest, segment triangles), saved with save_frame / load_frame.
3. Segment triangles are found by shape (side lengths do not depend on pose or frame), checked
   against chain geometry, then labelled by fitting the pose segment by segment (MarkerModel.initialize).
"""

import io
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mujoco import minimize
from sklearn.cluster import DBSCAN


def stationary_clusters(frames, eps=1.5e-3, min_fraction=0.8, max_spread=2e-3, max_unclustered=0.2):
    """Cluster centres of markers seen in at least min_fraction of frames.

    Returns (centres, still). still is False if any such marker moved more than max_spread, or if
    more than max_unclustered of all points fit no cluster (a moving marker smears into noise).
    """
    frames = [np.asarray(f, dtype=float).reshape(-1, 3) for f in frames]
    points = np.vstack(frames)
    if len(points) == 0:
        return np.zeros((0, 3)), False
    labels = DBSCAN(eps=eps, min_samples=max(2, int(0.5 * len(frames)))).fit_predict(points)
    centres, still = [], np.mean(labels == -1) <= max_unclustered
    for c in sorted(set(labels) - {-1}):
        p = points[labels == c]
        if len(p) < min_fraction * len(frames):
            continue
        centre = p.mean(axis=0)
        centres.append(centre)
        if np.linalg.norm(p - centre, axis=1).max() > max_spread:
            still = False
    return np.array(centres).reshape(-1, 3), still


def kabsch(src, dst):
    """R, t minimizing |R src + t - dst|; also the residual per point."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    return R, t, np.linalg.norm(src @ R.T + t - dst, axis=1)


def transform(points, frame):
    R, t = frame
    return np.asarray(points).reshape(-1, 3) @ np.asarray(R).T + np.asarray(t)


def match_point_set(points, ref, tol=3e-3):
    """Ordered subset of points whose pairwise distances match ref's (for k = len(ref) <= 4)."""
    k = len(ref)
    D = np.linalg.norm(points[:, None] - points[None], axis=2)
    Dr = np.linalg.norm(ref[:, None] - ref[None], axis=2)
    iu = np.triu_indices(k, 1)
    best, best_err = None, np.inf
    for perm in itertools.permutations(range(len(points)), k):
        err = np.sqrt(np.mean((D[np.ix_(perm, perm)][iu] - Dr[iu]) ** 2))
        if err < best_err:
            best, best_err = perm, err
    return (list(best), best_err) if best_err <= tol else (None, best_err)


def register_base_markers(points, base_ref, tol=3e-3):
    """Frame from static markers on the base mount (asymmetric layout). Returns (R, t, residual) or None."""
    idx, _ = match_point_set(points, base_ref, tol)
    if idx is None:
        return None
    R, t, res = kabsch(points[idx], base_ref)
    return R, t, res


def save_frame(path, R, t, residual, method):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'R': np.asarray(R).tolist(), 't': np.asarray(t).tolist(),
                   'residual_mm': (1e3 * np.asarray(residual)).round(2).tolist(),
                   'method': method, 'created': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)


def load_frame(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d['R']), np.array(d['t']), d


@dataclass
class InitResult:
    q: np.ndarray
    labels: dict                     # site name -> index into the points passed in
    residual: dict                   # site name -> marker error [m] after the fit
    groupings_tried: int = 0
    notes: list = field(default_factory=list)

    @property
    def rms(self):
        return float(np.sqrt(np.mean(np.square(list(self.residual.values())))))


class MarkerModel:
    """Marker geometry of the robot model, and pose fitting against labelled markers."""

    def __init__(self, model, site_names, q_prior, lam=1e-3):
        assert model.nq == model.nv, 'pose fit assumes hinge/slide joints only'
        self.model = model
        self.data = mujoco.MjData(model)
        self.q_prior = np.asarray(q_prior, dtype=float).copy()
        self.lam = lam
        limited = model.jnt_limited.astype(bool)
        self.lo = np.where(limited, model.jnt_range[:, 0], -10.0)
        self.hi = np.where(limited, model.jnt_range[:, 1], 10.0)

        # Segments in chain order, each with its three marker sites (marker_1, marker_2, marker_3)
        segs = {}
        for name in site_names:
            seg = name.split('_seg_')[1]
            segs.setdefault(seg, []).append(name)
        for seg, names in segs.items():
            if len(names) != 3:
                raise ValueError(f'segment {seg} needs three marker sites, got {names}')
        depth = {seg: self._depth(model.site(names[0]).id) for seg, names in segs.items()}
        self.segments = sorted(segs, key=lambda s: depth[s])
        self.seg_sites = {s: sorted(segs[s]) for s in self.segments}
        self.site_id = {n: model.site(n).id for n in site_names}

        # Pose-independent triangle side lengths, from the site positions in their body frames
        self.ref_sides = {s: self._sides(np.array([model.site(n).pos for n in self.seg_sites[s]]))
                          for s in self.segments}

        # Chain geometry at q = 0 (straight): bending only shortens the chord between segments
        d = self.data
        d.qpos[:] = 0.0
        mujoco.mj_kinematics(model, d)
        cent = {s: self._centroid(d, s) for s in self.segments}
        self.chord_max = {(a, b): np.linalg.norm(cent[b] - cent[a])
                          for k, a in enumerate(self.segments) for b in self.segments[k + 1:]}
        # The first segment's body hangs off a fixed body: its centroid keeps a fixed distance to the joint
        first_body = model.site_bodyid[self.site_id[self.seg_sites[self.segments[0]][0]]]
        parent = model.body_parentid[first_body]
        self.anchor = None
        if model.body_dofnum[parent] == 0 and model.body_jntnum[first_body] > 0:
            mujoco.mj_comPos(model, d)
            self.anchor = d.xanchor[model.body_jntadr[first_body]].copy()
            self.anchor_dist = np.linalg.norm(cent[self.segments[0]] - self.anchor)

    def _depth(self, site_id):
        b, n = self.model.site_bodyid[site_id], 0
        while b:
            b, n = self.model.body_parentid[b], n + 1
        return n

    @staticmethod
    def _sides(p):
        return np.sort([np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[1] - p[2])])

    def _centroid(self, d, seg):
        return d.site_xpos[[self.site_id[n] for n in self.seg_sites[seg]]].mean(axis=0)

    def site_positions(self, q, names):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[[self.site_id[n] for n in names]].copy()

    # ---- pose fit -------------------------------------------------------------------------------

    def fit(self, names, z, q0=None, max_iter=100):
        """Joint angles that put the named sites at z (shape (k, 3)), staying near the prior."""
        ids = [self.site_id[n] for n in names]
        z = np.asarray(z, dtype=float).reshape(-1)
        m, d, lam, qp = self.model, self.data, self.lam, self.q_prior
        jac = np.zeros((3, m.nv))

        def residual(x):
            x = x.reshape(m.nq, -1)
            out = np.empty((len(z) + m.nq, x.shape[1]))
            for k in range(x.shape[1]):
                d.qpos[:] = x[:, k]
                mujoco.mj_kinematics(m, d)
                out[:len(z), k] = d.site_xpos[ids].reshape(-1) - z
                out[len(z):, k] = lam * (x[:, k] - qp)
            return out

        def jacobian(x, r):
            d.qpos[:] = x.reshape(-1)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            J = np.zeros((len(z) + m.nq, m.nq))
            for j, sid in enumerate(ids):
                mujoco.mj_jacSite(m, d, jac, None, sid)
                J[3 * j:3 * j + 3] = jac
            J[len(z):] = lam * np.eye(m.nq)
            return J

        q0 = qp if q0 is None else q0
        q, _ = minimize.least_squares(np.clip(q0, self.lo, self.hi), residual, bounds=(self.lo, self.hi),
                                      jacobian=jacobian, verbose=0, max_iter=max_iter, output=io.StringIO())
        q = q.reshape(-1)
        err = np.linalg.norm(self.site_positions(q, names) - z.reshape(-1, 3), axis=1)
        return q, err

    # ---- labelling ------------------------------------------------------------------------------

    def candidate_groupings(self, points, shape_tol=6e-3, chain_tol=5e-3, max_groupings=3):
        """Disjoint triples of points, one per segment, ordered by how well their shapes match."""
        n = len(points)
        triples = list(itertools.combinations(range(n), 3))
        sides = {tr: self._sides(points[list(tr)]) for tr in triples}
        cands = {}
        for s in self.segments:
            ref = self.ref_sides[s]
            scored = []
            for tr in triples:
                err = np.sqrt(np.mean((sides[tr] - ref) ** 2))
                if err > shape_tol:
                    continue
                c = points[list(tr)].mean(axis=0)
                if s == self.segments[0] and self.anchor is not None:
                    if abs(np.linalg.norm(c - self.anchor) - self.anchor_dist) > chain_tol:
                        continue
                scored.append((err, tr, c))
            cands[s] = sorted(scored, key=lambda x: x[0])[:6]
            if not cands[s]:
                return []

        groupings = []

        def search(i, used, chosen, score):
            if i == len(self.segments):
                groupings.append((score, dict(chosen)))
                return
            s = self.segments[i]
            for err, tr, c in cands[s]:
                if used & set(tr):
                    continue
                if i > 0:
                    prev = self.segments[i - 1]
                    if np.linalg.norm(c - chosen[prev][1]) > self.chord_max[(prev, s)] + chain_tol:
                        continue
                chosen[s] = (tr, c)
                search(i + 1, used | set(tr), chosen, score + err ** 2)
                del chosen[s]

        search(0, set(), {}, 0.0)
        groupings.sort(key=lambda g: g[0])
        return [{s: tr for s, (tr, _) in g.items()} for _, g in groupings[:max_groupings]]

    def label_and_fit(self, points, grouping, q0=None):
        """Assign marker names within each triangle by fitting the pose segment by segment."""
        q = self.q_prior if q0 is None else q0
        labels = {}
        for s in self.segments:
            best = None
            for perm in itertools.permutations(grouping[s]):
                trial = dict(labels)
                trial.update(zip(self.seg_sites[s], perm))
                names = list(trial)
                q_try, err = self.fit(names, points[[trial[nm] for nm in names]], q, max_iter=60)
                score = np.sqrt(np.mean(err ** 2))
                if best is None or score < best[0]:
                    best = (score, trial, q_try)
            _, labels, q = best
        names = list(labels)
        q, err = self.fit(names, points[[labels[nm] for nm in names]], q)
        return q, labels, dict(zip(names, err))

    def initialize(self, points, q0=None, **grouping_kw):
        """Label unlabelled markers (in the MuJoCo frame) and fit the pose. None if no grouping fits."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        groupings = self.candidate_groupings(points, **grouping_kw)
        best = None
        for g in groupings:
            q, labels, res = self.label_and_fit(points, g, q0)
            r = InitResult(q, labels, res, len(groupings))
            if best is None or r.rms < best.rms:
                best = r
        return best

    def calibrate_frame_at_rest(self, points, q_rest, shape_tol=6e-3, chain_tol=5e-3, beam_width=36):
        """Vicon -> MuJoCo frame with the robot at rest, from the segment triangles.

        Labels come from the best rigid fit of all segments against the rest pose (the 3D layout of
        several triangles fixes the handedness); the frame itself uses the first segment only,
        which is the least affected by sag. Returns (R, t, residual of the first segment, labels).
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        names_all = [n for s in self.segments for n in self.seg_sites[s]]
        ref = dict(zip(names_all, self.site_positions(q_rest, names_all)))
        best = None
        # The anchor test needs the frame we are solving for, so group on shape and chords only
        anchor, self.anchor = self.anchor, None
        try:
            groupings = self.candidate_groupings(points, shape_tol, chain_tol, max_groupings=3)
        finally:
            self.anchor = anchor
        for g in groupings:
            # Segment by segment, keeping the best partial labellings (all orders of the first two
            # segments, whose fit is the first to fix the handedness): 6^segments fits otherwise
            beam = [{}]
            for s in self.segments:
                trials = []
                for labels in beam:
                    for perm in itertools.permutations(g[s]):
                        trial = dict(labels)
                        trial.update(zip(self.seg_sites[s], perm))
                        names = list(trial)
                        _, _, res = kabsch(points[[trial[n] for n in names]], np.array([ref[n] for n in names]))
                        trials.append((np.sqrt(np.mean(res ** 2)), trial))
                trials.sort(key=lambda x: x[0])
                beam = [labels for _, labels in trials[:beam_width]]
            score = trials[0][0]
            if best is None or score < best[0]:
                best = (score, beam[0])
        if best is None:
            return None
        labels = best[1]
        first = self.seg_sites[self.segments[0]]
        R, t, res = kabsch(points[[labels[n] for n in first]], np.array([ref[n] for n in first]))
        return R, t, res, labels
