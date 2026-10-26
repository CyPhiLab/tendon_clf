"""Frame-to-frame tracking of labelled Vicon markers.

Labels are fixed once at startup (spirob_zmq.pose_init). Each frame, every marker's position is
predicted at constant velocity, unlabelled detections are assigned to predictions in one global
assignment with a gate, and each segment's triangle must keep its side lengths. On hw_test1 the
constant-velocity prediction was off by up to 10.6 mm (segment 8, 2.7 m/s), hence the 15 mm gate.
The gate widens with the marker's last step (speed_gate): when the robot stops, the prediction
overshoots by about a frame's motion, up to 40 mm at the tip. Leaving a marker unmatched costs its
gate, so the assignment does not swap markers just to match more of them. Markers that are not
matched, or fail a check, are reported as NaN for that frame -- never carried forward.

The triangles are near-equilateral and neighbouring ones are alike, so the checks are geometric:
- each segment's orientation is tracked (constant angular velocity) and a match must stay within
  turn_tol + turn_accel * age^2 of it: the synthetic curls are off by up to 11, 30 and 53 degrees
  after 1, 2 and 3 frames, while a wrong order of the markers is 120 or 180 degrees off;
- noise does not make all three sides longer or shorter: a neighbouring segment's triangle does;
- in a neighbouring segment's frame a segment's markers move by at most max_bend_rate times their
  distance (in the curls at most 9 rad/s), while a swap moves them by the segment spacing;
- a segment needs two markers matched: nothing checks a lone one, and one takes a partner's or a
  neighbour's point when the robot stops short of the predictions.
A marker that is not seen is kept at its partner distances from the seen markers of its segment,
and the third marker of a segment whose other two are matched is looked for at those distances. A
marker lost for longer than max_coast frames with no partner seen is searched for around the
estimator's predicted site position (if given) with a wider gate.

A segment not held by two markers in the last frame is only taken back whole, from the points
nobody took: a triangle with that segment's measured side lengths, within reach of where it was
last seen, keeping the chain chords to the segments being tracked. The segments only bend, so the
order of its markers follows from the neighbouring segment in the chain (or the fixed base for the
first segment); see _order. The estimator's prediction, when near, must agree. This needs no
estimator: in run 20261022_140828 the tip segments were lost at 4 m/s and never came back, as
re-acquisition only searched around the estimator's predictions.

Markers that are on the robot but not used by the estimator (segments next to measured ones,
17-24 mm away with nearly the same triangle) must be tracked too, so that the global assignment
does not hand their points to the measured markers; the caller publishes only the measured ones.
"""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from spirob_zmq.pose_init import kabsch


def _twist(D, axis):
    """Signed angle of rotation D about axis, from its swing-twist decomposition."""
    x, y, z, w = Rotation.from_matrix(D).as_quat()
    return np.angle(np.exp(2j * np.arctan2(np.dot([x, y, z], axis), w)))


def _angle(R):
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))


def _constrain(p, anchors, dists):
    """Point nearest p at distances dists from one or two anchor points."""
    if len(anchors) == 1:
        u = p - anchors[0]
        return anchors[0] + dists[0] * u / max(np.linalg.norm(u), 1e-9)
    (a, b), (da, db) = anchors, dists
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    along = np.clip((da ** 2 - db ** 2 + L ** 2) / (2 * L), -da, da)
    centre, radius = a + along * u, np.sqrt(da ** 2 - along ** 2)
    w = p - centre
    w -= np.dot(w, u) * u
    return centre + radius * w / max(np.linalg.norm(w), 1e-9)


class MarkerTracker:
    def __init__(self, names, positions, t, gate=0.015, reacquire_gate=0.025, triangle_tol=0.0025,
                 max_coast=2, velocity_smoothing=0.5, speed_gate=0.5, max_velocity_gap=0.05,
                 size_tol=0.001, turn_tol=0.25, turn_accel=1000.0, max_turn=1.0, max_orientation_age=0.035,
                 max_bend_rate=10.0, chain=None, chord_max=None, anchor=None, chain_tol=0.005,
                 segment_shape_tol=0.001, segment_margin=0.0007, max_speed=6.0, max_twist=0.7,
                 min_chord_cos=0.15, parity_age=0.1):
        """names/positions: the labelled markers at time t (shape (N, 3)), named marker_k_seg_s.

        chain: segment names from base to tip (default: by segment number). chord_max: (a, b) ->
        largest centroid distance between segments a and b (MarkerModel.chord_max), to bound the
        search for a lost segment; without it only max_speed [m/s] does. anchor: the fixed joint the
        first segment hangs from (MarkerModel.anchor), so the chain can be found again from the base
        when every segment is lost.
        """
        self.names = list(names)
        self.pos = np.asarray(positions, dtype=float).reshape(-1, 3).copy()
        self.vel = np.zeros_like(self.pos)
        self.last_seen = np.full(len(self.names), float(t))
        self.missed = np.zeros(len(self.names), dtype=int)
        self.carried = np.zeros(len(self.names), dtype=bool)   # kept at partner distances last frame
        self.vel_known = np.ones(len(self.names), dtype=bool)  # velocity from consecutive frames
        self.source = np.full(len(self.names), '', dtype=object)  # how each marker was matched
        self.t = float(t)
        self.gate, self.reacquire_gate = gate, reacquire_gate
        self.triangle_tol, self.max_coast, self.alpha = triangle_tol, max_coast, velocity_smoothing
        self.chain_tol, self.shape_tol, self.margin = chain_tol, segment_shape_tol, segment_margin
        self.size_tol = size_tol
        self.max_speed, self.max_turn, self.max_age = max_speed, max_turn, max_orientation_age
        self.turn_tol, self.turn_accel = turn_tol, turn_accel
        self.max_bend_rate, self.max_twist, self.min_chord_cos = max_bend_rate, max_twist, min_chord_cos
        self.max_velocity_gap, self.speed_gate, self.parity_age = max_velocity_gap, speed_gate, parity_age
        self.anchor = None if anchor is None else np.asarray(anchor, dtype=float)

        # Rigid reference: distances between markers of the same segment, as measured at startup
        segs = {}
        for i, n in enumerate(self.names):
            segs.setdefault(n.split('_seg_')[1], []).append(i)
        self.seg_idx = {s: sorted(idx, key=lambda i: self.names[i]) for s, idx in segs.items()}
        self.segment_of = [n.split('_seg_')[1] for n in self.names]
        self.pairs = [(i, j, np.linalg.norm(self.pos[i] - self.pos[j]))
                      for idx in segs.values() for a, i in enumerate(idx) for j in idx[a + 1:]]
        self.dist = {(i, j): d0 for a, b, d0 in self.pairs for i, j in ((a, b), (b, a))}

        # Segment orientations, relative to the startup pose, for segments with three markers
        self.chain = [s for s in (chain if chain is not None else sorted(segs, key=int)) if s in segs]
        self.rank = {s: k for k, s in enumerate(self.chain)}
        self.chord_max = dict(chord_max or {})
        full = [s for s in self.chain if len(self.seg_idx[s]) == 3]
        self.ref = {s: self.pos[self.seg_idx[s]].copy() for s in full}
        self.ref_sides = {s: self._sides(p) for s, p in self.ref.items()}
        cent = {s: p.mean(axis=0) for s, p in self.ref.items()}
        self.normal = {}                                       # triangle normal, pointing to the tip
        for k, s in enumerate(full):
            p = self.ref[s]
            nrm = np.cross(p[1] - p[0], p[2] - p[0])
            along = cent[full[min(k + 1, len(full) - 1)]] - cent[full[max(k - 1, 0)]]
            self.normal[s] = np.sign(np.dot(nrm, along) or 1.0) * nrm / np.linalg.norm(nrm)
        self.seg_c = dict(cent)                                # centroid when last seen
        self.seg_R = {s: np.eye(3) for s in full}              # orientation when last seen
        self.seg_t = {s: float(t) for s in full}
        self.seg_w = {s: np.zeros(3) for s in full}            # angular velocity [rad/s]
        self.streak = {s: 1 for s in full}                     # frames in a row with two markers seen
        # Orientation of each segment relative to every other, and its markers in the other's
        # frame, from the last frame both were seen
        self.rel = {(s, n): np.eye(3) for s in full for n in full if n != s}
        self.rel_p = {(s, n): self.ref[s] - cent[n] for s in full for n in full if n != s}
        self.rel_t = {(s, n): float(t) for s in full for n in full if n != s}

    @staticmethod
    def _sides(p):
        return np.sort([np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[1] - p[2])])

    def _chord(self, a, b):
        return self.chord_max.get((a, b), self.chord_max.get((b, a)))

    def update(self, points, t, predicted=None):
        """Assign this frame's points (shape (M, 3)) to the markers; returns (N, 3) with NaN rows."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full((len(self.names), 3), np.nan)
        t_prev, self.t = self.t, float(t)

        tracked = self.missed == 0                  # matched in the previous frame
        lost = (self.missed > self.max_coast) & ~self.carried
        pred = self.pos + self.vel * (t - self.last_seen)[:, None]
        if predicted is not None:
            predicted = np.asarray(predicted, dtype=float).reshape(-1, 3)
            use = lost & ~np.isnan(predicted[:, 0])
            pred[use] = predicted[use]
        # A sudden stop puts a fast marker about one frame's motion off its prediction
        step = np.linalg.norm(self.vel, axis=1) * (t - self.last_seen)
        gates = np.where(lost, self.reacquire_gate, self.gate + self.speed_gate * step)

        # Markers of a segment that can only be taken back whole do not compete for points
        whole = np.array([s in self.ref and self.streak[s] == 0 for s in self.segment_of]) | ~self.vel_known
        if len(points):
            # Leaving a marker unmatched costs its gate: a cheaper swap would only add matches
            cost = np.linalg.norm(pred[:, None] - points[None], axis=2)
            n = len(self.names)
            C = np.full((n, len(points) + n), 1e3)
            C[:, :len(points)] = np.where((cost <= gates[:, None]) & ~whole[:, None], cost, 1e3)
            C[np.arange(n), len(points) + np.arange(n)] = gates
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if j < len(points) and C[i, j] <= gates[i]:
                    out[i] = points[j]

        def consistent(i, j, d0):
            return abs(np.linalg.norm(out[i] - out[j]) - d0) <= self.triangle_tol

        # Markers tracked through the last frame: reject both of any pair whose distance changed
        for i, j, d0 in self.pairs:
            if tracked[i] and tracked[j] and not np.isnan(out[i, 0]) and not np.isnan(out[j, 0]):
                if not consistent(i, j, d0):
                    out[i] = out[j] = np.nan
        # A marker coming back after a gap must be confirmed by a partner on its segment
        for i in np.where(~tracked & ~np.isnan(out[:, 0]))[0]:
            checks = [consistent(i, j if a == i else a, d0) for a, j, d0 in self.pairs
                      if i in (a, j) and not np.isnan(out[j if a == i else a, 0])]
            if not checks or not all(checks):
                out[i] = np.nan
        # Segments must keep their orientation. One not held by two markers in the last frame is only
        # taken back whole: its markers' predictions are poor and a neighbour's triangle is close by
        for s in self.ref:
            idx = self.seg_idx[s]
            on = [i for i in idx if not np.isnan(out[i, 0])]
            if on and (self.streak[s] == 0 or not self.vel_known[on].all()):
                ok = False
            elif len(on) == 3 and not self._size_ok(s, out[idx]):
                ok = False
            elif len(on) == 1:
                # Nothing checks a lone marker, and when the robot stops short of the predictions
                # one takes a partner's or the neighbouring segment's point
                ok = False
            elif on:
                ok = self._turn_ok(s, self._triangle(s, out, pred), t)
            if on and not ok:
                out[idx] = np.nan
        self._check_neighbours(out, t)
        self.source = np.where(np.isnan(out[:, 0]), '', 'gate').astype(object)

        self._complete(out, points, pred, gates, t)
        self._reacquire(out, points, pred, t, predicted)

        seen = ~np.isnan(out[:, 0])
        prev = self.pos + self.vel * (t_prev - self.last_seen)[:, None]   # everyone at the last frame
        known = tracked | self.carried
        for i in np.where(seen)[0]:
            gap = t - self.last_seen[i]
            if gap > 0 and (known[i] or gap <= self.max_velocity_gap):
                v = (out[i] - self.pos[i]) / gap
                self.vel[i] = self.alpha * v + (1 - self.alpha) * self.vel[i] if tracked[i] else v
                self.vel_known[i] = True
            else:
                self.vel[i], self.vel_known[i] = 0.0, False
            self.pos[i] = out[i]
            self.last_seen[i] = t
        self._carry_unseen(seen, known, prev, t - t_prev, t_prev, predicted)
        self.missed[seen] = 0
        self.missed[~seen] += 1
        self.vel[(t - self.last_seen > self.max_velocity_gap) & ~self.carried] = 0.0
        self._remember_segments(seen, t)
        return out

    # ---- segment orientation --------------------------------------------------------------------

    def _triangle(self, s, out, pred):
        """Segment s's markers: all matched, or two matched and the third at its partner distances."""
        idx = self.seg_idx[s]
        off = [i for i in idx if np.isnan(out[i, 0])]
        if not off:
            return out[idx]
        if len(off) > 1:
            return None
        P = out[idx].copy()
        on = [j for j in idx if j != off[0]]
        P[idx.index(off[0])] = _constrain(pred[off[0]], out[on], [self.dist[(off[0], j)] for j in on])
        return P

    def _size_ok(self, s, P):
        """Noise does not make all three sides longer or shorter: a neighbour's triangle does."""
        return abs(np.mean(self._sides_ordered(P) - self._sides_ordered(self.ref[s]))) <= self.size_tol

    @staticmethod
    def _sides_ordered(p):
        return np.array([np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[1] - p[2])])

    def _turn_ok(self, s, P, t):
        age = t - self.seg_t[s]
        if age > self.max_age:
            return False
        R_pred = Rotation.from_rotvec(self.seg_w[s] * age).as_matrix() @ self.seg_R[s]
        allowed = min(self.turn_tol + self.turn_accel * age ** 2, self.max_turn)
        return _angle(kabsch(self.ref[s], P)[0] @ R_pred.T) <= allowed

    def _check_neighbours(self, out, t):
        """Neighbouring segments bend slowly relative to each other: in one's frame, the other's
        markers move by at most max_bend_rate times their distance (a swap moves one by the spacing)."""
        matched = {s: ~np.isnan(out[idx, 0]) for s, idx in self.seg_idx.items()}
        frame = {s: (kabsch(self.ref[s], out[self.seg_idx[s]])[0], out[self.seg_idx[s]].mean(axis=0))
                 for s in self.ref if matched[s].all()}
        drop = set()
        for a, b in zip(self.chain, self.chain[1:]):
            age = t - self.rel_t.get((b, a), -np.inf)
            if age > self.max_age:
                continue
            bad = False
            for x, y in ((b, a), (a, b)):
                if y not in frame or not matched[x].any():
                    continue
                R, c = frame[y]
                r, r0 = (out[self.seg_idx[x]] - c) @ R, self.rel_p[(x, y)]
                allowed = self.max_bend_rate * age * np.linalg.norm(r0, axis=1) + self.triangle_tol
                bad |= bool(np.any((np.linalg.norm(r - r0, axis=1) > allowed)[matched[x]]))
            if bad:
                # Keep the one seen longer, if either
                if self.streak[a] != self.streak[b]:
                    drop.add(a if self.streak[a] < self.streak[b] else b)
                else:
                    drop.update((a, b))
        for s in drop:
            out[self.seg_idx[s]] = np.nan

    def _carry_unseen(self, seen, known, prev, dt, t_prev, predicted):
        """Keep markers not seen this frame at their partner distances from the seen ones.

        Seen markers do not show the turn about the axis through two of them (for the segments
        bending about that axis, all of it), so the guess comes from the estimator, else from how the
        neighbouring segments turned since the last frame. Not the marker's own velocity: that keeps
        it turning after the robot stops."""
        turned = {}
        for s, ref in self.ref.items():
            idx = self.seg_idx[s]
            if seen[idx].all() and self.seg_t[s] == t_prev:
                R = kabsch(ref, self.pos[idx])[0]
                turned[s] = Rotation.from_matrix(R @ self.seg_R[s].T).as_rotvec()
        for s, idx in self.seg_idx.items():
            on = [i for i in idx if seen[i]]
            self.carried[idx] = False
            if not on or len(on) == len(idx):
                continue
            v = self.vel[on].mean(axis=0)
            nbrs = [turned[n] for n in turned if s in self.rank and abs(self.rank[n] - self.rank[s]) == 1]
            dR = Rotation.from_rotvec(np.mean(nbrs, axis=0)).as_matrix() if nbrs else np.eye(3)
            c0, c1 = prev[on].mean(axis=0), self.pos[on].mean(axis=0)
            for i in idx:
                if seen[i]:
                    continue
                if predicted is not None and not np.isnan(predicted[i, 0]):
                    guess = predicted[i]
                elif known[i]:
                    guess = c1 + dR @ (prev[i] - c0)
                else:
                    guess = self.pos[i]
                p = _constrain(guess, self.pos[on], [self.dist[(i, j)] for j in on])
                self.pos[i], self.vel[i], self.vel_known[i] = p, v, self.vel_known[on].all()
                self.last_seen[i], self.carried[i] = self.t, True

    def _remember_segments(self, seen, t):
        R = {}                       # segments seen whole: their relative poses are learnt from these
        for s, ref in self.ref.items():
            idx = self.seg_idx[s]
            if seen[idx].sum() < 2:
                self.streak[s] = 0
                continue
            self.streak[s] += 1
            P = self.pos[idx]
            R_s = kabsch(ref, P)[0]
            dt = t - self.seg_t[s]
            fresh = 0 < dt <= self.max_age
            self.seg_w[s] = Rotation.from_matrix(R_s @ self.seg_R[s].T).as_rotvec() / dt if fresh else np.zeros(3)
            self.seg_R[s], self.seg_t[s], self.seg_c[s] = R_s, t, P.mean(axis=0)
            if seen[idx].all():
                R[s] = R_s
        for s in R:
            for n in R:
                if n != s:
                    self.rel[(s, n)] = R[n].T @ R[s]
                    self.rel_p[(s, n)] = (self.pos[self.seg_idx[s]] - self.seg_c[n]) @ R[n]
                    self.rel_t[(s, n)] = t

    # ---- finding markers without a prediction ---------------------------------------------------

    @staticmethod
    def _free(out, points):
        used = [j for j in range(len(points)) if np.any(np.all(out == points[j], axis=1))]
        return np.setdiff1d(np.arange(len(points)), used)

    def _complete(self, out, points, pred, gates, t):
        """The third marker of a segment whose other two are matched: right distances and turn, in its
        gate and no nearer another unmatched marker's prediction, unique."""
        free = self._free(out, points)
        unmatched = np.isnan(out[:, 0])
        for s in self.ref:
            idx = self.seg_idx[s]
            off = [i for i in idx if np.isnan(out[i, 0])]
            if len(off) != 1 or not len(free):
                continue
            i, on = off[0], [j for j in idx if j != off[0]]
            err = np.max([np.abs(np.linalg.norm(points[free] - out[j], axis=1) - self.dist[(i, j)]) for j in on], axis=0)
            hits = []
            for j in free[err <= self.triangle_tol]:
                d = np.linalg.norm(pred[unmatched] - points[j], axis=1)
                own = np.linalg.norm(pred[i] - points[j])
                if own > gates[i] or d.min() < own:
                    continue
                P = out[idx].copy()
                P[idx.index(i)] = points[j]
                if self._turn_ok(s, P, t) and self._size_ok(s, P):
                    hits.append(j)
            if len(hits) == 1:
                out[i] = points[hits[0]]
                self.source[i] = 'third'
                free = free[free != hits[0]]

    def _reacquire(self, out, points, pred, t, predicted):
        """Whole segments with no marker matched, found by shape and chain geometry."""
        lost = [s for s in self.ref if np.isnan(out[self.seg_idx[s], 0]).all()]
        free = self._free(out, points)
        if not lost or len(free) < 3:
            return
        triples = {tr: self._sides(points[list(tr)]) for tr in itertools.combinations(free, 3)}
        while lost and triples:
            known = {s: P for s in self.ref if (P := self._triangle(s, out, pred)) is not None}
            cands = sorted((err, s, tr) for s in lost for err, tr in self._candidates(s, triples, points, known, t))
            done = None
            for err, s, tr in cands:
                # Ambiguous if another triangle fits this segment nearly as well, or this triangle
                # (or one sharing a point with it) fits another lost segment nearly as well
                if any(e2 < err + self.margin and (s2 == s or set(tr2) & set(tr))
                       for e2, s2, tr2 in cands if (s2, tr2) != (s, tr)):
                    continue
                order = self._order(s, points[list(tr)], known, predicted, t)
                if order is not None:
                    done = s, [tr[k] for k in order]
                    break
            if done is None:
                return
            s, idx = done
            out[self.seg_idx[s]] = points[idx]
            self.source[self.seg_idx[s]] = 'segment'
            lost.remove(s)
            triples = {tr: v for tr, v in triples.items() if not set(tr) & set(idx)}

    def _candidates(self, s, triples, points, known, t):
        reach = self.max_speed * (t - self.seg_t[s]) + self.gate
        found = []
        for tr, sides in triples.items():
            err = np.sqrt(np.mean((sides - self.ref_sides[s]) ** 2))
            if err > self.shape_tol:
                continue
            c = points[list(tr)].mean(axis=0)
            if np.linalg.norm(c - self.seg_c[s]) > reach:
                continue
            if any(self._chord(s, n) is not None
                   and np.linalg.norm(c - P.mean(axis=0)) > self._chord(s, n) + self.chain_tol
                   for n, P in known.items()):
                continue
            found.append((err, tr))
        return found

    def _order(self, s, P, known, predicted, t):
        """Order of triangle P's points as segment s's markers, or None if not certain.

        Against the neighbouring segment n: the right order (an even or odd permutation) keeps the
        normal within 90 degrees of where it was relative to n when both were last seen; beyond
        parity_age that may no longer hold, and the normal must also point along the chord to n.
        Of the three orders left, the right one has not twisted relative to n (the others: 120 deg)."""
        c = P.mean(axis=0)
        geometric = None
        nbrs = [n for n in known if abs(self.rank[n] - self.rank[s]) == 1]
        ref = None
        if nbrs:
            n = min(nbrs, key=lambda m: self.rank[m])          # the proximal neighbour if both
            ref = (kabsch(self.ref[n], known[n])[0], self.rel[(s, n)], self.rel_t[(s, n)],
                   (c - known[n].mean(axis=0)) * (1.0 if self.rank[n] < self.rank[s] else -1.0))
        elif self.rank[s] == 0 and self.anchor is not None:
            ref = (np.eye(3), self.seg_R[s], self.seg_t[s], c - self.anchor)   # the fixed base
        if ref is not None:
            R_n, rel, t_rel, chord = ref
            chord /= np.linalg.norm(chord)
            a = self.normal[s]
            scored = []
            for perm in itertools.permutations(range(3)):
                R_p, _, res = kabsch(self.ref[s], P[list(perm)])
                D = rel.T @ (R_n.T @ R_p)
                if res.max() > self.triangle_tol or np.dot(a, D @ a) <= 0:
                    continue
                if t - t_rel > self.parity_age and np.dot(R_p @ a, chord) < self.min_chord_cos:
                    continue
                scored.append((abs(_twist(D, a)), perm))
            scored.sort()
            if scored and scored[0][0] < self.max_twist and all(x > 2 * self.max_twist for x, _ in scored[1:]):
                geometric = scored[0][1]
        if predicted is not None:
            q = predicted[self.seg_idx[s]]
            if not np.isnan(q).any() and np.linalg.norm(q.mean(axis=0) - c) < self.reacquire_gate:
                errs = sorted((np.linalg.norm(P[list(perm)] - q, axis=1).max(), perm)
                              for perm in itertools.permutations(range(3)))
                if errs[0][0] < self.reacquire_gate and errs[1][0] > errs[0][0] + self.gate:
                    if geometric is not None and geometric != errs[0][1]:
                        return None
                    return errs[0][1]
        return geometric
