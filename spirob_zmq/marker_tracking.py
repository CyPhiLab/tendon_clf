"""Frame-to-frame tracking of labelled Vicon markers.

Labels are fixed once at startup (spirob_zmq.pose_init). Each frame, every marker's position is
predicted at constant velocity, unlabelled detections are assigned to predictions in one global
assignment with a gate, and each segment's triangle must keep its side lengths. On hw_test1 the
constant-velocity prediction was off by up to 10.6 mm (segment 8, 2.7 m/s), hence the 15 mm gate,
well under the 35 mm spacing of the segment-8 markers. Markers that
are not matched, or fail the triangle check, are reported as NaN for that frame -- never carried
forward. A marker lost for longer than max_coast frames is searched for around the estimator's
predicted site position (if given) with a wider gate, and is only accepted back once its distance
to a partner on the same segment checks out.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class MarkerTracker:
    def __init__(self, names, positions, t, gate=0.015, reacquire_gate=0.025, triangle_tol=0.004,
                 max_coast=2, velocity_smoothing=0.5):
        self.names = list(names)
        self.pos = np.asarray(positions, dtype=float).reshape(-1, 3).copy()
        self.vel = np.zeros_like(self.pos)
        self.last_seen = np.full(len(self.names), float(t))
        self.missed = np.zeros(len(self.names), dtype=int)
        self.t = float(t)
        self.gate, self.reacquire_gate = gate, reacquire_gate
        self.triangle_tol, self.max_coast, self.alpha = triangle_tol, max_coast, velocity_smoothing

        # Rigid reference: distances between markers of the same segment, as measured at startup
        segs = {}
        for i, n in enumerate(self.names):
            segs.setdefault(n.split('_seg_')[1], []).append(i)
        self.pairs = [(i, j, np.linalg.norm(self.pos[i] - self.pos[j]))
                      for idx in segs.values() for a, i in enumerate(idx) for j in idx[a + 1:]]

    def update(self, points, t, predicted=None):
        """Assign this frame's points (shape (M, 3)) to the markers; returns (N, 3) with NaN rows."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full((len(self.names), 3), np.nan)
        self.t = float(t)

        tracked = self.missed == 0                  # matched in the previous frame
        lost = self.missed > self.max_coast
        pred = self.pos + self.vel * (t - self.last_seen)[:, None]
        if predicted is not None:
            predicted = np.asarray(predicted, dtype=float).reshape(-1, 3)
            pred[lost] = predicted[lost]
        gates = np.where(lost, self.reacquire_gate, self.gate)

        if len(points):
            cost = np.linalg.norm(pred[:, None] - points[None], axis=2)
            rows, cols = linear_sum_assignment(np.where(cost <= gates[:, None], cost, 1e3))
            for i, j in zip(rows, cols):
                if cost[i, j] <= gates[i]:
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

        seen = ~np.isnan(out[:, 0])
        for i in np.where(seen)[0]:
            gap = t - self.last_seen[i]
            if tracked[i] and gap > 0:
                v = (out[i] - self.pos[i]) / gap
                self.vel[i] = self.alpha * v + (1 - self.alpha) * self.vel[i]
            else:
                self.vel[i] = 0.0
            self.pos[i] = out[i]
            self.last_seen[i] = t
        self.missed[seen] = 0
        self.missed[~seen] += 1
        self.vel[self.missed > self.max_coast] = 0.0
        return out
