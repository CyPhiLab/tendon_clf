"""Load the cached hardware runs and compute per-run metrics (shared by the figure scripts)."""

from pathlib import Path

import numpy as np

CACHE = Path(__file__).with_name('cache')
T_ON = 5.04          # controller enabled (s after first tracking sample), same in every run
T_SS = (10.0, 14.5)  # steady-state window: >= 5 s after onset, before the end-of-run segment
MIN_DURATION = 15.0  # runs that stop earlier were aborted
BASE = np.array([0.0, 0.0, 0.125])  # robot base in the MuJoCo frame (spirob base_height)


def setpoint_key(name):
    return tuple(float(v) for v in name.split('_'))


class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.setpoint = self.path.parent.name
        self.stamp = self.path.stem
        d = np.load(path)
        self.d = {k: d[k] for k in d.files}
        self.target = self.d['target']
        self.t = self.d['tip_vicon_t']
        self.tip = self.d['tip_vicon']
        self.err_vec = self.tip - self.target
        self.err = np.linalg.norm(self.err_vec, axis=1)
        # EKF tip estimate resampled on the Vicon timestamps
        te, pe = self.d['tip_est_t'], self.d['tip_est']
        self.tip_est = np.column_stack([np.interp(self.t, te, pe[:, k]) for k in range(3)])
        self.err_est = np.linalg.norm(self.tip_est - self.target, axis=1)
        self.est_err = np.linalg.norm(self.tip_est - self.tip, axis=1)

    @property
    def duration(self):
        return self.t[-1]

    def window(self, lo, hi):
        return (self.t >= lo) & (self.t <= hi)

    def metrics(self):
        ss = self.window(*T_SS)
        e0 = self.err[self.t < T_ON].mean()
        after = self.t >= T_ON
        # time for the error to first fall below 10% of its initial value... and stay below 20%
        e_after, t_after = self.err[after], self.t[after]
        below = e_after < 0.1 * e0
        t90 = t_after[np.argmax(below)] - T_ON if below.any() else np.nan
        # settling: last time the error is outside max(2 x steady-state, 10% e0) before the end of the window
        ss_err = self.err[ss].mean()
        band = max(2 * ss_err, 0.1 * e0)
        inside = self.window(T_ON, T_SS[1])
        outside = np.where(self.err[inside] > band)[0]
        t_settle = self.t[inside][outside[-1]] - T_ON if len(outside) else 0.0
        cur = self.d['current']
        ctrl = (self.d['fb_t'] >= T_ON) & (self.d['fb_t'] <= T_SS[1])
        return dict(
            setpoint=self.setpoint, stamp=self.stamp, duration=self.duration,
            e0_mm=1e3 * e0,
            ss_err_mm=1e3 * ss_err,
            ss_std_mm=1e3 * np.linalg.norm(self.tip[ss].std(axis=0)),
            ss_bias_mm=1e3 * self.err_vec[ss].mean(axis=0),
            min_err_mm=1e3 * e_after.min(),
            t90_s=t90, t_settle_s=t_settle,
            est_err_ss_mm=1e3 * self.est_err[ss].mean(),
            est_err_all_mm=1e3 * self.est_err[self.window(0, T_SS[1])].mean(),
            ekf_ss_err_mm=1e3 * self.err_est[ss].mean(),
            peak_current_a=np.abs(cur[ctrl]).max(axis=0),
            ss_current_a=cur[(self.d['fb_t'] >= T_SS[0]) & (self.d['fb_t'] <= T_SS[1])].mean(axis=0),
            sat_frac=self.d['saturated'][ctrl].mean(),
            compute_ms=1e3 * np.nanmedian(self.d['compute_s']),
        )


def load_runs(complete_only=True):
    runs = [Run(p) for p in sorted(CACHE.glob('*/*.npz'))]
    if complete_only:
        runs = [r for r in runs if r.duration >= MIN_DURATION]
    return runs


def setpoints(runs):
    """Setpoint folder names ordered by initial distance (P1 = shortest reach)."""
    names = sorted({r.setpoint for r in runs})
    e0 = {n: np.mean([r.err[0] for r in runs if r.setpoint == n]) for n in names}
    return sorted(names, key=e0.get)
