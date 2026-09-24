"""Score the estimator against ground truth from a recorded run.

    python -m spirob_zmq.ekf_report run.jsonl
    python -m spirob_zmq.ekf_report run.jsonl --plot ekf.png

Each ``robot_state`` estimate is paired with the ``true_state`` carrying the
same stamp (else the latest one received before it). Works on ``lockstep``
output and on ``topic record`` output from a real-time run.
"""

import argparse
import bisect
import json

import numpy as np

from spirob_zmq.common import ROBOT_STATE, SITE_MEASUREMENT, TRUE_STATE


def load(path):
    est, truth, meas = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec['topic'] == ROBOT_STATE:
                est.append(rec)
            elif rec['topic'] == TRUE_STATE:
                truth.append(rec)
            elif rec['topic'] == SITE_MEASUREMENT:
                meas.append(rec)
    return est, truth, meas


def pair(est, truth):
    """Return aligned arrays (t, q_est, dq_est, ee_est, q_true, dq_true, ee_true).

    An estimate is matched to the truth with the same stamp (the EKF stamps its
    estimate with the measurement's time, the plant stamps truth the same way);
    failing that, to the latest truth received before it."""
    truth_t = [r['t'] for r in truth]
    by_stamp = {round(r['msg']['stamp'], 6): r for r in truth}
    rows = []
    for r in est:
        if not r['msg']['is_valid']:
            continue
        tr = by_stamp.get(round(r['msg']['stamp'], 6))
        if tr is None:
            i = bisect.bisect_right(truth_t, r['t'] + 1e-9) - 1
            if i < 0:
                continue
            tr = truth[i]
        m, tr = r['msg'], tr['msg']
        ee_true = tr['ee_pos'] if 'ee_pos' in tr else tr['site_pos'][-3:]
        rows.append((r['t'], m['q'], m['dq'], m['task_pos'], tr['q'], tr['dq'], ee_true))
    cols = list(zip(*rows))
    return [np.asarray(c, dtype=float) for c in cols]


def summarize(path):
    est, truth, meas = load(path)
    if not est or not truth:
        raise SystemExit(f'{path}: need both {ROBOT_STATE} and {TRUE_STATE} messages')
    t, q, dq, ee, q_t, dq_t, ee_t = pair(est, truth)
    ee_err = np.linalg.norm(ee - ee_t, axis=1)
    q_err = np.linalg.norm(q - q_t, axis=1)
    dq_err = np.linalg.norm(dq - dq_t, axis=1)
    duration = t[-1] - t[0] if len(t) > 1 else 0.0
    return {
        'n_est': len(t),
        'est_rate_hz': (len(t) - 1) / duration if duration > 0 else float('nan'),
        'ee_err_rms_mm': 1e3 * np.sqrt(np.mean(ee_err ** 2)),
        'ee_err_max_mm': 1e3 * ee_err.max(),
        'ee_err_final_mm': 1e3 * ee_err[-1],
        'q_err_rms_rad': np.sqrt(np.mean(q_err ** 2)),
        'dq_err_rms_rad_s': np.sqrt(np.mean(dq_err ** 2)),
        'ee_true_final': ee_t[-1].round(4).tolist(),
    }, (t, ee_err, q_err, dq_err, ee, ee_t)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--plot', help='save an error-vs-time figure to this path')
    args = parser.parse_args(argv)

    series = {}
    for path in args.runs:
        stats, series[path] = summarize(path)
        print(path)
        for k, v in stats.items():
            print(f'  {k:18s} {v:.4g}' if isinstance(v, float) else f'  {k:18s} {v}')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for path, (t, ee_err, q_err, dq_err, *_) in series.items():
            axes[0].plot(t - t[0], 1e3 * ee_err, label=path)
            axes[1].plot(t - t[0], q_err)
            axes[2].plot(t - t[0], dq_err)
        axes[0].set_ylabel('ee error [mm]')
        axes[1].set_ylabel('|q error| [rad]')
        axes[2].set_ylabel('|dq error| [rad/s]')
        axes[2].set_xlabel('time [s]')
        for ax in axes:
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f'saved {args.plot}')


if __name__ == '__main__':
    main()
