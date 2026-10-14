"""Score the estimator against ground truth from a recorded run."""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np

from spirob_zmq.common import MOTOR_STATE, ROBOT_STATE, SITE_MEASUREMENT, TRUE_STATE


def load(path):
    est, truth, meas, motor = [], [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec['topic'] == ROBOT_STATE:
                est.append(rec)
            elif rec['topic'] == TRUE_STATE:
                truth.append(rec)
            elif rec['topic'] == SITE_MEASUREMENT:
                meas.append(rec)
            elif rec['topic'] == MOTOR_STATE:
                motor.append(rec)
    return est, truth, meas, motor


def motor_series(records):
    by_motor = {}
    for rec in records:
        msg = rec['msg']
        motor_id = msg.get('motor_id')
        if motor_id is None:
            continue
        series = by_motor.setdefault(motor_id, {'t': [], 'current': [], 'commanded_current': [], 'velocity': []})
        series['t'].append(rec['t'])
        series['velocity'].append(msg.get('velocity', float('nan')))
        series['current'].append(msg.get('current', float('nan')))
        series['commanded_current'].append(msg.get('commanded_current', msg.get('current', float('nan'))))
    return {
        motor_id: {key: np.asarray(values, dtype=float) for key, values in series.items()}
        for motor_id, series in by_motor.items()
    }


def pair(est, truth):
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
    est, truth, meas, motor = load(path)
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
    }, {
        't0': min(t[0], motor[0]['t']) if motor else t[0],
        'errors': (t, ee_err, q_err, dq_err, ee, ee_t),
        'motor': motor_series(motor),
    }


def plot_hardware(runs, out):
    """Per motor: commanded vs feedback current, and feedback shaft velocity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    motor_ids = sorted({mid for data in runs.values() for mid in data})
    if not motor_ids:
        raise SystemExit(f'no {MOTOR_STATE} messages to plot')
    nrows = 3 * len(motor_ids)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 2.2 * nrows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for path, data in runs.items():
        label = Path(path).stem
        t0 = min(m['t'][0] for m in data.values() if len(m['t']))
        for k, mid in enumerate(motor_ids):
            m = data.get(mid)
            if m is None or len(m['t']) == 0:
                continue
            ts = m['t'] - t0
            axes[3 * k].plot(ts, m['commanded_current'], label=f'{label} commanded')
            axes[3 * k + 1].plot(ts, m['current'], label=f'{label} feedback')
            axes[3 * k + 2].plot(ts, m['velocity'], label=f'{label} feedback')
    for k, mid in enumerate(motor_ids):
        axes[3 * k].set_ylabel(f'motor {mid}\ncommanded\ncurrent [A]')
        axes[3 * k + 1].set_ylabel(f'motor {mid}\nfeedback\ncurrent [A]')
        axes[3 * k + 2].set_ylabel(f'motor {mid}\nvelocity [rad/s]')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('time [s]')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--plot', help='save an error-vs-time figure to this path')
    args = parser.parse_args(argv)

    # Hardware runs have no ground truth: plot motor feedback only
    loaded = {path: load(path) for path in args.runs}
    if not all(est and truth for est, truth, _, _ in loaded.values()):
        if not args.plot:
            raise SystemExit('no ground truth in recording; use --plot to plot motor feedback')
        plot_hardware({path: motor_series(motor) for path, (_, _, _, motor) in loaded.items()}, args.plot)
        return

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
        motor_ids = sorted({motor_id for data in series.values() for motor_id in data['motor']})
        nrows = 3 + len(motor_ids)
        fig, axes = plt.subplots(nrows, 1, figsize=(9, max(8, 2.4 * nrows)), sharex=True)
        if nrows == 1:
            axes = [axes]
        run_labels = {path: Path(path).stem for path in series}
        for path, data in series.items():
            t0 = data['t0']
            t, ee_err, q_err, dq_err, *_ = data['errors']
            axes[0].plot(t - t[0], 1e3 * ee_err, label=path)
            axes[1].plot(t - t[0], q_err)
            axes[2].plot(t - t[0], dq_err)
            for row, motor_id in enumerate(motor_ids, start=3):
                motor = data['motor'].get(motor_id)
                if motor is None or len(motor['t']) == 0:
                    continue
                ts = motor['t'] - t0
                label = run_labels[path]
                axes[row].plot(ts, motor['current'], label=f'{label} feedback')
                axes[row].plot(ts, motor['commanded_current'], '--', label=f'{label} commanded')
        axes[0].set_ylabel('ee error [mm]')
        axes[1].set_ylabel('|q error| [rad]')
        axes[2].set_ylabel('|dq error| [rad/s]')
        for row, motor_id in enumerate(motor_ids, start=3):
            axes[row].set_ylabel(f'motor {motor_id}\ncurrent [A]')
            axes[row].grid(True, which='both', alpha=0.3)
            axes[row].legend(fontsize=8)
        axes[-1].set_xlabel('time [s]')
        for ax in axes[:3]:
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f'saved {args.plot}')


if __name__ == '__main__':
    main()
