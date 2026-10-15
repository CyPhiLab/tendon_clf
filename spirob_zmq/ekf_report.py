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


def marker_series(meas, est):
    """Vicon, MuJoCo-predicted (EKF prior) and EKF-corrected marker positions, (T, n_markers, 3)."""
    def stack(records, key, valid=lambda m: True):
        rows = [(r['t'], r['msg'][key]) for r in records if key in r['msg'] and valid(r['msg'])]
        if not rows:
            return None, None
        t, data = zip(*rows)
        data = np.asarray(data, dtype=float)
        return np.asarray(t, dtype=float), data.reshape(len(t), -1, 3)
    is_valid = lambda m: m.get('is_valid', True)
    return {
        'Vicon': stack(meas, 'data'),
        'MuJoCo predicted': stack(est, 'site_pos_pred', valid=is_valid),
        'EKF corrected': stack(est, 'site_pos', valid=is_valid),
    }


def plot_markers(path, out):
    """One row per marker, one column per axis: Vicon vs MuJoCo prediction vs EKF corrected."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spirob_zmq.robots import DEFAULT_ROBOT, ROBOTS

    est, _, meas, _ = load(path)
    series = marker_series(meas, est)
    t_m, p_m = series['Vicon']
    if p_m is None:
        print(f'{path}: no {SITE_MEASUREMENT} messages; skipping marker plot')
        return
    n = p_m.shape[1]
    series = {k: v for k, v in series.items() if v[1] is not None and v[1].shape[1] == n}
    names = ROBOTS[DEFAULT_ROBOT].get('site_names', [])
    names = names if len(names) == n else [f'marker {i}' for i in range(n)]
    t0 = min(t[0] for t, _ in series.values())

    # EKF corrected is drawn wide and underneath so MuJoCo predicted stays visible when they overlap
    styles = {'Vicon': dict(color='k', lw=0.8, alpha=0.6, zorder=3),
              'MuJoCo predicted': dict(color='tab:orange', lw=1.0, ls=':', zorder=4),
              'EKF corrected': dict(color='tab:blue', lw=2.5, alpha=0.5, zorder=2)}
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.0 * n), sharex=True, squeeze=False)
    for i in range(n):
        for k, axis in enumerate('xyz'):
            ax = axes[i, k]
            for label, (t, p) in series.items():
                ax.plot(t - t0, 1e3 * p[:, i, k], label=label, **styles[label])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f'{axis} [mm]')
        axes[i, 0].set_ylabel(names[i], fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('time [s]')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


def marker_plot_path(plot):
    p = Path(plot)
    return p.with_name(f'{p.stem}_markers{p.suffix or ".png"}')


def plot_tracking(path, out):
    """Error of the tracked site (control_node's site_name) to target_pos over time.

    Shows the Vicon-measured error when the site is a marker, plus the EKF-estimated error.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spirob_zmq.robots import DEFAULT_ROBOT, ROBOTS

    profile = ROBOTS[DEFAULT_ROBOT]
    site = profile.get('site_name', 'ee')
    target = np.asarray(profile['target_pos'], dtype=float)
    names = list(profile.get('site_names', []))

    est, _, meas, _ = load(path)
    series = marker_series(meas, est)
    curves = {}
    if site in names:
        i = names.index(site)
        for label in ('Vicon', 'EKF corrected'):
            t, p = series[label]
            if p is not None and p.shape[1] == len(names):
                curves[label] = (t, p[:, i, :])
    elif site == 'ee':
        rows = [(r['t'], r['msg']['task_pos']) for r in est if r['msg'].get('is_valid', True)]
        if rows:
            t, p = zip(*rows)
            curves['EKF corrected'] = (np.asarray(t, float), np.asarray(p, float))
    if not curves:
        print(f'{path}: no data for tracked site {site!r}; skipping tracking plot')
        return

    t0 = min(t[0] for t, _ in curves.values())
    styles = {'Vicon': dict(color='k', lw=0.8, alpha=0.7),
              'EKF corrected': dict(color='tab:blue', lw=1.2, ls='--')}
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for label, (t, p) in curves.items():
        valid = np.all(np.isfinite(p), axis=1)
        t, err = t[valid] - t0, 1e3 * (p[valid] - target)
        for k, axis in enumerate('xyz'):
            axes[k].plot(t, err[:, k], label=label, **styles[label])
        axes[3].plot(t, np.linalg.norm(err, axis=1), label=label, **styles[label])
        if len(err):
            print(f'  {label:14s} final |error| {np.linalg.norm(err[-1]):.1f} mm')
    for k, axis in enumerate('xyz'):
        axes[k].set_ylabel(f'{axis} error [mm]')
        axes[k].axhline(0.0, color='gray', lw=0.6)
    axes[3].set_ylabel('|error| [mm]')
    axes[0].set_title(f'{site} - target {np.round(target, 3).tolist()}')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('time [s]')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')


def extra_plots(runs, plot):
    """Marker and tracking figures, saved next to the main plot."""
    for path in runs:
        base = Path(plot) if len(runs) == 1 else Path(plot).with_name(f'{Path(path).stem}.png')
        plot_markers(path, marker_plot_path(base))
        plot_tracking(path, base.with_name(f'{base.stem}_tracking.png'))


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
        extra_plots(args.runs, args.plot)
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
        extra_plots(args.runs, args.plot)


if __name__ == '__main__':
    main()
