"""Report plots from evaluate.py results: per run, motor angles (recorded vs options) and marker error."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'A': '#c0392b', 'B': '#8e44ad', 'C': '#e67e22', 'D': '#16a085', 'E': '#2c6fbb', 'EBu250': '#2e9d4f', 'EBs250': '#2e9d4f'}
IDS = [10, 1, 2]


def short(run):
    return run.replace('20261103_', '').replace('target_', 'target ')


def plot_run(res, run, names, path):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8.5))
    first = res[names[0]][run]
    t = np.array(first['t']); rec = np.array(first['rec'])
    for j, ax in enumerate(axes[:3]):
        ax.plot(t, rec[:, j], color='k', lw=1.6, label='recorded')
        for n in names:
            d = res[n][run]
            ax.plot(d['t'], np.array(d['sim'])[:, j], color=COLORS.get(n, COLORS.get(n[0])), lw=1.0, ls='--', label=n)
        ax.set_ylabel(f'motor {IDS[j]}\nangle [rad]')
        ax.grid(alpha=0.3)
        lo, hi = rec[:, j].min(), rec[:, j].max()
        pad = max(0.3, 0.5 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].legend(loc='upper right', fontsize=8, ncol=len(names) + 1)
    ax = axes[3]
    for n in names:
        d = res[n][run]
        ax.plot(d['mt'], 1e3 * np.array(d['err']), color=COLORS.get(n, COLORS.get(n[0])), lw=1.0,
                label=f"{n} ({1e3 * d['marker_rms']:.0f} mm rms)")
    ax.set_ylabel('marker error\nrms over markers [mm]')
    ax.set_xlabel('time [s]')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.set_ylim(bottom=0)
    fig.suptitle(f'{short(run)}: open-loop replay of the recorded currents', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    res = json.load(open(sys.argv[1])); out = sys.argv[2]; names = sys.argv[3].split(',')
    os.makedirs(out, exist_ok=True)
    runs = [r for r in res[names[0]] if not r.startswith('_')]
    for run in runs:
        plot_run(res, run, names, os.path.join(out, f'replay_{short(run).replace(" ", "_")}.png'))
    print('plots in', out)
