"""Paper figures for the hardware set-point experiments (EKF + CLF-QP velocity control).

Usage: python tools/hardware_figs/make_figs.py   (after extract.py and render_poses.py)
Writes PDF + PNG figures, a per-run metrics CSV and a LaTeX summary table to ./figures.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runs import BASE, T_ON, T_SS, load_runs, setpoints

OUT = Path(__file__).with_name('figures')
OUT.mkdir(exist_ok=True)

# IEEE column widths (in)
COL, PAGE = 3.5, 7.16
# categorical slots 1-5 of the validated reference palette (light), fixed order P1..P5
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, INK3, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.2, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.constrained_layout.use': True,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', OUT / f'{name}.pdf')


def panel_label(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold', color=INK)


def resample(runs, attr, grid):
    return np.array([np.interp(grid, r.t - T_ON, getattr(r, attr)) for r in runs])


def ring_centroids(r, lo, hi):
    m = r.d['markers']
    sel = (r.d['markers_t'] >= lo) & (r.d['markers_t'] <= hi)
    return m[sel].mean(0).reshape(-1, 3, 3).mean(1)          # (n_rings, 3)


def skeleton(r, lo, hi):
    return np.vstack([BASE, ring_centroids(r, lo, hi)])


# --------------------------------------------------------------------------------------------
def draw_workspace(ax_side, ax_top, runs, sps, labels):
    # (a) workspace: initial shape (grey), per-trial tip paths, mean final shape, targets
    init = np.mean([skeleton(r, 0, T_ON - 0.5) for r in runs], axis=0)
    for ax, (i, j) in ((ax_side, (1, 2)), (ax_top, (1, 0))):
        ax.plot(1e3 * init[:, i], 1e3 * init[:, j], '-o', color=INK3, ms=2.5, lw=1.4, zorder=2)
        for k, sp in enumerate(sps):
            rr = [r for r in runs if r.setpoint == sp]
            for r in rr:
                live = r.window(T_ON, T_SS[1])
                ax.plot(1e3 * r.tip[live, i], 1e3 * r.tip[live, j], color=COLORS[k], lw=0.35, alpha=0.45,
                        zorder=3)
            fin = np.mean([skeleton(r, *T_SS) for r in rr], axis=0)
            ax.plot(1e3 * fin[:, i], 1e3 * fin[:, j], '-', color=COLORS[k], lw=1.1, zorder=4)
            tgt = rr[0].target
            ax.plot(1e3 * tgt[i], 1e3 * tgt[j], 'x', color=INK, ms=4.5, mew=1.0, zorder=6)
        ax.plot(0, 1e3 * BASE[j], 's', color=INK, ms=4, zorder=5)
        ax.set_aspect('equal')
        ax.grid(False)
    ax_side.set_ylabel('$z$ [mm]')
    ax_top.set_ylabel('$x$ [mm]')
    ax_top.set_xlabel('$y$ [mm]')
    ax_side.tick_params(labelbottom=False)
    ax_side.set_xlim(-10, 350)
    ax_top.set_xlim(-10, 350)
    ax_side.set_ylim(15, 215)
    ax_top.set_ylim(-85, 85)
    # direct target labels: offsets chosen so labels sit clear of the paths
    # P1, P2, P5 are labelled in the side view; P3/P4 coincide there and are separated in the top view
    side_off = {0: (2, -11), 1: (-12, -11), 4: (-14, 3)}
    top_off = {2: (-16, 0), 3: (-16, -6)}
    for k, sp in enumerate(sps):
        tgt = 1e3 * [r for r in runs if r.setpoint == sp][0].target
        if k in side_off:
            ax_side.annotate(labels[k], (tgt[1], tgt[2]), xytext=side_off[k], textcoords='offset points',
                             fontsize=7, color=INK)
        if k in top_off:
            ax_top.annotate(labels[k], (tgt[1], tgt[0]), xytext=top_off[k], textcoords='offset points',
                            fontsize=7, color=INK)
    tgt = 1e3 * [r for r in runs if r.setpoint == sps[2]][0].target
    ax_side.annotate('P3,P4', (tgt[1], tgt[2]), xytext=(7, 1), textcoords='offset points', fontsize=7, color=INK)
    ax_side.annotate('start', (1e3 * init[-1, 1], 1e3 * init[-1, 2]), xytext=(-5, -9), textcoords='offset points',
                     fontsize=7, color=INK2, ha='right')
    ax_side.annotate('base', (0, 1e3 * BASE[2]), xytext=(4, 5), textcoords='offset points', fontsize=7,
                     color=INK2)



def draw_error_time(ax_err, runs, sps, labels):
    # (b) tip error norm vs time since the controller is enabled: median and IQR over trials
    grid = np.linspace(-0.5, T_SS[1] - T_ON, 800)
    for k, sp in enumerate(sps):
        rr = [r for r in runs if r.setpoint == sp]
        e = 1e3 * resample(rr, 'err', grid)
        lo, med, hi = np.percentile(e, [25, 50, 75], axis=0)
        ax_err.fill_between(grid, lo, hi, color=COLORS[k], alpha=0.18, lw=0)
        ax_err.plot(grid, med, color=COLORS[k], lw=1.2, label=f'{labels[k]} ($n$={len(rr)})')
    ax_err.set_yscale('log')
    ax_err.set_ylim(2, 250)
    ax_err.set_yticks([2, 5, 10, 20, 50, 100, 200])
    ax_err.set_yticklabels(['2', '5', '10', '20', '50', '100', '200'])
    ax_err.set_xlim(grid[0], grid[-1])
    ax_err.set_xlabel('time since controller start [s]')
    ax_err.set_ylabel(r'tip error $\|p - p^\star\|$ [mm]')
    ax_err.axvspan(T_SS[0] - T_ON, T_SS[1] - T_ON, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax_err.text((T_SS[0] + T_SS[1]) / 2 - T_ON, 200, 'steady-state\nwindow', ha='center', va='top', fontsize=6.5,
                color=INK2)
    ax_err.legend(loc='center right', frameon=False, handlelength=1.2, borderaxespad=0.2,
                  bbox_to_anchor=(1.0, 0.62))



def draw_steady_state(ax_ss, runs, sps, labels):
    # (c) steady-state error per trial (Vicon), median tick
    rng = np.random.default_rng(0)
    for k, sp in enumerate(sps):
        ss = np.array([1e3 * r.err[r.window(*T_SS)].mean() for r in runs if r.setpoint == sp])
        x = k + rng.uniform(-0.17, 0.17, len(ss))
        ax_ss.scatter(x, ss, s=9, color=COLORS[k], edgecolors='white', linewidths=0.4, zorder=3)
        ax_ss.plot([k - 0.3, k + 0.3], [np.median(ss)] * 2, color=INK, lw=1.0, zorder=4)
    allss = np.array([1e3 * r.err[r.window(*T_SS)].mean() for r in runs])
    ax_ss.axhline(np.median(allss), color=INK3, lw=0.6, ls='--', zorder=1)
    ax_ss.text(-0.55, np.median(allss) + 0.3, f'all trials: {np.median(allss):.1f}', ha='left', va='bottom',
               fontsize=6.5, color=INK2)
    ax_ss.set_xticks(range(len(sps)))
    ax_ss.set_xticklabels(labels)
    ax_ss.set_xlim(-0.6, len(sps) - 0.4)
    ax_ss.set_ylim(0, None)
    ax_ss.grid(axis='x', visible=False)
    ax_ss.set_ylabel('steady-state tip error [mm]')
    ax_ss.set_xlabel('set point')


def draw_renders(fig, spec, labels):
    """Row of MuJoCo renders (render_poses.py), one per set point, labelled underneath."""
    sub = spec.subgridspec(1, len(labels), wspace=0.0)
    axes = []
    for k, label in enumerate(labels):
        ax = fig.add_subplot(sub[0, k])
        ax.imshow(plt.imread(OUT / f'render_{label}.png'), interpolation='antialiased')
        ax.set_axis_off()
        ax.text(0.10, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontsize=8, color=INK)
        axes.append(ax)
    return axes


def fig_tracking(runs, sps, labels):
    """MuJoCo renders of the reached poses | tip error vs time | steady-state error per set point."""
    fig = plt.figure(figsize=(PAGE, 3.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.3], width_ratios=[1.55, 1])
    renders = draw_renders(fig, gs[0, :], labels)
    ax_err, ax_ss = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    draw_error_time(ax_err, runs, sps, labels)
    draw_steady_state(ax_ss, runs, sps, labels)
    panel_label(renders[0], '(a)', x=0.02, y=0.82)
    panel_label(ax_err, '(b)', x=-0.08)
    panel_label(ax_ss, '(c)', x=-0.12)
    save(fig, 'fig_hw_setpoint_tracking')


def fig_tracking_markers(runs, sps, labels):
    """Alternative to fig_tracking with Vicon marker projections instead of renders."""
    fig = plt.figure(figsize=(PAGE, 2.25))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 0.8], height_ratios=[1.35, 1])
    ax_side, ax_top = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_err, ax_ss = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])
    draw_workspace(ax_side, ax_top, runs, sps, labels)
    draw_error_time(ax_err, runs, sps, labels)
    draw_steady_state(ax_ss, runs, sps, labels)
    panel_label(ax_side, '(a)', x=-0.18)
    panel_label(ax_err, '(b)', x=-0.1)
    panel_label(ax_ss, '(c)', x=-0.16)
    save(fig, 'fig_hw_setpoint_tracking_markers')


# --------------------------------------------------------------------------------------------
def fig_estimation(runs, sps, labels):
    """|EKF tip estimate - Vicon tip| over time, median/IQR per set point."""
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    grid = np.linspace(-T_ON, T_SS[1] - T_ON, 1000)
    for k, sp in enumerate(sps):
        rr = [r for r in runs if r.setpoint == sp]
        e = 1e3 * resample(rr, 'est_err', grid)
        lo, med, hi = np.percentile(e, [25, 50, 75], axis=0)
        ax.fill_between(grid, lo, hi, color=COLORS[k], alpha=0.18, lw=0)
        ax.plot(grid, med, color=COLORS[k], lw=1.0, label=labels[k])
    ax.axvline(0, color=INK2, lw=0.6, ls='--')
    ax.text(0.15, 33, 'controller on', fontsize=6.5, color=INK2, va='top')
    ax.text(-T_ON + 0.1, 33, 'robot at rest', fontsize=6.5, color=INK2, va='top')
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 35)
    ax.set_xlabel('time since controller start [s]')
    ax.set_ylabel(r'$\|\hat p - p_{\mathrm{Vicon}}\|$ [mm]')
    ax.legend(ncol=5, loc='lower center', frameon=False, handlelength=1.0, columnspacing=0.8,
              bbox_to_anchor=(0.5, 0.98))
    save(fig, 'fig_hw_estimation_error')


# --------------------------------------------------------------------------------------------
def fig_example(runs, sps, labels, k=4):
    """One representative trial: tip coordinates vs target and the three tendon currents."""
    rr = [r for r in runs if r.setpoint == sps[k]]
    ss = np.array([r.err[r.window(*T_SS)].mean() for r in rr])
    r = rr[np.argsort(ss)[len(ss) // 2]]          # median steady-state error
    print(f'example trial: {r.setpoint}/{r.stamp}, ss error {1e3 * ss[np.argsort(ss)[len(ss) // 2]]:.1f} mm')
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.4), sharex=True, height_ratios=[1.3, 1])
    t = r.t - T_ON
    live = (t >= -1) & (t <= T_SS[1] - T_ON)
    axis_colors = COLORS[:3]
    for j, name in enumerate('xyz'):
        a1.plot(t[live], 1e3 * r.tip[live, j], color=axis_colors[j], lw=1.1, label=f'${name}$')
        a1.plot(t[live], 1e3 * r.tip_est[live, j], color=axis_colors[j], lw=0.8, ls=':')
        a1.axhline(1e3 * r.target[j], color=axis_colors[j], lw=0.6, ls='--', alpha=0.9)
        a1.text(T_SS[1] - T_ON + 0.1, 1e3 * r.tip[live, j][-1], f'${name}$', va='center', fontsize=7, color=INK)
    a1.set_ylabel('tip position [mm]')
    a1.plot([], [], color=INK2, lw=1.1, label='Vicon')
    a1.plot([], [], color=INK2, lw=0.8, ls=':', label='EKF')
    a1.plot([], [], color=INK2, lw=0.6, ls='--', label='target')
    h, l = a1.get_legend_handles_labels()
    a1.legend(h[3:], l[3:], ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.18),
              handlelength=1.6, columnspacing=1.0)
    fb = r.d['fb_t'] - T_ON
    sel = (fb >= -1) & (fb <= T_SS[1] - T_ON)
    for j in range(3):
        a2.plot(fb[sel], -r.d['current'][sel, j], color=axis_colors[j], lw=0.6, label=f'tendon {j + 1}')
    a2.axhline(0.1, color=INK3, lw=0.6, ls='--')
    a2.axhline(1.9, color=INK3, lw=0.6, ls='--')
    a2.set_ylim(0, 2.0)
    a2.set_ylabel('current [A]')
    a2.set_xlabel('time since controller start [s]')
    a2.set_xlim(-1, T_SS[1] - T_ON + 0.6)
    a2.legend(ncol=3, frameon=False, loc='upper right', handlelength=1.0, columnspacing=0.8)
    save(fig, 'fig_hw_example_trial')


# --------------------------------------------------------------------------------------------
def table(runs, sps, labels):
    rows = [r.metrics() for r in runs]
    df = pd.DataFrame(rows)
    df['label'] = df.setpoint.map(dict(zip(sps, labels)))
    flat = df.drop(columns=['ss_bias_mm', 'peak_current_a', 'ss_current_a'])
    flat.sort_values(['label', 'stamp']).to_csv(OUT / 'hw_setpoint_metrics.csv', index=False, float_format='%.4f')

    def ms(x):
        return f'{np.mean(x):.1f} $\\pm$ {np.std(x, ddof=1):.1f}'

    lines = [r'\begin{tabular}{lcccccc}', r'\toprule',
             r'Set point & target $(x,y,z)$ [mm] & $n$ & $e_0$ [mm] & $t_{90}$ [s] & $e_{ss}$ [mm] & '
             r'$\|\hat p - p\|_{ss}$ [mm] \\', r'\midrule']
    for k, sp in enumerate(sps):
        g = df[df.setpoint == sp]
        tgt = ', '.join(f'{1e3 * v:.0f}' for v in g.iloc[0:1].pipe(lambda _: [float(s) for s in sp.split('_')]))
        lines.append(f'{labels[k]} & ({tgt}) & {len(g)} & {g.e0_mm.mean():.0f} & '
                     f'{np.nanmedian(g.t90_s):.2f} & {ms(g.ss_err_mm)} & {ms(g.est_err_ss_mm)} \\\\')
    lines += [r'\midrule',
              f'All & -- & {len(df)} & -- & {np.nanmedian(df.t90_s):.2f} & {ms(df.ss_err_mm)} & '
              f'{ms(df.est_err_ss_mm)} \\\\', r'\bottomrule', r'\end{tabular}']
    (OUT / 'hw_setpoint_table.tex').write_text('\n'.join(lines) + '\n')
    print('saved', OUT / 'hw_setpoint_table.tex')
    return df


def main():
    runs = load_runs()
    sps = setpoints(runs)
    labels = [f'P{k + 1}' for k in range(len(sps))]
    print(f'{len(runs)} complete runs;', dict(zip(labels, sps)))
    fig_tracking(runs, sps, labels)
    fig_tracking_markers(runs, sps, labels)
    fig_estimation(runs, sps, labels)
    fig_example(runs, sps, labels)
    df = table(runs, sps, labels)
    print(df.groupby('label')[['e0_mm', 't90_s', 'ss_err_mm', 'est_err_ss_mm', 'sat_frac']].median().round(2))


if __name__ == '__main__':
    main()
