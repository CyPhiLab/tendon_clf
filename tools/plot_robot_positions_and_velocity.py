#!/usr/bin/env python3
"""Plot task positions per robot and compare mean angular joint velocity."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.2,
})

ROBOTS = ("tendon", "helix", "spirob_horz")
NAMES = {"tendon": "Finger", "spirob_horz": "SpiRob", "helix": "Helix"}
COLORS = {"x": "#0072B2", "y": "#D55E00", "z": "#009E73"}


def load_trace(path):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No trace data in {path}")
    time = np.asarray([float(row["time_s"]) for row in rows])
    actual = np.column_stack([
        [float(row[next(field for field in fields
                        if field.endswith(f"_{axis}_m") and not field.startswith("target_"))])
         for row in rows]
        for axis in "xyz"
    ])
    target = np.column_stack([
        [float(row[f"target_{axis}_m"]) for row in rows]
        for axis in "xyz"
    ])
    qdot_mean = np.asarray([float(row["qvel_angular_mean_rad_s"]) for row in rows])
    lyapunov = np.asarray([float(row["lyapunov_V"]) for row in rows])
    return time, actual, target, qdot_mean, lyapunov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/sequence_clockwise_20261119")
    parser.add_argument("--output", default="plots/sequence_clockwise_20261119/task_positions_and_mean_velocity.png")
    args = parser.parse_args()

    data = {robot: load_trace(Path(args.results_dir) / f"{robot}.csv")
            for robot in ROBOTS}
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True,
                             constrained_layout=True)

    for robot in ROBOTS:
        time, _, _, _, lyapunov = data[robot]
        initial_value = lyapunov[0]
        if not np.isfinite(initial_value) or np.isclose(initial_value, 0.0):
            raise ValueError(f"Cannot normalize Lyapunov value for {robot}: invalid V(0)")
        color = {"tendon": "#0072B2", "helix": "#D55E00", "spirob_horz": "#009E73"}[robot]
        axes[0].plot(time, lyapunov / initial_value, color=color, label=NAMES[robot])
    axes[0].set_ylabel(r"$V(t)/V(0)$")
    axes[0].set_ylim(0, 1.5)
    axes[0].legend(loc="best", ncol=3)

    for index, robot in enumerate(ROBOTS):
        time, actual, target, _, _ = data[robot]
        ax = axes[index + 1]
        for dim, axis in enumerate("xyz"):
            ax.plot(time, actual[:, dim], color=COLORS[axis], label=f"{axis} actual")
            ax.plot(time, target[:, dim], color=COLORS[axis], ls="--", alpha=0.75,
                    label="target" if dim == 0 else "_nolegend_")
        ax.set_ylabel(NAMES[robot])
        if index == 0:
            ax.legend(ncol=2, loc="best")

    for robot in ROBOTS:
        time, _, _, qdot_mean, _ = data[robot]
        scale = np.nanmax(np.abs(qdot_mean))
        if not np.isfinite(scale) or np.isclose(scale, 0.0):
            raise ValueError(f"Cannot normalize mean q_dot for {robot}: no nonzero finite values")
        color = {"tendon": "#0072B2", "helix": "#D55E00", "spirob_horz": "#009E73"}[robot]
        axes[4].plot(time, qdot_mean / scale, color=color, label=NAMES[robot])
    axes[4].set_ylabel(r"$\dot{q}/\max|\dot{q}|$")
    axes[4].set_ylim(-1, 1)
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="best", ncol=3)

    for ax in axes:
        ax.set_xlim(0, 80)
        ax.grid(True, alpha=0.25)
        ax.tick_params(width=1.5, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        for boundary in (20, 40, 60):
            ax.axvline(boundary, color="0.4", ls="--", lw=1.3)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
