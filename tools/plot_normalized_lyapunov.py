#!/usr/bin/env python3
"""Compare sequential-run Lyapunov values normalized by each run's initial value."""
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
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.4,
})


def load_normalized_trace(path):
    with Path(path).open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No data in {path}")
    time = np.asarray([float(row["time_s"]) for row in rows])
    value = np.asarray([float(row["lyapunov_V"]) for row in rows])
    if not np.isfinite(value[0]) or value[0] == 0:
        raise ValueError(f"Invalid initial Lyapunov value in {path}")
    return time, value / value[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/sequence_clockwise_20261119")
    parser.add_argument("--output", default="plots/sequence_clockwise_20261119/normalized_lyapunov_comparison.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    colors = {"tendon": "#0072B2", "spirob_horz": "#D55E00", "helix": "#009E73"}
    labels = {"tendon": "Tendon", "spirob_horz": "SpiRob", "helix": "Helix"}
    for robot in ("tendon", "spirob_horz", "helix"):
        time, normalized = load_normalized_trace(Path(args.results_dir) / f"{robot}.csv")
        ax.plot(time, normalized, color=colors[robot], label=labels[robot])

    for boundary in (20, 40, 60):
        ax.axvline(boundary, color="0.4", ls="--", lw=1.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Normalized Lyapunov value, $V(t)/V(0)$")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    ax.tick_params(width=1.5, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
