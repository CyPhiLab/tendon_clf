#!/usr/bin/env python3
"""Plot diagnostics from a spirob_clf.py sequential set-point CSV trace."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.2,
})


def read_trace(path):
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No trace rows in {path}")

    def series(name):
        return np.asarray([float(row[name]) for row in rows])

    time_s = series("time_s")
    position_cols = [next(name for name in fields
                          if name.endswith(f"_{axis}_m") and not name.startswith("target_"))
                     for axis in "xyz"]
    actual = np.column_stack([series(name) for name in position_cols])
    target = np.column_stack([series(f"target_{axis}_m") for axis in "xyz"])
    def velocity_group(mean_name, std_name):
        if mean_name not in fields:
            return None
        return {
            "mean": series(mean_name),
            "std": series(std_name),
        }

    angular = velocity_group("qvel_angular_mean_rad_s", "qvel_angular_std_rad_s")
    linear = velocity_group("qvel_linear_mean_m_s", "qvel_linear_std_m_s")
    return rows, time_s, actual, target, angular, linear


def plot_trace(path, output=None):
    rows, t, actual, target, angular, linear = read_trace(path)
    path = Path(path)
    if output is None:
        output = Path("plots") / f"{path.stem}_diagnostics.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    velocity_panels = []
    if linear is not None and np.any(np.isfinite(linear["mean"])):
        velocity_panels.append((linear, "Mean Generalized Linear Joint Velocity (m/s)"))
    if angular is not None and np.any(np.isfinite(angular["mean"])):
        velocity_panels.append((angular, r"$\dot{q}$ (rad/s)"))

    n_panels = 3 + len(velocity_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.0 * n_panels), sharex=True,
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    axes[0].plot(t, [float(row["lyapunov_V"]) for row in rows], color="#343a40", lw=2.4)
    axes[0].set_ylabel("Lyapunov V")

    colors = ("#0072B2", "#D55E00", "#009E73")
    for i, axis in enumerate("xyz"):
        axes[1].plot(t, actual[:, i], color=colors[i], label=f"{axis} actual")
        axes[1].plot(t, target[:, i], color=colors[i], ls="--", alpha=0.7,
                     label=f"{axis} target")
    axes[1].set_ylabel("Task position (m)")
    axes[1].legend(ncol=3, loc="best")

    error_mm = np.asarray([float(row["position_error_m"]) for row in rows]) * 1000.0
    axes[2].plot(t, error_mm, color="#CC79A7", lw=2.4)
    axes[2].set_ylabel("Position error (mm)")

    for ax, (group, ylabel) in zip(axes[3:], velocity_panels):
        mean, std = group["mean"], group["std"]
        valid = np.isfinite(mean) & np.isfinite(std)
        if not np.any(valid):
            continue
        ax.plot(t[valid], mean[valid], color="#0072B2", lw=2.2, label="Mean across DOFs")
        ax.fill_between(t[valid], mean[valid] - std[valid], mean[valid] + std[valid],
                        color="#0072B2", alpha=0.2,
                        label="Mean ± standard deviation")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(width=1.5, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        for boundary in (20, 40, 60):
            if t[0] < boundary < t[-1]:
                ax.axvline(boundary, color="0.4", ls="--", lw=1.3)
    for i, ax in enumerate(axes):
        for j, label in enumerate(("P1", "P2", "P3", "P4")):
            midpoint = 10 + 20 * j
            if midpoint <= t[-1]:
                ax.text(midpoint, 0.98, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=12, color="0.35")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="One or more sequential trace CSV files")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory for diagnostic PNGs")
    args = parser.parse_args()
    for trace in args.csv:
        plot_trace(trace, Path(args.output_dir) / f"{Path(trace).stem}_diagnostics.png")


if __name__ == "__main__":
    main()
