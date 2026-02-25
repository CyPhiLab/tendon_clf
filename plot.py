import os
import glob
import ast
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import plotting

# Define font style

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",   # math matches Times
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
})


# Naming utilities
def name_mapping(ctrl):
    """
    Convert internal controller name to paper-friendly legend label.
    """

    if ctrl is None:
        return ""

    c = ctrl.lower().strip()

    mapping = {
        "id_clf_qp": "ID-CLF-QP",
        "mpc": "MPC",
        "impedance": "IC",
        "impedance_pd": "IC-PD",
        "impedance_qp": "IC-QP",
    }

    if c in mapping:
        return mapping[c]

    # fallback: Capitalize words but don't scream case
    return "-".join(word.capitalize() for word in c.split("_"))



def robot_naming(robot):
    """
    Convert robot folder name to paper-friendly label:
      - helix  -> Helix
      - tendon -> Finger
      - spirob -> SpiRob
    """
    if robot is None:
        return ""
    r = robot.lower().strip()

    mapping = {
        "tendon": "Finger",
        "helix": "Helix",
        "spirob": "SpiRob"
    }
    return mapping.get(r, r.capitalize())


def legend_above(ax, ncol=None):
    """
    Legend above plot, but always put 'Reference' last.
    """
    handles, labels = ax.get_legend_handles_labels()

    # move Reference to the end
    ordered = sorted(
        zip(handles, labels),
        key=lambda hl: (hl[1] == "Reference", hl[1])
    )

    handles, labels = zip(*ordered)

    if ncol is None:
        ncol = len(labels)

    ax.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=ncol,
        frameon=True,
        facecolor="white"
    )



def finalize_figure(fig, ax):
    """
    Ensure there is room for the outside legend.
    """
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

# CSV parsing
def parse_vec_series(series):
    """Parse strings like '[0.1, 0, 0.2]' into (N,3) numpy array."""
    return np.vstack(series.apply(ast.literal_eval).to_numpy())


def load_single_csv(path):
    df = pd.read_csv(path, comment="#")

    t = df["time"].to_numpy()
    sim_time = df["sim_time"].to_numpy() if "sim_time" in df.columns else None

    V = df["lyapunov_V"].to_numpy() if "lyapunov_V" in df.columns else None
    error = df["task_error"].to_numpy() if "task_error" in df.columns else None

    u_cols = [c for c in df.columns if c.startswith("u")]
    u = df[u_cols].to_numpy() if len(u_cols) > 0 else None

    x = parse_vec_series(df["x_log"]) if "x_log" in df.columns else None
    xd = parse_vec_series(df["xd_log"]) if "xd_log" in df.columns else None

    return t, sim_time, V, error, u, x, xd

# Experiment processing
def read_rtf(path):
    """
    Extract real_time_factor from commented header of CSV.
    Returns NaN if not present.
    """
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "real_time_factor" in line:
                    # line format: # real_time_factor,0.771987
                    return float(line.strip().split(",")[1])
    except Exception:
        pass

    return float("nan")

def set_experiment(files):
    """
    Aggregate multiple SET runs into mean/std on a common time grid.
    - Always produces err_mean/std
    - Produces V_mean/std only if ALL runs contain lyapunov_V
    """
    time_list: List[np.ndarray] = []
    err_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []

    has_V_all = True

    for f in files:
        print("SET:", f)
        t, _, V, e, _, _, _ = load_single_csv(f)
        time_list.append(t)
        err_list.append(e if e is not None else np.zeros_like(t))

        if V is None:
            has_V_all = False
        else:
            V_list.append(V)

    # common time grid
    t_min = max(t[0] for t in time_list)
    t_max = min(t[-1] for t in time_list)
    N = max(len(t) for t in time_list)
    t_grid = np.linspace(t_min, t_max, N)

    err_interp = [np.interp(t_grid, t, e) for t, e in zip(time_list, err_list)]
    result: Dict[str, Any] = {
        "time": t_grid,
        "err_mean": np.mean(err_interp, axis=0),
        "err_std":  np.std(err_interp, axis=0),
    }

    if has_V_all and len(V_list) == len(files):
        V_interp = []
        for t, V in zip(time_list, V_list):
            V0 = V[0] if (V is not None and len(V) > 0 and V[0] != 0) else 1.0
            V_interp.append(np.interp(t_grid, t, V / V0))
        result["V_mean"] = np.mean(V_interp, axis=0)
        result["V_std"]  = np.std(V_interp, axis=0)

    return result


def tracking_experiment(file):
    """
    Single tracking run (no averaging)
    """
    print("TRACK:", file)
    t, sim_time, V, e, _, x, xd = load_single_csv(file)

    time_axis = sim_time if sim_time is not None else t

    out: Dict[str, Any] = {"time": time_axis, "error": e}
    if V is not None:
        V0 = V[0] if (len(V) > 0 and V[0] != 0) else 1.0
        out["V"] = V / V0
    if x is not None and xd is not None:
        out["x"] = x
        out["xd"] = xd
    return out


def load_results(root):
    """
    Structure:
      robots[robot][ctrl]["set"]      -> aggregated dict
      robots[robot][ctrl]["tracking"] -> single-run dict
    """
    robots: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Results folder not found: {root}")

    for robot in sorted(os.listdir(root)):
        robot_path = os.path.join(root, robot)
        if not os.path.isdir(robot_path):
            continue

        robots[robot] = {}

        for ctrl in sorted(os.listdir(robot_path)):
            ctrl_path = os.path.join(robot_path, ctrl)
            if not os.path.isdir(ctrl_path):
                continue

            set_files = sorted(glob.glob(os.path.join(ctrl_path, "set_*.csv")))
            track_files = sorted(glob.glob(os.path.join(ctrl_path, "tracking*.csv")))

            if len(set_files) == 0 and len(track_files) == 0:
                continue

            robots[robot][ctrl] = {}

            if len(set_files) > 0:
                robots[robot][ctrl]["set"] = set_experiment(set_files)

            if len(track_files) > 0:
                robots[robot][ctrl]["tracking"] = tracking_experiment(track_files[0])

    return robots

# Plotting
def clf_plot(robots, control, experiment):

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False

    items = []

    for robot, controllers in robots.items():
        for ctrl, data in controllers.items():

            if experiment not in data:
                continue

            exp_data = data[experiment]

            # key depends on experiment type
            if experiment == "set":
                if "V_mean" not in exp_data:
                    continue
            elif experiment == "tracking":
                if "V" not in exp_data:
                    continue
            else:
                raise ValueError("experiment must be 'set' or 'tracking'")

            if control is not None and ctrl != control:
                continue

            items.append((robot, ctrl, exp_data))

    robot_order = ["tendon", "helix", "spirob"]

    items.sort(
        key=lambda x: robot_order.index(x[0]) if x[0] in robot_order else 999
    )

    for robot, ctrl, exp_data in items:

        if experiment == "set":
            t = exp_data["time"]
            V = exp_data["V_mean"]
            S = exp_data["V_std"]

            line, = ax.plot(t, V, linewidth=4,
                            label=robot_naming(robot), zorder=3)

            ax.fill_between(t, V - S, V + S,
                            color=line.get_color(),
                            alpha=0.25,
                            linewidth=0,
                            zorder=2)

        else:  # tracking
            t = exp_data["time"]
            V = exp_data["V"]

            ax.plot(t, V, linewidth=4,
                    label=robot_naming(robot))

        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"No {experiment.upper()} CLF curves found to plot.")
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Lyapunov Function")
    ax.grid(True)

    if experiment == "set":
        plt.xlim(0, 0.8)
        plt.ylim(0, 2)
    else:
        plt.xlim(0, 2)

    legend_above(ax, ncol=None)
    finalize_figure(fig, ax)
    plt.show()


def plot_tracking_trajectory(robots, robot_list, plane, start_time):
    """
    Plot multiple robots in one figure with shared legend.
    Each robot becomes one subplot.
    """

    plane = plane.lower()
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in idx:
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    i, j = idx[plane]
    axis_names = ["x", "y", "z"]

    n = len(robot_list)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 6.2*n), sharex=False, sharey=False)


    if n == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    # controller ordering
    ctrl_order = ["id_clf_qp", "impedance", "impedance_QP", "mpc"]

    for ax, robot in zip(axes, robot_list):

        controllers = robots[robot]
        plotted_ref = False

        items = list(controllers.items())
        items.sort(key=lambda kv: ctrl_order.index(kv[0]) if kv[0] in ctrl_order else 999)

        for ctrl, data in items:
            if "tracking" not in data:
                continue

            tr = data["tracking"]
            if "x" not in tr or "xd" not in tr:
                continue

            t = tr["time"]
            x = tr["x"]
            xd = tr["xd"]

            mask = t >= start_time
            if np.sum(mask) < 10:
                continue

            x = x[mask]
            xd = xd[mask]

            label = name_mapping(ctrl)
            line, = ax.plot(x[:, i], x[:, j], linewidth=4, label=label)

            # add uniquely
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

            # reference
            if not plotted_ref:
                ref_line, = ax.plot(xd[:, i], xd[:, j], "k--", linewidth=4, label="Reference")
                if "Reference" not in legend_labels:
                    legend_handles.append(ref_line)
                    legend_labels.append("Reference")
                plotted_ref = True


        ax.set_xlabel(f"{axis_names[i]} (m)")
        ax.set_ylabel(f"{axis_names[j]} (m)")
        ax.set_title(robot_naming(robot))
        ax.axis("equal")
        ax.grid(True)


    # ---- reorder so Reference is last ----
    ordered = sorted(
        zip(legend_handles, legend_labels),
        key=lambda hl: (hl[1] == "Reference", hl[1])
    )
    legend_handles, legend_labels = zip(*ordered)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,              
        frameon=True,
        columnspacing=1.6,
        handlelength=2.4,
        handletextpad=0.6,
        borderpad=0.4
    )

    # -------- subplot labels (a), (b) --------
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lab in zip(axes, labels):
        ax.text(
            0.5, -0.30, lab,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=20
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# Reports
def compute_rtf(df):
    return df["sim_time"].iloc[-1] / df["time"].iloc[-1]


def max_control_input(df):
    u_cols = [c for c in df.columns if c.startswith("u")]
    if len(u_cols) == 0:
        return float("nan")
    return float(np.abs(df[u_cols].to_numpy()).max())


def final_error(df):
    return float(df["task_error"].iloc[-1])


def mse_error(df):
    e = df["task_error"].to_numpy()
    return float(np.mean(e**2))


def summarize_set(files):
    finals: List[float] = []
    max_inputs: List[float] = []
    rtfs: List[float] = []

    for f in files:
        df = pd.read_csv(f, comment="#")

        finals.append(final_error(df))
        max_inputs.append(max_control_input(df))

        rtf = read_rtf(f)
        if not np.isnan(rtf):
            rtfs.append(rtf)

    return {
        "final_mean": float(np.mean(finals)),
        "final_std":  float(np.std(finals)),
        "max_input":  float(np.max(max_inputs)),
        "rtf_mean": float(np.mean(rtfs)) if len(rtfs) > 0 else float("nan"),
        "rtf_std":  float(np.std(rtfs))  if len(rtfs) > 0 else float("nan"),
    }


def summarize_tracking(file):
    df = pd.read_csv(file, comment="#")

    rtf = read_rtf(file)
    if np.isnan(rtf):  # fallback safety
        rtf = compute_rtf(df)

    return {
        "mse": mse_error(df),
        "rtf": rtf,
        "max_input": max_control_input(df)
    }

control_limits = {
    "tendon": "-1 ≤ u ≤ 1",
    "helix": "-25 ≤ u ≤ 25",
    "spirob": "u ≤ 0"
}


def generate_report(root):
    set_rows = []
    track_rows = []

    for robot in sorted(os.listdir(root)):
        robot_path = os.path.join(root, robot)
        if not os.path.isdir(robot_path):
            continue

        for ctrl in sorted(os.listdir(robot_path)):
            ctrl_path = os.path.join(robot_path, ctrl)
            if not os.path.isdir(ctrl_path):
                continue

            set_files = sorted(glob.glob(os.path.join(ctrl_path, "set_*.csv")))
            track_files = sorted(glob.glob(os.path.join(ctrl_path, "tracking*.csv")))

            if len(set_files) > 0:
                s = summarize_set(set_files)
                set_rows.append([
                robot_naming(robot),
                name_mapping(ctrl),
                "Set Point",
                f"{s['final_mean']:.4f} ± {s['final_std']:.4f}",
                f"{s['rtf_mean']:.2f} ± {s['rtf_std']:.2f}x",
                f"{s['max_input']:.3f}",
                control_limits.get(robot, "")
            ])


            if len(track_files) > 0:
                t = summarize_tracking(track_files[0])
                track_rows.append([
                    robot_naming(robot),
                    name_mapping(ctrl),
                    "Trajectory Tracking",
                    f"{t['mse']:.5f}",
                    f"{t['rtf']:.2f}x",
                    f"{t['max_input']:.3f}",
                    control_limits.get(robot, "")
                ])

    set_df = pd.DataFrame(set_rows, columns=[
    "Robot", "Controller", "Experiment",
    "Mean Final Error ± std", "% Real-time", "Max Control Input", "Control Limit"
    ])



    track_df = pd.DataFrame(track_rows, columns=[
        "Robot", "Controller", "Experiment",
        "MSE", "% Real-time", "Max Input", "Control Limit"
    ])

    set_df.to_csv("set_report.csv", index=False)
    track_df.to_csv("tracking_report.csv", index=False)

    print("\nGenerated:")
    print("  set_report.csv")
    print("  tracking_report.csv")


if __name__ == "__main__":
    robots = load_results("results")

    # Lyapunov plotting
    clf_plot(robots, control="id_clf_qp", experiment="set")
    clf_plot(robots, control="id_clf_qp", experiment="tracking")

    # Trajectory circles
    plot_tracking_trajectory(
    robots,
    robot_list=["tendon", "helix", "spirob"],
    plane="xz",
    start_time=8.0
)

    # Reports
    generate_report("results")
