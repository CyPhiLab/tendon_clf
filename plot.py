import os
import glob
import ast
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Paper style
# ============================================================

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


# ============================================================
# Naming / legend utilities
# ============================================================

def pretty_controller_name(ctrl: str) -> str:
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



def pretty_robot_name(robot: str) -> str:
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



def finalize_figure(fig: plt.Figure, ax: plt.Axes) -> None:
    """
    Ensure there is room for the outside legend.
    """
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)


# ============================================================
# CSV parsing
# ============================================================

def parse_vec_series(series: pd.Series) -> np.ndarray:
    """Parse strings like '[0.1, 0, 0.2]' into (N,3) numpy array."""
    return np.vstack(series.apply(ast.literal_eval).to_numpy())


def load_single_csv(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
                                        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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


# ============================================================
# Experiment processing
# ============================================================
def read_rtf_from_header(path: str) -> float:
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

def process_set_experiment(files: List[str]) -> Dict[str, Any]:
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


def process_tracking_experiment(file: str) -> Dict[str, Any]:
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


def load_all_results(root: str = "results") -> Dict[str, Dict[str, Dict[str, Any]]]:
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
                robots[robot][ctrl]["set"] = process_set_experiment(set_files)

            if len(track_files) > 0:
                robots[robot][ctrl]["tracking"] = process_tracking_experiment(track_files[0])

    return robots


# ============================================================
# Plotting
# ============================================================
def plot_set_clf_all_robots_one_plot(
    robots: Dict[str, Dict[str, Dict[str, Any]]],
    controller_filter: Optional[str] = None
) -> None:
    """
    One plot, multiple robot curves (SET experiment).
    Uses mean/std shading if V_mean exists.
    Legend above plot, paper labels.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False

    # -------------------------------------------------
    # collect curves first
    # -------------------------------------------------
    items = []
    for robot, controllers in robots.items():
        for ctrl, data in controllers.items():
            if "set" not in data:
                continue
            if "V_mean" not in data["set"]:
                continue
            if controller_filter is not None and ctrl != controller_filter:
                continue

            items.append((robot, ctrl, data))

    # -------------------------------------------------
    # deterministic ROBOT ordering: Finger → Helix → SpiRob
    # (based on folder names)
    # -------------------------------------------------
    ROBOT_ORDER_INTERNAL = ["tendon", "helix", "spirob"]

    items.sort(
        key=lambda x: ROBOT_ORDER_INTERNAL.index(x[0]) if x[0] in ROBOT_ORDER_INTERNAL else 999
    )

    # -------------------------------------------------
    # plot AFTER sorting
    # -------------------------------------------------
    for robot, ctrl, data in items:
        t = data["set"]["time"]
        V = data["set"]["V_mean"]
        S = data["set"]["V_std"]

        label = f"{pretty_robot_name(robot)}"
        line, = ax.plot(t, V, linewidth=4, label=label, zorder=3)
        color = line.get_color()

        ax.fill_between(t, V - S, V + S, color=color, alpha=0.25, linewidth=0, zorder=2)

        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("No SET CLF curves found to plot.")
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Lyapunov Function")
    ax.grid(True)

    legend_above(ax, ncol=None)
    finalize_figure(fig, ax)
    plt.show()


def plot_tracking_xy_circles_combined(
    robots: Dict[str, Dict[str, Dict[str, Any]]],
    robot_list: List[str],
    plane: str = "xz",
    start_time: float = 8.0
):
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
    # fig, axes = plt.subplots(1, n, figsize=(6.2*n, 6.2), sharex=False, sharey=False)
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

            label = pretty_controller_name(ctrl)
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
        ax.set_title(pretty_robot_name(robot))
        ax.axis("equal")
        ax.grid(True)

    # -------- Shared legend --------
    # fig.legend(
    #     legend_handles,
    #     legend_labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.0),
    #     ncol=len(legend_labels),
    #     frameon=True
    # )

    # ---- reorder so Reference is last ----
    # ---- reorder so Reference is last ----
    ordered = sorted(
        zip(legend_handles, legend_labels),
        key=lambda hl: (hl[1] == "Reference", hl[1])
    )
    legend_handles, legend_labels = zip(*ordered)

    # ---- three column legend ----
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,                 # <<< NOW 3 COLUMNS
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


def plot_tracking_clf_all_robots_one_plot(
    robots: Dict[str, Dict[str, Dict[str, Any]]],
    controller_filter: Optional[str] = None,
    xlim: Tuple[float, float] = (0.0, 4.0)
) -> None:
    """
    One plot, multiple robot curves (TRACKING experiment).
    Legend above plot, paper labels.
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False

    # -------------------------------------------------
    # flatten all curves into a single list
    # -------------------------------------------------
    items = []
    for robot, controllers in robots.items():
        for ctrl, data in controllers.items():
            if "tracking" not in data:
                continue
            if "V" not in data["tracking"]:
                continue
            if controller_filter is not None and ctrl != controller_filter:
                continue
            items.append((robot, ctrl, data))

    # -------------------------------------------------
    # deterministic ROBOT ordering: Finger → Helix → SpiRob
    # -------------------------------------------------
    robot_order = ["tendon", "helix", "spirob"]   # folder names (not pretty names)

    items.sort(
        key=lambda x: robot_order.index(x[0]) if x[0] in robot_order else 999
    )

    # -------------------------------------------------
    # plot
    # -------------------------------------------------
    for robot, ctrl, data in items:
        t = data["tracking"]["time"]
        V = data["tracking"]["V"]

        label = pretty_robot_name(robot)
        ax.plot(t, V, linewidth=4, label=label)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("No TRACKING CLF curves found to plot.")
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Lyapunov Function ")
    ax.grid(True)
    ax.set_xlim(*xlim)

    legend_above(ax, ncol=None)
    finalize_figure(fig, ax)
    plt.show()



def plot_tracking_xy_circles(
    robots: Dict[str, Dict[str, Dict[str, Any]]],
    plane: str = "xz",
    start_time: float = 8.0
) -> None:
    """
    For EACH ROBOT:
      - one figure
      - all controllers overlayed
      - plot only after t >= start_time
      - legend above plot
      - controller labels: ID_CLF_QP, MPC, IC
      - reference label: Reference
    """
    plane = plane.lower()
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in idx:
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    i, j = idx[plane]
    axis_names = ["x", "y", "z"]

    for robot, controllers in robots.items():
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        plotted_any = False
        plotted_ref = False

        # deterministic legend order: MPC, IC, ID_CLF_QP (then Reference)
        ctrl_order = ["id_clf_qp", "impedance", "impedance_QP", "mpc"]
        items = list(controllers.items())
        items.sort(key=lambda kv: ctrl_order.index(kv[0]) if kv[0] in ctrl_order else 999)

        for ctrl, data in items:
            if "tracking" not in data:
                continue

            tr = data["tracking"]
            if "x" not in tr or "xd" not in tr:
                print(f"Skipping {robot}-{ctrl}: no x_log/xd_log")
                continue

            t = tr["time"]
            x = tr["x"]
            xd = tr["xd"]

            mask = t >= start_time
            if np.sum(mask) < 10:
                print(f"Skipping {robot}-{ctrl}: not enough data after {start_time}s")
                continue

            x = x[mask]
            xd = xd[mask]

            # controller trajectory
            ax.plot(
                x[:, i], x[:, j],
                linewidth=2.8,
                label=pretty_controller_name(ctrl)
            )

            # reference plotted once
            if not plotted_ref:
                ax.plot(
                    xd[:, i], xd[:, j],
                    "k--", linewidth=4.0,
                    label="Reference"
                )
                plotted_ref = True

            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            print(f"No tracking trajectories for {robot}")
            continue

        ax.set_xlabel(f"{axis_names[i]} (m)")
        ax.set_ylabel(f"{axis_names[j]} (m)")
        ax.grid(True)
        ax.axis("equal")

        legend_above(ax, ncol=None)
        finalize_figure(fig, ax)
        plt.show()


# ============================================================
# Reports
# ============================================================

def compute_rtf(df: pd.DataFrame) -> float:
    return df["sim_time"].iloc[-1] / df["time"].iloc[-1]


def max_control_input(df: pd.DataFrame) -> float:
    u_cols = [c for c in df.columns if c.startswith("u")]
    if len(u_cols) == 0:
        return float("nan")
    return float(np.abs(df[u_cols].to_numpy()).max())


def final_error(df: pd.DataFrame) -> float:
    return float(df["task_error"].iloc[-1])


def mse_error(df: pd.DataFrame) -> float:
    e = df["task_error"].to_numpy()
    return float(np.mean(e**2))


def summarize_set(files: List[str]) -> Dict[str, float]:
    finals: List[float] = []
    max_inputs: List[float] = []
    rtfs: List[float] = []

    for f in files:
        df = pd.read_csv(f, comment="#")

        finals.append(final_error(df))
        max_inputs.append(max_control_input(df))

        rtf = read_rtf_from_header(f)
        if not np.isnan(rtf):
            rtfs.append(rtf)

    return {
        "final_mean": float(np.mean(finals)),
        "final_std":  float(np.std(finals)),
        "max_input":  float(np.max(max_inputs)),
        "rtf_mean": float(np.mean(rtfs)) if len(rtfs) > 0 else float("nan"),
        "rtf_std":  float(np.std(rtfs))  if len(rtfs) > 0 else float("nan"),
    }


def summarize_tracking(file: str) -> Dict[str, float]:
    df = pd.read_csv(file, comment="#")

    rtf = read_rtf_from_header(file)
    if np.isnan(rtf):  # fallback safety
        rtf = compute_rtf(df)

    return {
        "mse": mse_error(df),
        "rtf": rtf,
        "max_input": max_control_input(df)
    }




CONTROL_LIMITS = {
    "tendon": "-1 ≤ u ≤ 1",
    "helix": "-25 ≤ u ≤ 25",
    "spirob": "u ≤ 0"
}


def generate_report(root: str = "results") -> None:
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
                pretty_robot_name(robot),
                pretty_controller_name(ctrl),
                "Static Point Tracking",
                f"{s['final_mean']:.4f} ± {s['final_std']:.4f}",
                f"{s['rtf_mean']:.2f} ± {s['rtf_std']:.2f}x",
                f"{s['max_input']:.3f}",
                CONTROL_LIMITS.get(robot, "")
            ])


            if len(track_files) > 0:
                t = summarize_tracking(track_files[0])
                track_rows.append([
                    pretty_robot_name(robot),
                    pretty_controller_name(ctrl),
                    "Trajectory Tracking",
                    f"{t['mse']:.5f}",
                    f"{t['rtf']:.2f}x",
                    f"{t['max_input']:.3f}",
                    CONTROL_LIMITS.get(robot, "")
                ])

    set_df = pd.DataFrame(set_rows, columns=[
    "Robot", "Controller", "Experiment",
    "Avg Final Error ± std", "RTF", "Max Control Input", "Control Limit"
    ])



    track_df = pd.DataFrame(track_rows, columns=[
        "Robot", "Controller", "Experiment",
        "MSE", "RTF", "Max Input", "Control Limit"
    ])

    set_df.to_csv("set_report.csv", index=False)
    track_df.to_csv("tracking_report.csv", index=False)

    print("\nGenerated:")
    print("  set_report.csv")
    print("  tracking_report.csv")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    robots = load_all_results("results")

    # SET Lyapunov
    plot_set_clf_all_robots_one_plot(robots, controller_filter="id_clf_qp")

    # TRACKING Lyapunov
    plot_tracking_clf_all_robots_one_plot(robots, controller_filter="id_clf_qp", xlim=(0.0, 4.0))

    # Trajectory circles
    # plot_tracking_xy_circles(robots, plane="xz", start_time=8.0)
    plot_tracking_xy_circles_combined(
    robots,
    robot_list=["tendon", "helix"],   # order defines (a) and (b)
    plane="xz",
    start_time=8.0
)


    # Reports
    generate_report("results")
