import os
import glob
import ast
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from robot import Robot


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
})


def naming(name, mapping):
    if name is None:
        return ""
    n = name.lower().strip()
    if n in mapping:
        return mapping[n]
    return "-".join(word.capitalize() for word in n.split("_"))


control_name = {
    "clf_qp": "CLF-QP",
    "id_clf_qp": "ID-CLF-QP",
    "impedance": "IC",
    "uosc": "UIC",
    "osc": "EOSC",
    "impedance_qp": "IC-QP"
}

robot_name = {
    "tendon": "Finger",
    "helix": "Helix",
    "spirob": "SpiRob"
}

def legend_above(ax, ncol=None):
    handles, labels = ax.get_legend_handles_labels()
    ordered = sorted(zip(handles, labels), key=lambda hl: (hl[1] == "Reference", hl[1]))
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
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)


def parse_vec_series(series):
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


def read_control_time(path):
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "control_time" in line:
                    return float(line.strip().split(",")[1])
    except Exception:
        pass
    return float("nan")


def pick_tracking_file(files: List[str], omega_tag: str = "omg2") -> Optional[str]:
    if len(files) == 0:
        return None
    for f in files:
        if omega_tag in os.path.basename(f).lower():
            return f
    return files[0]


def aggregate_experiment(files: List[str]) -> Dict[str, Any]:
    time_list: List[np.ndarray] = []
    err_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []

    has_V_all = True

    for f in files:
        t, sim_time, V, e, _, _, _ = load_single_csv(f)
        time_axis = sim_time if sim_time is not None else t

        time_list.append(time_axis)
        err_list.append(e if e is not None else np.zeros_like(time_axis))

        if V is None:
            has_V_all = False
        else:
            V_list.append(V)

    t_min = max(t[0] for t in time_list)
    t_max = min(t[-1] for t in time_list)
    N = max(len(t) for t in time_list)
    t_grid = np.linspace(t_min, t_max, N)

    err_interp = [np.interp(t_grid, t, e) for t, e in zip(time_list, err_list)]

    result: Dict[str, Any] = {
        "time": t_grid,
        "err_mean": np.mean(err_interp, axis=0),
        "err_std": np.std(err_interp, axis=0),
    }

    if has_V_all and len(V_list) == len(files):
        V_interp = []
        for t, V in zip(time_list, V_list):
            V0 = V[0] if (len(V) > 0 and V[0] != 0) else 1.0
            V_interp.append(np.interp(t_grid, t, V / V0))
        result["V_mean"] = np.mean(V_interp, axis=0)
        result["V_std"] = np.std(V_interp, axis=0)

    return result


def tracking_trajectory_single(file: str) -> Dict[str, Any]:
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


def load_results(root: str, traj_omega_tag: str = "omg2") -> Dict[str, Dict[str, Dict[str, Any]]]:
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
                robots[robot][ctrl]["set"] = aggregate_experiment(set_files)

            if len(track_files) > 0:
                robots[robot][ctrl]["tracking"] = aggregate_experiment(track_files)

                chosen = pick_tracking_file(track_files, omega_tag=traj_omega_tag)
                if chosen is not None:
                    tr = tracking_trajectory_single(chosen)
                    if "x" in tr and "xd" in tr:
                        robots[robot][ctrl]["tracking_traj"] = tr

    return robots


def clf_plot(robots, control, experiment):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    items = []

    for robot, controllers in robots.items():
        for ctrl, data in controllers.items():
            if experiment not in data:
                continue
            if control is not None and ctrl != control:
                continue
            exp_data = data[experiment]
            if "V_mean" not in exp_data:
                continue
            items.append((robot, ctrl, exp_data))

    robot_order = ["tendon", "helix", "spirob"]
    items.sort(key=lambda x: robot_order.index(x[0]) if x[0] in robot_order else 999)

    for robot, ctrl, exp_data in items:
        t = exp_data["time"]
        V = exp_data["V_mean"]
        S = exp_data["V_std"]

        line, = ax.plot(t, V, linewidth=4, label=naming(robot, robot_name), zorder=3)
        ax.fill_between(t, V - S, V + S, color=line.get_color(), alpha=0.25, linewidth=0, zorder=2)

        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Lyapunov Function")
    ax.grid(True)

    if experiment == "set":
        plt.xlim(0, 0.8)
    else:
        plt.xlim(0, 4)

    legend_above(ax, ncol=None)
    finalize_figure(fig, ax)
    plt.show()


def plot_tracking_trajectory(robots, robot_list, plane, start_time):
    plane = plane.lower()
    idx = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    i, j = idx[plane]

    ctrl_order = ["clf_qp", "id_clf_qp", "impedance", "uosc", "osc", "impedance_QP"]

    controller_colors = {
        "id_clf_qp": "#1f77b4",
        "impedance": "#ff7f0e",
        "osc": "#d62728",
        "impedance_QP": "#2ca02c",
        "clf_qp": "#9467bd",
        "uosc": "#8c564b"
    }

    fig, axes = plt.subplots(
        nrows=len(robot_list),
        ncols=1,
        figsize=(8, 6 * len(robot_list))
    )

    if len(robot_list) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for ax, robot in zip(axes, robot_list):
        if robot not in robots:
            continue

        active_ctrls = [
            c for c in ctrl_order
            if c in robots[robot] and "tracking_traj" in robots[robot][c]
        ]

        n = len(active_ctrls)
        if n == 0:
            continue

        x_all, y_all = [], []

        for ctrl in active_ctrls:
            tr = robots[robot][ctrl]["tracking_traj"]
            mask = tr["time"] >= start_time

            x = tr["x"][mask]
            xd = tr["xd"][mask]

            x_all.extend([x[:, i], xd[:, i]])
            y_all.extend([x[:, j], xd[:, j]])

        x_min, x_max = min(np.min(v) for v in x_all), max(np.max(v) for v in x_all)
        y_min, y_max = min(np.min(v) for v in y_all), max(np.max(v) for v in y_all)

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min)

        base_scale = span * 1.25

        if robot == "tendon":
            base_scale *= 0.95
            horizontal_spacing = base_scale * 0.75
            vertical_spacing = base_scale * 0.4
        elif robot == "helix":
            base_scale *= 1.0
            horizontal_spacing = base_scale * 1.0
            vertical_spacing = base_scale * 0.45
        else:
            horizontal_spacing = base_scale * 1.0
            vertical_spacing = base_scale * 0.45

        if n >= 4:
            top_count = n // 2 + n % 2
            bottom_count = n - top_count
        else:
            top_count = n
            bottom_count = 0

        positions = {}

        for k in range(top_count):
            x_pos = (k - (top_count - 1) / 2) * horizontal_spacing
            y_pos = vertical_spacing
            positions[active_ctrls[k]] = (x_pos, y_pos)

        for k in range(bottom_count):
            x_pos = (k - (bottom_count - 1) / 2) * horizontal_spacing
            y_pos = -vertical_spacing
            positions[active_ctrls[top_count + k]] = (x_pos, y_pos)

        for ctrl in active_ctrls:
            tr = robots[robot][ctrl]["tracking_traj"]
            mask = tr["time"] >= start_time

            x = tr["x"][mask]
            xd = tr["xd"][mask]

            dx, dy = positions[ctrl]

            x_norm = x[:, i] - cx
            y_norm = x[:, j] - cy
            xd_norm = xd[:, i] - cx
            yd_norm = xd[:, j] - cy

            color = controller_colors[ctrl]

            line, = ax.plot(
                x_norm + dx,
                y_norm + dy,
                color=color,
                linewidth=5.5
            )

            ax.plot(
                xd_norm + dx,
                yd_norm + dy,
                color="black",
                linestyle="--",
                linewidth=3
            )

            label = naming(ctrl, control_name)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.35)

        x_plot_all = []
        y_plot_all = []

        for ctrl in active_ctrls:
            tr = robots[robot][ctrl]["tracking_traj"]
            mask = tr["time"] >= start_time

            x = tr["x"][mask]
            xd = tr["xd"][mask]

            dx, dy = positions[ctrl]

            x_plot_all.extend([x[:, i] - cx + dx, xd[:, i] - cx + dx])
            y_plot_all.extend([x[:, j] - cy + dy, xd[:, j] - cy + dy])

        label_char = chr(ord('a') + robot_list.index(robot))
        ax.text(0.96, 0.15, f"({label_char})", transform=ax.transAxes, fontsize=20, ha="right", va="top")

        x_min = min(np.min(v) for v in x_plot_all)
        x_max = max(np.max(v) for v in x_plot_all)
        y_min = min(np.min(v) for v in y_plot_all)
        y_max = max(np.max(v) for v in y_plot_all)

        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)

        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False)

    ref_line = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=3)
    legend_handles.append(ref_line)
    legend_labels.append("Reference")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=4,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def final_error(df):
    return float(df["task_error"].iloc[-1])


def mse_error(df):
    e = df["task_error"].to_numpy()
    return float(np.mean(e ** 2))


def summarize_set(files):
    error = []
    ctrl_time = []

    for f in files:
        df = pd.read_csv(f, comment="#")
        error.append(final_error(df))

        control_time = read_control_time(f)
        if not np.isnan(control_time):
            ctrl_time.append(control_time)

    return {
        "final_mean": float(np.mean(error) * 1e2),
        "final_std": float(np.std(error) * 1e2),
        "control_time_mean": float(np.mean(ctrl_time)) if len(ctrl_time) > 0 else float("nan"),
        "control_time_std": float(np.std(ctrl_time)) if len(ctrl_time) > 0 else float("nan"),
    }


def summarize_tracking(files):
    mse = []
    ctrl_time = []

    for f in files:
        df = pd.read_csv(f, comment="#")
        mse.append(mse_error(df))

        control_time = read_control_time(f)
        if not np.isnan(control_time):
            ctrl_time.append(control_time)

    return {
        "mse_mean": float(np.mean(mse) * 1e4),
        "mse_std": float(np.std(mse) * 1e4),
        "control_time_mean": float(np.mean(ctrl_time)) if len(ctrl_time) > 0 else float("nan"),
        "control_time_std": float(np.std(ctrl_time)) if len(ctrl_time) > 0 else float("nan"),
    }

def get_robot_parameters(robot_name_dict):

    rows = []

    for robot_key in robot_name_dict.keys():
        for ctrl_key in control_name.keys():

            try:
                r = Robot(model_name=robot_key,
                          control_scheme=ctrl_key)

                # ---- w1 logic ----
                if ctrl_key in ["clf_qp", "impedance_qp", "id_clf_qp"]:
                    w1_value = 1
                else:
                    w1_value = "--"

                row = {
                    "Robot": naming(robot_key, robot_name),
                    "Controller": naming(ctrl_key, control_name),

                    "Kp": getattr(r, "Kp", "--"),
                    "epsilon": getattr(r, "e", "--"),
                    "w1": w1_value,
                    "w2": getattr(r, "reg_qdd", "--"),
                    "w3": getattr(r, "reg_u", "--"),
                    "w4": getattr(r, "reg_null", "--"),
                    "rho": getattr(r, "reg_dl", "--"),
                }

            except Exception:
                row = {
                    "Robot": naming(robot_key, robot_name),
                    "Controller": naming(ctrl_key, control_name),
                    "Kp": "--",
                    "epsilon": "--",
                    "w1": "--",
                    "w2": "--",
                    "w3": "--",
                    "w4": "--",
                    "rho": "--",
                }

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("controller_parameter_table.csv", index=False)

def generate_combined_report(root):
    rows = []
    robots = ["tendon", "helix", "spirob"]

    for robot in robots:
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
                final_error_str = f"{s['final_mean']:.2f} ± {s['final_std']:.2f}"
            else:
                final_error_str = "-"

            if len(track_files) > 0:
                t = summarize_tracking(track_files)
                mse_str = f"{t['mse_mean']:.2f} ± {t['mse_std']:.2f}"
            else:
                mse_str = "-"

            rows.append([
                naming(robot, robot_name),
                naming(ctrl, control_name),
                final_error_str,
                mse_str
            ])

    df = pd.DataFrame(rows, columns=[
        "Robot",
        "Controller",
        "Final Error (SP) (cm)",
        "TT–MSE ($cm^2$)"
    ])

    df.to_csv("combined_benchmark.csv", index=False)
    print("\nGenerated: combined_benchmark.csv")


def csv_to_latex_table(csv_file, output_tex="combined_table.tex"):
    df = pd.read_csv(csv_file)
    df = df.replace("±", r"$\pm$", regex=False)
    df = df.replace("-", "--")

    controllers = [
        "CLF-QP",
        "ID-CLF-QP",
        "IC",
        "UIC",
        "EOSC",
        "IC-QP"
    ]

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Combined benchmark comparison for set point (SP) and trajectory tracking (TT) across three robot platforms.}")
    latex.append(r"\label{tab:combined_benchmark}")
    latex.append(r"\resizebox{0.5\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{|c|c|c|c|}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Robot} & \textbf{Controller} & "
                 r"\textbf{Final Error (SP) (cm)} & "
                 r"\textbf{TT--MSE ($\mathbf{cm^2}$)} \\")
    latex.append(r"\hline")

    robots_unique = df["Robot"].unique()

    for robot in robots_unique:
        robot_df = df[df["Robot"] == robot]
        first_row = True

        for ctrl in controllers:
            row_data = robot_df[robot_df["Controller"] == ctrl]

            if not row_data.empty:
                row = row_data.iloc[0]
                final_error = row["Final Error (SP) (cm)"]
                tt_mse = row["TT–MSE ($cm^2$)"]
            else:
                final_error = "Failed Convergence"
                tt_mse = "Failed Convergence"

            if ctrl == "ID-CLF-QP":
                ctrl_str = rf"\textbf{{{ctrl}}}"
                final_error = rf"\textbf{{{final_error}}}"
                tt_mse = rf"\textbf{{{tt_mse}}}"
            else:
                ctrl_str = ctrl

            if robot.lower() == "spirob":
                tt_mse = "--"

            if first_row:
                latex.append(rf"\multirow{{6}}{{*}}{{{robot}}} & {ctrl_str} & {final_error} & {tt_mse} \\")
                first_row = False
            else:
                latex.append(rf"& {ctrl_str} & {final_error} & {tt_mse} \\")

        latex.append(r"\hline")

    latex.append(r"\end{tabular}}")
    latex.append(r"\end{table*}")

    with open(output_tex, "w") as f:
        f.write("\n".join(latex))

    print(f"LaTeX table saved to {output_tex}")


if __name__ == "__main__":
    robots = load_results("results", traj_omega_tag="omg2")

    clf_plot(robots, control="id_clf_qp", experiment="set")
    clf_plot(robots, control="id_clf_qp", experiment="tracking")

    plot_tracking_trajectory(
        robots,
        robot_list=["tendon", "helix"],
        plane="xz",
        start_time=8.0
    )

    generate_combined_report("results")
    csv_to_latex_table("combined_benchmark.csv", output_tex="combined_table.tex")

    get_robot_parameters(robot_name)