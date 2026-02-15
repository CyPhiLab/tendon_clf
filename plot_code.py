import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# -------- Paper style (sans serif) --------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def load_single_csv(file):

    df = pd.read_csv(file)

    # --- required columns ---
    t = df["time"].values

    # Lyapunov (some controllers might not have it)
    if "lyapunov_V" in df.columns:
        V = df["lyapunov_V"].values
    else:
        V = None

    # task error (always exists)
    error = df["task_error"].values

    # control inputs (any number of actuators)
    u_cols = [c for c in df.columns if c.startswith("u")]
    u = df[u_cols].values if len(u_cols) > 0 else None

    return t, V, error, u


def process_set_experiment(files):

    time_list, V_list, err_list = [], [], []
    has_V = True

    for f in files:
        print("SET:", f)
        t, V, e, _ = load_single_csv(f)

        time_list.append(t)
        err_list.append(e)

        if V is None:
            has_V = False
        else:
            V_list.append(V)

    # common time grid
    t_min = max(t[0] for t in time_list)
    t_max = min(t[-1] for t in time_list)
    N = max(len(t) for t in time_list)
    t_grid = np.linspace(t_min, t_max, N)

    # --- error always exists ---
    err_interp = [
        np.interp(t_grid, t, e)
        for t, e in zip(time_list, err_list)
    ]

    result = {
        "time": t_grid,
        "err_mean": np.mean(err_interp, axis=0),
        "err_std":  np.std(err_interp, axis=0),
    }

    # --- Lyapunov only if available ---
    if has_V:
        V_interp = [
            np.interp(t_grid, t, V/V[0])
            for t, V in zip(time_list, V_list)
        ]

        result["V_mean"] = np.mean(V_interp, axis=0)
        result["V_std"]  = np.std(V_interp, axis=0)

    return result

def process_tracking_experiment(file):

    print("TRACK:", file)
    t, V, e, _ = load_single_csv(file)

    result = {"time": t, "error": e}

    if V is not None:
        result["V"] = V / V[0]

    return result


def load_all_results(root="results"):

    robots = {}

    for robot in os.listdir(root):

        robot_path = os.path.join(root, robot)
        if not os.path.isdir(robot_path):
            continue

        robots[robot] = {}

        for ctrl in os.listdir(robot_path):

            ctrl_path = os.path.join(robot_path, ctrl)
            if not os.path.isdir(ctrl_path):
                continue

            set_files = sorted(glob.glob(os.path.join(ctrl_path, "set_*.csv")))
            track_files = glob.glob(os.path.join(ctrl_path, "tracking*.csv"))

            if len(set_files) == 0:
                continue

            robots[robot][ctrl] = {}

            # SET
            robots[robot][ctrl]["set"] = process_set_experiment(set_files)

            # TRACKING
            if len(track_files) > 0:
                robots[robot][ctrl]["tracking"] = process_tracking_experiment(track_files[0])

    return robots

def plot_set_clf(robots):

    for robot, controllers in robots.items():

        plt.figure(figsize=(8,5))
        plotted_any = False

        for ctrl, data in controllers.items():

            # ---- skip controllers without Lyapunov ----
            if "V_mean" not in data["set"]:
                continue

            t = data["set"]["time"]
            V = data["set"]["V_mean"]
            S = data["set"]["V_std"]

            plt.plot(t, V, linewidth=3, label=ctrl)
            plt.fill_between(t, V-S, V+S, alpha=0.25)

            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.title(f"{robot.upper()} — Lyapunov Convergence")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Lyapunov Function")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_tracking_error(robots):

    for robot, controllers in robots.items():

        plt.figure(figsize=(8,5))

        for ctrl, data in controllers.items():

            if "tracking" not in data:
                continue

            t = data["tracking"]["time"]
            e = data["tracking"]["error"]

            plt.plot(t, e, linewidth=3, label=ctrl)

        plt.title(f"{robot.upper()} — Tracking Experiment")
        plt.xlabel("Time (s)")
        plt.ylabel("Task Error (m)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def compute_rtf(df):
    return df["sim_time"].iloc[-1] / df["time"].iloc[-1]

def max_control_input(df):
    u_cols = [c for c in df.columns if c.startswith("u")]
    if len(u_cols) == 0:
        return np.nan
    return np.abs(df[u_cols].values).max()


def final_error(df):
    return df["task_error"].iloc[-1]


def mse_error(df):
    e = df["task_error"].values
    return np.mean(e**2)

def summarize_set(files):

    finals = []
    max_inputs = []

    for f in files:
        df = pd.read_csv(f)

        finals.append(final_error(df))
        max_inputs.append(max_control_input(df))

    return {
        "final_mean": np.mean(finals),
        "final_std":  np.std(finals),
        "max_input":  np.max(max_inputs)
    }

def summarize_tracking(file):

    df = pd.read_csv(file)

    return {
        "mse": mse_error(df),
        "rtf": compute_rtf(df),
        "max_input": max_control_input(df)
    }

CONTROL_LIMITS = {
    "tendon": "-1 ≤ u ≤ 1",
    "helix": "-25 ≤ u ≤ 25",
    "spirob": "u ≤ 0"
}

def generate_report(root="results"):

    set_rows = []
    track_rows = []

    for robot in os.listdir(root):

        robot_path = os.path.join(root, robot)
        if not os.path.isdir(robot_path):
            continue

        for ctrl in os.listdir(robot_path):

            ctrl_path = os.path.join(robot_path, ctrl)
            if not os.path.isdir(ctrl_path):
                continue

            set_files = sorted(glob.glob(os.path.join(ctrl_path, "set_*.csv")))
            track_files = glob.glob(os.path.join(ctrl_path, "tracking*.csv"))

            # ---------- SET ----------
            if len(set_files) > 0:
                s = summarize_set(set_files)

                set_rows.append([
                    robot.capitalize(),
                    ctrl.upper(),
                    "Static Point Tracking",
                    f"{s['final_mean']:.4f} ± {s['final_std']:.4f}",
                    f"{s['max_input']:.3f}",
                    CONTROL_LIMITS.get(robot, "")
                ])

            # ---------- TRACK ----------
            if len(track_files) > 0:
                t = summarize_tracking(track_files[0])

                track_rows.append([
                    robot.capitalize(),
                    ctrl.upper(),
                    "Trajectory Tracking",
                    f"{t['mse']:.5f}",
                    f"{t['rtf']:.2f}x",
                    f"{t['max_input']:.3f}",
                    CONTROL_LIMITS.get(robot, "")
                ])

    set_df = pd.DataFrame(set_rows, columns=[
        "Robot","Controller","Experiment",
        "Avg Final Error ± std","Max Control Input","Control Limit"
    ])

    track_df = pd.DataFrame(track_rows, columns=[
        "Robot","Controller","Experiment",
        "MSE","RTF","Max Input","Control Limit"
    ])

    set_df.to_csv("set_report.csv", index=False)
    track_df.to_csv("tracking_report.csv", index=False)

    print("\nGenerated:")
    print("  set_report.csv")
    print("  tracking_report.csv")

if __name__ == "__main__":

    robots = load_all_results("results")

    plot_set_clf(robots)
    plot_tracking_error(robots)
    generate_report("results")
