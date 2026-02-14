# import pandas as pd
# import numpy as np
# from pathlib import Path

# # -----------------------------
# # Configuration
# # -----------------------------
# csv_files = [
#     "helix_pd_qp_pos1.csv",
#     "helix_pd_qp_pos2.csv",
#     "helix_pd_qp_pos3.csv",
#     "helix_pd_qp_pos4.csv",
# ]

# ERROR_THRESHOLD = 1e-2   # convergence definition

# # -----------------------------
# # Storage
# # -----------------------------
# final_errors = []
# converge_times = []
# max_controls = []
# min_controls = []

# # -----------------------------
# # Processing
# # -----------------------------
# for file in csv_files:
#     df = pd.read_csv(file)

#     # ---- Final error
#     final_error = df["task_error"].iloc[-1]
#     final_errors.append(final_error)

#     # ---- Control extrema (NOT averaged)
#     u_cols = [c for c in df.columns if c.startswith("u")]
#     max_controls.append(df[u_cols].to_numpy().max())
#     min_controls.append(df[u_cols].to_numpy().min())

#     # ---- Convergence time
#     converged = df[df["task_error"] <= ERROR_THRESHOLD]
#     if len(converged) > 0:
#         converge_times.append(converged["time"].iloc[0])
#     else:
#         converge_times.append(np.nan)  # did not converge

# # -----------------------------
# # Aggregate statistics
# # -----------------------------
# results = {
#     "average_final_error": np.mean(final_errors),
#     "average_converging_time": np.nanmean(converge_times),
#     "max_control_overall": np.max(max_controls),
#     "min_control_overall": np.min(min_controls),
# }

# # -----------------------------
# # Print nicely
# # -----------------------------
# print("\n===== CLF-QP Trajectory Statistics =====")
# print(f"Average final error        : {results['average_final_error']:.6e}")
# print(f"Average converging time [s]: {results['average_converging_time']:.4f}")
# print(f"Max control input (global) : {results['max_control_overall']:.3f}")
# print(f"Min control input (global) : {results['min_control_overall']:.3f}")

# # -----------------------------
# # Optional: save summary CSV
# # -----------------------------
# summary_df = pd.DataFrame({
#     "file": csv_files,
#     "final_error": final_errors,
#     "converging_time": converge_times,
#     "max_control": max_controls,
#     "min_control": min_controls,
# })

# summary_df.to_csv("helix_clf_qp_summary.csv", index=False)
# print("\nSaved per-run summary to helix_clf_qp_summary.csv")

# --------------------------------------------------------------

# import pandas as pd
# import numpy as np

# # -----------------------------
# # Load CSV
# # -----------------------------
# df = pd.read_csv("helix_osc_tracking.csv")

# # Drop t = 0 row (no control applied yet)
# df = df.iloc[1:].reset_index(drop=True)

# # -----------------------------
# # Average task-space error
# # -----------------------------
# avg_error = df["task_error"].mean()

# # -----------------------------
# # Control extrema
# # -----------------------------
# u_cols = [c for c in df.columns if c.startswith("u")]
# u_values = df[u_cols].to_numpy()

# u_max = u_values.max()
# u_min = u_values.min()

# # -----------------------------
# # Print results
# # -----------------------------
# print("===== Tracking Statistics =====")
# print(f"Average error over time : {avg_error:.6e}")
# print(f"Max control input       : {u_max:.3f}")
# print(f"Min control input       : {u_min:.3f}")



# ------------------------------------------
import pandas as pd
import numpy as np
import glob
import os

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./"   # change if needed

robots = ["tendon", "helix", "spirob"]
controllers = {
    "id_clf_qp": "ID-CLF-QP",
    "impedance": "IC",
    "mpc": "MPC"
}

# ============================================================
# HELPERS
# ============================================================

def load_csv(file):
    return pd.read_csv(file)

def final_error(df):
    return df["task_error"].iloc[-1]

def avg_error(df):
    return df["task_error"].mean()

def max_input(df):
    u_cols = [c for c in df.columns if c.startswith("u")]
    return np.abs(df[u_cols]).values.max()

def real_time_factor(df, wall_time=None):
    sim_time = df["time"].iloc[-1]
    if wall_time is None:
        return None
    return sim_time / wall_time

# ============================================================
# TABLE 1 — STATIC POINT TRACKING
# ============================================================

print("\n================ STATIC POINT TRACKING =================\n")

for robot in robots:
    for ctrl_key, ctrl_name in controllers.items():

        pattern = os.path.join(DATA_DIR, f"{robot}_{ctrl_key}_pos*.csv")
        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            continue

        final_errors = []
        max_inputs = []

        for f in files:
            df = load_csv(f)
            final_errors.append(final_error(df))
            max_inputs.append(max_input(df))

        mean_err = np.mean(final_errors)
        std_err  = np.std(final_errors)
        max_u    = np.max(max_inputs)

        print(f"{robot.capitalize():8s} | {ctrl_name:10s} | "
              f"Final Error: {mean_err:.4f} ± {std_err:.4f} | "
              f"Max Input: {max_u:.3f}")

# ============================================================
# TABLE 2 — TRAJECTORY TRACKING
# ============================================================

print("\n================ TRAJECTORY TRACKING =================\n")

for robot in robots:
    for ctrl_key, ctrl_name in controllers.items():

        file = os.path.join(DATA_DIR, f"{robot}_{ctrl_key}_tracking.csv")

        if not os.path.exists(file):
            continue

        df = load_csv(file)

        avg_err = avg_error(df)
        max_u   = max_input(df)

        sim_time = df["time"].iloc[-1]
        # If you know wall clock time, insert it manually here:
        wall_time = None
        rtf = real_time_factor(df, wall_time)

        print(f"{robot.capitalize():8s} | {ctrl_name:10s} | "
              f"Avg Error: {avg_err:.4f} | "
              f"Max Input: {max_u:.3f} | "
              f"Sim Time: {sim_time:.2f}s | "
              f"RTF: {rtf if rtf is not None else 'N/A'}")
