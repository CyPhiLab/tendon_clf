import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
plt.rcParams.update({
    "font.size": 16,          # base font
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

# ============================================================
# Helper: Load logs from one robot
# ============================================================

def load_robot_logs(pattern):

    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError(f"No files found for pattern: {pattern}")

    time_list, V_list, error_list, u_list = [], [], [], []

    for f in files:
        print("Loading:", f)
        df = pd.read_csv(f)

        t     = df.iloc[:, 0].values
        V     = df.iloc[:, 1].values
        error = df.iloc[:, 2].values
        u     = df.iloc[:, 3:].values

        time_list.append(t)
        V_list.append(V)
        error_list.append(error)
        u_list.append(u)

    return time_list, V_list, error_list, u_list


# ============================================================
# Helper: Interpolate + Mean/Std
# ============================================================

def interpolate_and_average(time_list, V_list, error_list, u_list):

    t_min = max(t[0] for t in time_list)
    t_max = min(t[-1] for t in time_list)

    N = max(len(t) for t in time_list)
    t_grid = np.linspace(t_min, t_max, N)

    V_interp, err_interp, u_interp = [], [], []

    for t, V, err, u in zip(time_list, V_list, error_list, u_list):

        # Normalize V by initial value
        V_norm = V / V[0]

        V_interp.append(np.interp(t_grid, t, V_norm))
        err_interp.append(np.interp(t_grid, t, err))

        u_i = np.zeros((len(t_grid), u.shape[1]))
        for j in range(u.shape[1]):
            u_i[:, j] = np.interp(t_grid, t, u[:, j])

        u_interp.append(u_i)

    V_interp   = np.array(V_interp)
    err_interp = np.array(err_interp)
    u_interp   = np.array(u_interp)

    V_mean, V_std     = np.mean(V_interp, axis=0), np.std(V_interp, axis=0)
    err_mean, err_std = np.mean(err_interp, axis=0), np.std(err_interp, axis=0)

    u_mean = np.mean(u_interp, axis=0)
    u_std  = np.std(u_interp, axis=0)

    return t_grid, V_mean, V_std, err_mean, err_std, u_mean, u_std


# ============================================================
# Load ALL robots
# ============================================================

helix_time, helix_V, helix_err, helix_u = load_robot_logs(
    "helix_id_clf_qp_pos*.csv"
)

tendon_time, tendon_V, tendon_err, tendon_u = load_robot_logs(
    "tendon_id_clf_qp_pos*.csv"
)

spirob_time, spirob_V, spirob_err, spirob_u = load_robot_logs(
    "spirob_id_clf_qp_pos*.csv"
)

# helix_time, helix_V, helix_err, helix_u = load_robot_logs(
#     "helix_id_clf_qp_tracking.csv"
# )

# tendon_time, tendon_V, tendon_err, tendon_u = load_robot_logs(
#     "tendon_id_clf_qp_tracking.csv"
# )

# spirob_time, spirob_V, spirob_err, spirob_u = load_robot_logs(
#     "spirob_id_clf_qp_tracking.csv"
# )

# ============================================================
# Process ALL robots
# ============================================================

tA, VA_mean, VA_std, eA_mean, eA_std, uA_mean, uA_std = interpolate_and_average(
    helix_time, helix_V, helix_err, helix_u
)

tB, VB_mean, VB_std, eB_mean, eB_std, uB_mean, uB_std = interpolate_and_average(
    tendon_time, tendon_V, tendon_err, tendon_u
)

tC, VC_mean, VC_std, eC_mean, eC_std, uC_mean, uC_std = interpolate_and_average(
    spirob_time, spirob_V, spirob_err, spirob_u
)


# # ============================================================
# # Plot 1: Task Error Comparison
# # ============================================================

# plt.figure(figsize=(8,5))

# plt.plot(tA, eA_mean, linewidth=2, label="Helix Error")
# plt.fill_between(tA, eA_mean - eA_std, eA_mean + eA_std, alpha=0.2)

# plt.plot(tB, eB_mean, linewidth=2, label="Tendon Error")
# plt.fill_between(tB, eB_mean - eB_std, eB_mean + eB_std, alpha=0.2)

# plt.plot(tC, eC_mean, linewidth=2, label="Spirob Error")
# plt.fill_between(tC, eC_mean - eC_std, eC_mean + eC_std, alpha=0.2)

# plt.xlabel("Time (s)")
# plt.ylabel("Task Error (m)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# ============================================================
# Plot 2: Normalized CLF Comparison V/V(0)
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(tA, VA_mean, linewidth=4, label="Helix")
plt.fill_between(tA, VA_mean - VA_std, VA_mean + VA_std, alpha=0.3)

plt.plot(tB, VB_mean, linewidth=4, label="Tendon")
plt.fill_between(tB, VB_mean - VB_std, VB_mean + VB_std, alpha=0.3)

plt.plot(tC, VC_mean, linewidth=4, label="Spirob")
plt.fill_between(tC, VC_mean - VC_std, VC_mean + VC_std, alpha=0.3)

plt.xlabel("Time (s)")
plt.ylabel("Normalized Lyapunov Function ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# # ============================================================
# # Plot 3: Mean Control Effort Comparison
# # ============================================================

# uA_scalar_mean = np.mean(uA_mean, axis=1)
# uB_scalar_mean = np.mean(uB_mean, axis=1)
# uC_scalar_mean = np.mean(uC_mean, axis=1)

# uA_scalar_std  = np.mean(uA_std, axis=1)
# uB_scalar_std  = np.mean(uB_std, axis=1)
# uC_scalar_std  = np.mean(uC_std, axis=1)

# plt.figure(figsize=(8,5))

# plt.plot(tA, uA_scalar_mean, linewidth=2, label="Helix Control")
# plt.fill_between(tA,
#                  uA_scalar_mean - uA_scalar_std,
#                  uA_scalar_mean + uA_scalar_std,
#                  alpha=0.2)

# plt.plot(tB, uB_scalar_mean, linewidth=2, linestyle="--", label="Tendon Control")
# plt.fill_between(tB,
#                  uB_scalar_mean - uB_scalar_std,
#                  uB_scalar_mean + uB_scalar_std,
#                  alpha=0.2)

# plt.plot(tC, uC_scalar_mean, linewidth=2, linestyle=":", label="Spirob Control")
# plt.fill_between(tC,
#                  uC_scalar_mean - uC_scalar_std,
#                  uC_scalar_mean + uC_scalar_std,
#                  alpha=0.2)

# plt.xlabel("Time (s)")
# plt.ylabel("Mean Control Input")
# plt.title("Mean Control Effort Comparison")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# -------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # ============================================================
# # User: Put all your CSV files in the same folder as this script
# # ============================================================

# controller_files = {
#     "MPC": "helix_mpc_tracking.csv",
#     "OSC": "helix_osc_tracking.csv",
#     "Impedance": "helix_impedance_tracking.csv",
#     "ID-CLF-QP": "helix_id_clf_qp_tracking.csv"
# }

# # ============================================================
# # 1) Load all controller logs
# # ============================================================

# data = {}

# print("\nLoading controller CSV logs...\n")

# for name, filename in controller_files.items():

#     if not os.path.exists(filename):
#         print(f"❌ File not found: {filename}")
#         continue

#     df = pd.read_csv(filename)
#     data[name] = df

#     print(f"✅ Loaded: {filename} ({len(df)} rows)")

# # ============================================================
# # 2) Plot Task Error Over Time
# # ============================================================

# plt.figure(figsize=(9,6))

# for name, df in data.items():
#     plt.plot(df["time"], df["task_error"], label=name)

# plt.xlabel("Time (s)")
# plt.ylabel("Task Error (m)")
# # plt.title("Task-Space Error Over Time (4 Controllers)")
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ============================================================
# # 3) Create Summary Table Report
# # ============================================================

# rows = []

# for name, df in data.items():

#     # --- Average task error ---
#     avg_error = df["task_error"].mean()

#     # --- Extract control input columns u0,u1,... ---
#     u_cols = [c for c in df.columns if c.startswith("u")]

#     # --- Max/Min input across all actuators ---
#     max_input = df[u_cols].max().max()
#     min_input = df[u_cols].min().min()

#     rows.append([name, avg_error, max_input, min_input])

# # Build table
# summary = pd.DataFrame(
#     rows,
#     columns=["Control Scheme", "Avg Error Over Time (m)", "Max Input", "Min Input"]
# )

# print("\n===================================================")
# print(" Controller Comparison Report")
# print("===================================================\n")
# print(summary.to_string(index=False))

# # Optional: Save table as CSV
# summary.to_csv("controller_report.csv", index=False)
# print("\n✅ Report saved as: controller_report.csv\n")

# # ----------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # ============================================================
# # CSV files for the 4 controllers
# # ============================================================

# controller_files = {
#     "MPC": "helix_mpc_tracking.csv",
#     "OSC": "helix_osc_tracking.csv",
#     "Impedance": "helix_impedance_tracking.csv",
#     "ID-CLF-QP": "helix_id_clf_qp_tracking.csv"
# }

# # ============================================================
# # Plot Control Inputs (Mean ± Std)
# # ============================================================

# plt.figure(figsize=(9,6))

# for name, filename in controller_files.items():

#     if not os.path.exists(filename):
#         print(f"❌ File not found: {filename}")
#         continue

#     df = pd.read_csv(filename)

#     # ---- time ----
#     t = df["time"].values

#     # ---- control columns u0,u1,... ----
#     u_cols = [c for c in df.columns if c.startswith("u")]
#     U = df[u_cols].values   # shape = (T, nu)

#     # ---- mean + std across actuators ----
#     u_mean = np.mean(U, axis=1)
#     u_std  = np.std(U, axis=1)

#     # ---- plot mean curve ----
#     plt.plot(t, u_mean, label=f"{name} Input")

#     # ---- shaded band ----
#     plt.fill_between(
#         t,
#         u_mean - u_std,
#         u_mean + u_std,
#         alpha=0.2
#     )

# # ============================================================
# # Plot formatting
# # ============================================================

# plt.xlabel("Time (s)")
# plt.ylabel("Control Input (mean ± std)")
# # plt.title("Control Inputs Over Time (4 Controllers)")
# plt.legend()
# plt.tight_layout()
# plt.show()
