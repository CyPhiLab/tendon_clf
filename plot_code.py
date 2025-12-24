# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# file_path = r"D:\Control\Soft Arm Project\Ablation Study\Full.csv"
# df = pd.read_csv(file_path)

# # Time and core metrics
# time = df["time"].values
# V = df["V"].values
# V_dot = df["V_dot"].values
# V_dot = np.array([float(str(v).strip("[]")) for v in V_dot])

# stability_bound = df["stability_bound"].values
# task_error = df["task_error"].values

# # Control inputs
# u_0 = df["u_0 (A Flexor)"].values
# u_1 = df["u_1 (A Extensor)"].values
# u_2 = df["u_2 (B Flexor)"].values
# u_3 = df["u_3 (B Extensor)"].values

# # Joint torques
# tau_0 = df["tau_0"].values
# tau_1 = df["tau_1"].values
# tau_2 = df["tau_2"].values
# tau_3 = df["tau_3"].values

# # --- Lyapunov Function ---
# plt.figure()
# plt.plot(time, V, label="Lyapunov Function V")
# plt.xlabel("Time (s)")
# plt.ylabel("V")
# plt.title("Lyapunov Function Over Time")
# plt.grid(True)
# plt.legend()

# # --- V_dot vs Stability Bound ---
# plt.figure()
# plt.plot(time, V_dot, label="V_dot")
# plt.plot(time, stability_bound, label="−2/e·V + dl")
# plt.xlabel("Time (s)")
# plt.ylabel("V_dot")
# plt.title("V_dot vs Stability Bound")
# plt.grid(True)
# plt.legend()

# # --- Task-Space Error ---
# plt.figure()
# plt.plot(time, task_error, label="‖x_des - x‖")
# plt.xlabel("Time (s)")
# plt.ylabel("Task-Space Error (m)")
# plt.title("End-Effector Task Error")
# plt.grid(True)
# plt.legend()

# # --- Control Inputs u ---
# fig_u, axs_u = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
# u_labels = ["u_0 (A Flexor)", "u_1 (A Extensor)", "u_2 (B Flexor)", "u_3 (B Extensor)"]
# u_data = [u_0, u_1, u_2, u_3]

# for i in range(4):
#     axs_u[i].plot(time, u_data[i])
#     axs_u[i].set_ylabel(u_labels[i])
#     axs_u[i].grid(True)
# axs_u[-1].set_xlabel("Time (s)")
# fig_u.suptitle("Control Inputs Over Time")
# fig_u.tight_layout(rect=[0, 0.03, 1, 0.95])

# # --- Joint Torques τ ---
# fig_tau, axs_tau = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
# tau_labels = ["tau_0", "tau_1", "tau_2", "tau_3"]
# tau_data = [tau_0, tau_1, tau_2, tau_3]

# for i in range(4):
#     axs_tau[i].plot(time, tau_data[i])
#     axs_tau[i].set_ylabel(tau_labels[i])
#     axs_tau[i].grid(True)
# axs_tau[-1].set_xlabel("Time (s)")
# fig_tau.suptitle("Joint Torques Over Time")
# fig_tau.tight_layout(rect=[0, 0.03, 1, 0.95])

# # Show all plots
# plt.show()

# # Folder containing all CSVs
# folder = r"D:\Control\Soft Arm Project\Ablation Study"

# # Create plot
# plt.figure(figsize=(10, 6))

# for file in os.listdir(folder):
#     if file.endswith(".csv"):
#         path = os.path.join(folder, file)
#         df = pd.read_csv(path)

#         # Clean up 'V' column in case it's stored as "[value]"
#         df["V"] = df["V"].apply(lambda x: float(str(x).strip("[]")))
#         time = df["time"]
#         V = df["V"]

#         label = file.replace(".csv", "")
#         plt.plot(time, V, label=label)

# plt.xlabel("Time (s)")
# plt.ylabel("Lyapunov Function V")
# plt.title("Comparison of Lyapunov V Across Ablation Conditions")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob

# def load_and_interpolate(files_pattern, n_points=1000):
#     files = sorted(glob.glob(files_pattern))
#     dfs = [pd.read_csv(f) for f in files]

#     # Use full available range for this dataset
#     t_min = max(df['time'].min() for df in dfs)
#     t_max = min(df['time'].max() for df in dfs)

#     time_grid = np.linspace(t_min, t_max, n_points)

#     # Interpolate task_error
#     errors = []
#     for df in dfs:
#         errors.append(np.interp(time_grid, df['time'], df['task_error']))
#     errors = np.array(errors)

#     mean_error = np.mean(errors, axis=0)
#     std_error = np.std(errors, axis=0)

#     return time_grid, mean_error, std_error

# # Load both datasets separately
# time_id, id_mean, id_std = load_and_interpolate("simulation_results_clf_*.csv")
# time_imp, imp_mean, imp_std = load_and_interpolate("simulation_results_imp_*.csv")

# # Plot
# plt.figure(figsize=(10,6))

# # ID-CLF-QP
# plt.plot(time_id, id_mean, color="darkblue", label="ID-CLF-QP",linewidth=3)
# plt.fill_between(time_id, id_mean-id_std, id_mean+id_std,
#                  color="lightblue", alpha=0.4)

# # IMP-QP
# plt.plot(time_imp, imp_mean, color="darkorange", label="Impedance Control",linewidth=3)
# plt.fill_between(time_imp, imp_mean-imp_std, imp_mean+imp_std,
#                  color="navajowhite", alpha=0.5)

# plt.xlabel("Time (s)", fontsize=18)
# plt.ylabel("Task Error (m)", fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=16)
# # plt.title("Task Error vs Time (ID-CLF-QP vs Impedance Control)")
# plt.legend(fontsize=14)
# plt.grid(True)
# plt.show()


# # Generate a Lyapunov vs Time plot (mean ± std) for simulation_results_clf_*.csv
# import glob
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Find files
# files = sorted(glob.glob("simulation_results_clf_*.csv"))
# assert len(files) > 0, "No files matching simulation_results_clf_*.csv were found."

# # Load dataframes
# dfs = [pd.read_csv(f) for f in files]

# # Determine overlapping time range across all runs
# t_min = max(df['time'].min() for df in dfs)
# t_max = min(df['time'].max() for df in dfs)

# # Build common grid
# time_grid = np.linspace(t_min, t_max, 1000)

# # Interpolate Lyapunov onto the grid
# lyap_mat = []
# for df in dfs:
#     lyap_mat.append(np.interp(time_grid, df['time'], df['Lyapunov']))
# lyap_mat = np.array(lyap_mat)

# # Mean and std
# lyap_mean = lyap_mat.mean(axis=0)
# lyap_std = lyap_mat.std(axis=0)

# # Plot
# plt.figure(figsize=(9,6))
# plt.plot(time_grid, lyap_mean, linewidth=3)
# plt.fill_between(time_grid, lyap_mean - lyap_std, lyap_mean + lyap_std, alpha=0.35, label="±1 std")
# plt.xlabel("Time (s)", fontsize=18)
# plt.ylabel("Lyapunov Function", fontsize=18)
# plt.tick_params(axis='both', which='major', labelsize=16)
# # plt.title("Lyapunov vs Time (simulation_results_clf_*, full overlapping range)")
# # plt.legend(fontsize=14)
# plt.grid(True)

# plt.show()






# # ---------------Trajectory Visualization--------------------
# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# c = np.array([0, 0.00, 0.35])
# Ax, Ay, Az = 0.1, 0.1, 0.05
# w = 5

# # Time
# t = np.linspace(0, 2, 500)

# # Trajectory
# wx = np.pi * 0.5
# wy = np.pi * 2
# x = c[0] + Ax * np.cos(wx * t)
# y = c[1] + Ay * np.sin(wy * t)
# z = np.full_like(t, c[2])  # make z an array instead of scalar!

# # Plot 3D trajectory
# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z, label="Trajectory", color="darkblue", linewidth=3)
# ax.scatter(c[0], c[1], 0.7, color="red", s=60, label="Robot Base $c$")

# ax.set_xlabel("X [m]")
# ax.set_ylabel("Y [m]")
# ax.set_zlabel("Z [m]")

# plt.tight_layout()
# plt.legend(fontsize=12)
# plt.show()

# # --------------------Target Trajectory----------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# # Create custom legend handles (only two items)
# import matplotlib.lines as mlines



# # Center of the circles (robot base) — aligned with the base in xz-plane
# c = np.array([0.0, 0.0, 0.7])    # (x, y, z)

# # Radii
# A_full = 0.435
# A_half = A_full / 2.0

# # Target point angles
# thetas = np.array([-np.pi/2, -np.pi/3, -np.pi/6, 0.0, np.pi/6])

# # Parametric circles (xz-plane; y=0)
# th = np.linspace(0, 2*np.pi, 500)
# x_full = c[0] + A_full*np.cos(th)
# z_full = c[2] + A_full*np.sin(th)
# x_half = c[0] + A_half*np.cos(th)
# z_half = c[2] + A_half*np.sin(th)

# # Target points on each circle
# x_full_pts = c[0] + A_full*np.cos(thetas)
# z_full_pts = c[2] + A_full*np.sin(thetas)
# x_half_pts = c[0] + A_half*np.cos(thetas)
# z_half_pts = c[2] + A_half*np.sin(thetas)

# # Plot
# plt.figure(figsize=(6.2, 6.2))

# # dashed base circles
# plt.plot(x_full, z_full, linestyle=(0,(6,4)), linewidth=1, color='black', label="Base circle (A=0.435 m)")
# plt.plot(x_half, z_half, linestyle=(0,(6,4)), linewidth=1, color='black', label="Base circle (A=0.2175 m)")

# # robot base (center)
# plt.scatter([c[0]], [c[2]], s=180, color='red', label="Robot base (center)")

# robot_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Robot base')
# target_legend = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=8, label='Target points')

# # target points (big blue dots)
# plt.scatter(x_full_pts, z_full_pts, s=90, color='tab:blue', label="Targets on A=0.435 m")
# plt.scatter(x_half_pts, z_half_pts, s=90, color='tab:blue',  label="Targets on A=0.2175 m")
# plt.legend(handles=[robot_legend, target_legend], loc='upper right')
# plt.gca().set_aspect('equal', 'box')
# plt.xlabel("x (m)", fontsize = 14)
# plt.ylabel("z (m)",fontsize=14)
# # plt.title("Target Points in x-plane")
# plt.tight_layout()
# plt.show()

# -------------------Robot Coordinates-----------------------------
# === Sketch-style 3D scene matching the drawing ===
# - Robot base at [0, 0, 0.7]
# - Axes with x (left), y (right), z (up)
# - Rays from base to P and P_d
# - Small point "dangling" above base, with a short arc connector

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Key points ---
origin = np.array([0.0, 0.0, 0.0])
base   = np.array([0.0, 0.0, 0.7])      # robot base
Pd      = np.array([0.2, 0.4, 0.2])      # current end-effector
P2      = np.array([0.2, 0.55, 0.4]) 
P     = np.array([0.2, 0.5, 0.6])   # desired end-effector

# --- Equal scaling helper ---
def set_axes_equal(ax):
    """Make 3D plot have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    for center, axis in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        axis([center - radius, center + radius])

# --- Plot setup ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# Arrows from origin to P and Pd
def draw_arrow(start, end, color, label=None):
    vec = end - start
    ax.quiver(start[0], start[1], start[2],
              vec[0], vec[1], vec[2],
              color=color, arrow_length_ratio=0.1, linewidth=2.5)
    if label:
        ax.text(*(end + np.array([0.02, 0.02, 0.0])), label, color=color, fontsize=11)
# --- Updated draw_arrow function with fontsize option ---
def draw_arrow(start, end, color, label=None, fontsize=12):
    vec = end - start
    ax.quiver(start[0], start[1], start[2],
              vec[0], vec[1], vec[2],
              color=color, arrow_length_ratio=0.1, linewidth=2.5)
    if label:
        ax.text(*(end + np.array([0.02, 0.02, 0.0])), label,
                color=color, fontsize=fontsize, fontweight='bold')
        
draw_arrow(origin, P, color='blue', label='$p$',fontsize = 16)
draw_arrow(origin, Pd, color='black', label='$p_d$',fontsize = 16)

# --- Curve from robot base to P (reverse curvature) ---
t = np.linspace(0, 1, 50)[:, None]  # (50,1) for broadcasting

curvature = -0.3  # <--- negative reverses the bend (positive bows upward)
control = (base + P) / 2 + np.array([0.0, 0.0, curvature])

curve = ((1 - t)**2) * base + (2 * (1 - t) * t) * control + (t**2) * P

ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
        color='orchid', linewidth=25, label='path from base to $p$')

t = np.linspace(0, 1, 50)[:, None]  # (50,1) for broadcasting

curvature = -0.3  # <--- negative reverses the bend (positive bows upward)
control = (base + P2) / 2 + np.array([0.0, 0.0, curvature])

curve = ((1 - t)**2) * base + (2 * (1 - t) * t) * control + (t**2) * P2

ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
        color='orchid', alpha = 0.7 , linewidth=25, label='path from base to $p$')

t = np.linspace(0, 1, 50)[:, None]  # (50,1) for broadcasting

curvature = -0.3  # <--- negative reverses the bend (positive bows upward)
control = (base + Pd) / 2 + np.array([0.0, 0.0, curvature])

curve = ((1 - t)**2) * base + (2 * (1 - t) * t) * control + (t**2) * Pd

ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
        color='orchid', alpha = 0.4 , linewidth=25, label='path from base to $p$')



# Points
ax.scatter(*base, color='red', s=50)
ax.text(*(base + np.array([-0.05, -0.2, 0.22])),  r'$p_{base}$ = [0,0,0.7]', color='red', fontsize = 16)

# Dashed line from robot base to origin
ax.plot([base[0], origin[0]], [base[1], origin[1]], [base[2], origin[2]],
        linestyle='--', color='gray', linewidth=2)

# --- Style ---
ax.set_xlabel("x (m)", fontsize = 12)
ax.set_ylabel("y (m)",fontsize = 12)
ax.set_zlabel("z (m)",fontsize = 12)

# Make origin appear centered
ax.set_xlim(-0.4, 0.6)
ax.set_ylim(-0.4, 0.6)
ax.set_zlim(0, 1)

# Equal scale and view
set_axes_equal(ax)
ax.view_init(elev=20, azim=-60)
# ax.legend()

plt.tight_layout()
plt.show()




# ----------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # === Paths (edit if needed) ===
# clf_path = Path("clf_tracking.csv")
# imp_path = Path("imp_tracking.csv")

# # --- Load ---
# clf = pd.read_csv(clf_path)
# imp = pd.read_csv(imp_path)

# if "time" not in clf.columns or "time" not in imp.columns:
#     raise ValueError("Both CSVs must contain a 'time' column.")

# # --- Plot settings ---
# U_COLS = [f"u{i}" for i in range(1, 10)]
# time_clf = clf["time"].to_numpy()
# time_imp = imp["time"].to_numpy()

# fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
# axes = axes.ravel()

# for i, ucol in enumerate(U_COLS):
#     ax = axes[i]
#     # plot CLF and IMP controls
#     ax.plot(time_clf, clf[ucol], label="CLF", linewidth=2)
#     ax.plot(time_imp, imp[ucol], label="IMP", color="red", linewidth=2, linestyle="--")

#     # control limits (label only once)
#     ax.axhline(20,  color="green", linewidth=3, label="Control Limit" if i == 0 else "")
#     ax.axhline(-20, color="green", linewidth=3)

#     ax.set_title(ucol.upper(), fontsize=12)
#     ax.grid(True, alpha=0.3)

#     # add padding for axis labels
#     ax.xaxis.labelpad = 12
#     ax.yaxis.labelpad = 12

# # common labels with extra padding
# fig.text(0.5, 0.02, "Time [s]", ha="center", fontsize=12)
# fig.text(0.02, 0.5, "Control Input u", va="center", rotation="vertical", fontsize=12)

# # legend (includes Control Limit)
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=12)

# plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
# plt.show()


# # 9-input control comparison: ID-CLF-QP vs Impedance
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from matplotlib.lines import Line2D

# # ---------- Config ----------
# clf_pattern = "tracking_clf_*.csv"
# # try both "imp" and "impm" patterns (use whichever matches files)
# imp_patterns = ["tracking_imp_*.csv", "tracking_impm_*.csv"]  # <-- list, not str
# CONTROL_LIMIT = 20.0                     # green bands at ± this value
# N_POINTS = 1200                          # interpolation points (smooth curves)
# LINEWIDTH = 2.5
# DASH = (0, (6, 4))                       # dashed style for Impedance

# # ---------- Helpers ----------
# def pick_existing_pattern(patterns):
#     for p in patterns:
#         if len(glob.glob(p)) > 0:
#             return p
#     raise FileNotFoundError("No impedance files found matching any of: " + ", ".join(patterns))

# def load_interp_mean_std(files_pattern, cols, n_points=N_POINTS):
#     files = sorted(glob.glob(files_pattern))
#     if not files:
#         raise FileNotFoundError(f"No files match pattern: {files_pattern}")
#     dfs = [pd.read_csv(f) for f in files]

#     # sanity: require time column
#     for i, df in enumerate(dfs):
#         if "time" not in df.columns:
#             raise ValueError(f"'time' column missing in {files[i]}")

#     # common overlapping time grid
#     t_min = max(df["time"].min() for df in dfs)
#     t_max = min(df["time"].max() for df in dfs)
#     if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
#         raise ValueError("Non-overlapping time ranges across files.")
#     tg = np.linspace(t_min, t_max, n_points)

#     means = {}
#     stds  = {}
#     for c in cols:
#         series = []
#         for df in dfs:
#             if c not in df.columns:
#                 raise ValueError(f"Column '{c}' missing in one of the files for pattern {files_pattern}")
#             series.append(np.interp(tg, df["time"].values, df[c].values))
#         arr = np.vstack(series)  # runs × time
#         means[c] = arr.mean(axis=0)
#         stds[c]  = arr.std(axis=0)
#     return tg, means, stds

# # ---------- Load data ----------
# U_COLS = [f"u{i}" for i in range(1, 10)]

# time_clf, mean_clf, std_clf = load_interp_mean_std(clf_pattern, U_COLS, N_POINTS)
# imp_pattern = pick_existing_pattern(imp_patterns)
# time_imp, mean_imp, std_imp = load_interp_mean_std(imp_pattern, U_COLS, N_POINTS)

# #CLF
# # ---------- Task error comparison ----------
# TASK_COL = "task_error"

# # interpolate mean ± std for both CLF and IMP runs
# time_task_clf, mean_task_clf, std_task_clf = load_interp_mean_std(clf_pattern, [TASK_COL], N_POINTS)
# time_task_imp, mean_task_imp, std_task_imp = load_interp_mean_std(imp_pattern, [TASK_COL], N_POINTS)

# plt.figure(figsize=(8, 4.8))

# # CLF (solid blue)
# plt.plot(time_task_clf, mean_task_clf[TASK_COL], color="tab:blue", linewidth=2.5, label="ID-CLF-QP")
# plt.fill_between(
#     time_task_clf,
#     mean_task_clf[TASK_COL] - std_task_clf[TASK_COL],
#     mean_task_clf[TASK_COL] + std_task_clf[TASK_COL],
#     color="tab:blue", alpha=0.15, linewidth=0
# )

# # Impedance (dashed red)
# plt.plot(time_task_imp, mean_task_imp[TASK_COL], color="tab:red", linewidth=2.5,
#           label="Impedance")
# plt.fill_between(
#     time_task_imp,
#     mean_task_imp[TASK_COL] - std_task_imp[TASK_COL],
#     mean_task_imp[TASK_COL] + std_task_imp[TASK_COL],
#     color="tab:red", alpha=0.15, linewidth=0
# )

# plt.xlabel("Time (s)",fontsize = 12)
# plt.ylabel("Task error (m)",fontsize = 12)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize = 14)
# plt.tight_layout()


# # ---------- Plot ----------
# plt.figure(figsize=(12, 8))
# axes = []
# for k in range(9):
#     ax = plt.subplot(3, 3, k + 1)
#     ucol = f"u{k+1}"

#     # ID-CLF-QP (solid blue)
#     ax.plot(time_clf, mean_clf[ucol], label="CLF", color="tab:blue", linewidth=LINEWIDTH)
#     ax.fill_between(time_clf,
#                     mean_clf[ucol] - std_clf[ucol],
#                     mean_clf[ucol] + std_clf[ucol],
#                     color="tab:blue", alpha=0.15, linewidth=0)

#     # Impedance (dashed red)
#     ax.plot(time_imp, mean_imp[ucol], label="IMP", color="tab:red",
#             linewidth=LINEWIDTH, linestyle=DASH)
#     ax.fill_between(time_imp,
#                     mean_imp[ucol] - std_imp[ucol],
#                     mean_imp[ucol] + std_imp[ucol],
#                     color="tab:red", alpha=0.15, linewidth=0)

#     # Control limits (green)
#     ax.axhline(+CONTROL_LIMIT, color="tab:green", linewidth=2)
#     ax.axhline(-CONTROL_LIMIT, color="tab:green", linewidth=2)

#     ax.set_title(f"U{k+1}")
#     ax.set_xlim(min(time_clf[0], time_imp[0]), max(time_clf[-1], time_imp[-1]))
#     ax.grid(True, alpha=0.3)
#     if k // 3 == 2:  # bottom row: show x label
#         ax.set_xlabel("Time [s]")


#     axes.append(ax)

# # --- remove individual x labels (we'll add a shared one)
# for ax in axes:
#     ax.set_xlabel("")

# # shared x/y labels for the whole figure
# fig = plt.gcf()
# fig.supxlabel("Time (s)", fontsize=14)
# fig.supylabel("Control Input U (Nm)", fontsize=14)

# # build a clean, single legend (CLF, IMP, Control Limit) for all subplots
# legend_handles = [
#     Line2D([0], [0], color="tab:blue", linewidth=LINEWIDTH, label="CLF"),
#     Line2D([0], [0], color="tab:red",  linewidth=LINEWIDTH, linestyle=DASH, label="IMP"),
#     Line2D([0], [0], color="tab:green", linewidth=2, label="Control Limit"),
# ]

# fig.legend(
#     handles=legend_handles,
#     loc="upper center",
#     ncol=3,
#     bbox_to_anchor=(0.5, 0.98),
#     frameon=False,
#     fontsize=12,
# )

# plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave room for the legend at the top
# plt.show()



# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
