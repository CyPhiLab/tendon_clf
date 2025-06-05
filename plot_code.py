import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_path = r"D:\Control\Soft Arm Project\Ablation Study\Full.csv"
df = pd.read_csv(file_path)

# Time and core metrics
time = df["time"].values
V = df["V"].values
V_dot = df["V_dot"].values
V_dot = np.array([float(str(v).strip("[]")) for v in V_dot])

stability_bound = df["stability_bound"].values
task_error = df["task_error"].values

# Control inputs
u_0 = df["u_0 (A Flexor)"].values
u_1 = df["u_1 (A Extensor)"].values
u_2 = df["u_2 (B Flexor)"].values
u_3 = df["u_3 (B Extensor)"].values

# Joint torques
tau_0 = df["tau_0"].values
tau_1 = df["tau_1"].values
tau_2 = df["tau_2"].values
tau_3 = df["tau_3"].values

# --- Lyapunov Function ---
plt.figure()
plt.plot(time, V, label="Lyapunov Function V")
plt.xlabel("Time (s)")
plt.ylabel("V")
plt.title("Lyapunov Function Over Time")
plt.grid(True)
plt.legend()

# --- V_dot vs Stability Bound ---
plt.figure()
plt.plot(time, V_dot, label="V_dot")
plt.plot(time, stability_bound, label="−2/e·V + dl")
plt.xlabel("Time (s)")
plt.ylabel("V_dot")
plt.title("V_dot vs Stability Bound")
plt.grid(True)
plt.legend()

# --- Task-Space Error ---
plt.figure()
plt.plot(time, task_error, label="‖x_des - x‖")
plt.xlabel("Time (s)")
plt.ylabel("Task-Space Error (m)")
plt.title("End-Effector Task Error")
plt.grid(True)
plt.legend()

# --- Control Inputs u ---
fig_u, axs_u = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
u_labels = ["u_0 (A Flexor)", "u_1 (A Extensor)", "u_2 (B Flexor)", "u_3 (B Extensor)"]
u_data = [u_0, u_1, u_2, u_3]

for i in range(4):
    axs_u[i].plot(time, u_data[i])
    axs_u[i].set_ylabel(u_labels[i])
    axs_u[i].grid(True)
axs_u[-1].set_xlabel("Time (s)")
fig_u.suptitle("Control Inputs Over Time")
fig_u.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Joint Torques τ ---
fig_tau, axs_tau = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
tau_labels = ["tau_0", "tau_1", "tau_2", "tau_3"]
tau_data = [tau_0, tau_1, tau_2, tau_3]

for i in range(4):
    axs_tau[i].plot(time, tau_data[i])
    axs_tau[i].set_ylabel(tau_labels[i])
    axs_tau[i].grid(True)
axs_tau[-1].set_xlabel("Time (s)")
fig_tau.suptitle("Joint Torques Over Time")
fig_tau.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show all plots
plt.show()

# Folder containing all CSVs
folder = r"D:\Control\Soft Arm Project\Ablation Study"

# Create plot
plt.figure(figsize=(10, 6))

for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)

        # Clean up 'V' column in case it's stored as "[value]"
        df["V"] = df["V"].apply(lambda x: float(str(x).strip("[]")))
        time = df["time"]
        V = df["V"]

        label = file.replace(".csv", "")
        plt.plot(time, V, label=label)

plt.xlabel("Time (s)")
plt.ylabel("Lyapunov Function V")
plt.title("Comparison of Lyapunov V Across Ablation Conditions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()