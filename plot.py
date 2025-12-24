import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

# Pattern for your files
files = sorted(glob.glob("simulation_results_cd_pos*.csv"))

# Storage
data = {}

for f in files:
    df = pd.read_csv(f)
    
    # Ensure consistent column naming
    df.columns = [c.strip().lower() for c in df.columns]
    
    time_vec = df["time"].values
    err_vec = df["task_error"].values
    
    data[f] = {
        "time": time_vec,
        "error": err_vec
    }

    

# # Example: print shapes
# for fname, vals in data.items():
#     print(f"{fname}: time={len(vals['time'])}, clf={len(vals['clf'])}, error={len(vals['error'])}")

# t = data["simulation_results_pos3.csv"]["time"]
# clf = data["simulation_results_pos3.csv"]["clf"]
# err = data["simulation_results_pos3.csv"]["error"]


plt.figure(figsize=(8, 5))


for fname, vals in data.items():
    # Extract number from "posX"
    match = re.search(r"pos(\d+)", fname)
    if match:
        label = f"Target {match.group(1)}"
    else:
        label = fname
    
    plt.plot(vals["time"], vals["error"], label=label)
    
    # Print final error
    final_error = vals["error"][-1]
    print(f"{label}: final error = {final_error:.4f}")

plt.xlabel("Time [s]")
plt.ylabel("Task Error (m)")
plt.title("Errors Convergence at Different Target Positions")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
