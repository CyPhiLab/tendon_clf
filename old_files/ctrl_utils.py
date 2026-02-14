import argparse
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import os
from pathlib import Path
import time
from scipy import linalg
import cvxpy as cp
from scipy.linalg import eigh
import gurobipy
import csv
import pandas as pd

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


model_name = f"helix_control"

# Cartesian impedance control gains.
impedance_pos = np.asarray([50.0, 50.0, 50.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.


# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95
stiffness = 0.2


f_ctrl = 2000.0 
T = 1
w = 2 * np.pi / T

B = np.zeros((36, 9))
for i in range(3):  # Iterate over u1, u2, u3 blocks
    for j in range(4):  # Repeat each block 4 times
        row_start = i * 12 + j * 3  # Compute row index
        col_start = i * 3  # Compute column index
        B[row_start:row_start+3, col_start:col_start+3] = np.eye(3)  # Assign identity

def get_coriolis_and_gravity(model, data):
    """
    Calculate the Coriolis matrix and gravity vector for a MuJoCo model

    Parameters:
        model: MuJoCo model object
        data: MuJoCo data object

    Returns:
        C: Coriolis matrix (nv x nv)
        g: Gravity vector (nv,)
    """
    nv = model.nv  # number of degrees of freedom

    # Calculate gravity vector
    g = np.zeros(nv)
    dummy = np.zeros(nv,)
    mujoco.mj_factorM(model, data)  # Compute sparse M factorization
    mujoco.mj_rne(model, data, 0, dummy)  # Run RNE with zero acceleration and velocity
    g = data.qfrc_bias.copy()

    # Calculate Coriolis matrix
    C = np.zeros((nv, nv))
    q_vel = data.qvel.copy()

    # Compute each column of C using finite differences
    eps = 1e-6
    for i in range(nv):
        # Save current state
        vel_orig = q_vel.copy()

        # Perturb velocity
        q_vel[i] += eps
        data.qvel = q_vel

        # Calculate forces with perturbed velocity
        mujoco.mj_rne(model, data, 0, dummy)
        tau_plus = data.qfrc_bias.copy()

        # Restore original velocity
        q_vel = vel_orig
        data.qvel = q_vel

        # Compute column of C using finite difference
        C[:, i] = (tau_plus - data.qfrc_bias) / eps

    return C, g

def compute_jacobian_derivative(model, data, site_id, h=1e-6):
    """
    Compute the time derivative of the Jacobian in MuJoCo.
    
    Parameters:
    - model: The MuJoCo model (mjModel).
    - data: The MuJoCo data structure (mjData).
    - jac_func: Function to compute the Jacobian (e.g., mj_jacBody or mj_jacSite).
    - h: Small positive step for numerical differentiation.
    
    Returns:
    - Jdot: The time derivative of the Jacobian.
    """
    # Step 1: Update kinematics
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    
    # Step 2: Compute the initial Jacobian
    J = np.zeros((6, model.nv))  # Assuming a 6xnv Jacobian for full spatial representation
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)
    
    # Step 3: Integrate position using velocity
    qpos_backup = np.copy(data.qpos)  # Backup original qpos
    mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
    
    # Step 4: Update kinematics again
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    
    # Step 5: Compute the new Jacobian
    Jh = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, Jh[:3], Jh[3:], site_id)
    
    # Step 6: Compute Jdot
    Jdot = (Jh - J) / h
    
    # Step 7: Restore qpos
    data.qpos[:] = qpos_backup
    
    return Jdot

# Pre-compute invariant matrices (computed once, used throughout simulation)
def precompute_invariants(model):
    """Pre-compute matrices that don't change during simulation"""
    Kp_null = np.asarray([1] * model.nv)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)
    
    # CLF matrices
    m = 6
    F = np.zeros((2*m, 2*m))
    F[:m, m:] = np.eye(m, m)
    G = np.zeros((2*m, m))
    G[m:, :] = np.eye(m)
    e = 0.1
    Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))
    Kp = 200
    Kd = 2 * np.sqrt(Kp)
    
    # Input matrix pseudoinverse
    pinv_B = np.linalg.pinv(B)
    
    # Selection matrix for compression/extension actuators
    nu = 9
    sel = np.ones((nu, 1))
    sel[[2, 5, 8]] = 0.0
    
    return {
        'Kp_null': Kp_null,
        'Kd_null': Kd_null,
        'F': F,
        'G': G,
        'Pe': Pe,
        'pinv_B': pinv_B,
        'sel': sel,
        'e': e,
        'Kp': Kp,
        'Kd': Kd
    }

def simulate_model(controller, headless=False):
    model_path = Path("mujoco_models/helix") / (str(model_name) + str(".xml"))
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    model.jnt_stiffness[:] = stiffness
    
    model.dof_damping[:] = 1.0
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    data.qpos[2] = 0.0
    model.jnt_range[range(2,len(data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(data.qpos),3)]
    model.jnt_stiffness[range(2,len(data.qpos),3)] = 0.2
    # model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Pre-compute invariant matrices
    print("Pre-computing invariant matrices...")
    invariants = precompute_invariants(model)
    
    # Initialize solution cache
    previous_solution = None

    sim_ts = dict(
        ts=[],
        base_pos=[],
        base_vel=[],
        base_acc=[],
        base_force=[],
        base_torque=[],
        q=[],
        qvel=[],
        ctrl=[],
        actuator_force=[],
        qfrc_fluid=[],
        q_des=[],
    )
    time_last_ctrl = 0.0
    q_des = np.ones(data.qpos.shape[0])*0.2

    V_log = []
    task_error_log = []
    u_log = []
    q_vel = []
    q_pos = []
    time_log = []
    t = 0.0
    dt = model.opt.timestep


    last_ctrl = time.time()
    max_sim_time = 125.0  # Run for 1 second of simulation time
    log_frequency = 5  # Log every 5 steps instead of every step
    step_count = 0
    threshold  = 1e-3

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        while (len(task_error_log) < 2 or
            abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold ):
            step_start = time.time()
            V, task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution)
            mujoco.mj_step(model, data)
            
            # Only log every N steps to reduce overhead
            if step_count % log_frequency == 0:
                print(f"Sim time: {data.time:.3f}s")
                
                sim_ts["ts"].append(data.time)
                # extract the sensor data
                sim_ts["base_pos"].append(data.sensordata[:3].copy())
                sim_ts["base_vel"].append(data.sensordata[3:6].copy())
                sim_ts["base_acc"].append(data.sensordata[6:9].copy())
                sim_ts["base_force"].append(data.sensordata[9:12].copy())
                sim_ts["base_torque"].append(data.sensordata[12:15].copy())
                sim_ts["q"].append(data.qpos.copy())
                sim_ts["qvel"].append(data.qvel.copy())
                sim_ts["ctrl"].append(data.ctrl.copy())
                sim_ts["actuator_force"].append(data.actuator_force.copy())
                sim_ts["qfrc_fluid"].append(data.qfrc_fluid.copy())
                sim_ts["q_des"].append(q_des.copy())

                V_log.append(V)
                task_error_log.append(task_error)
                if u is not None:
                    u_log.append(u.squeeze())
                else:
                    u_log.append(np.full(9, np.nan))                
                q_vel.append(dq.squeeze().copy())
                q_pos.append(q.squeeze().copy())
                time_log.append(t)
            
            step_count += 1
            t += dt
    else:
        # Run simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("ortho_side").id

            sim_start = time.time()
            while viewer.is_running() and (
                len(task_error_log) < 2 or
                abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold 
            ):
                step_start = time.time()
                first_time = time.time()
                V, task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution)
                mujoco.mj_step(model, data)
                
                # Only log every N steps to reduce overhead
                if step_count % log_frequency == 0:
                    print(data.time)
                    
                    sim_ts["ts"].append(data.time)
                    # extract the sensor data
                    sim_ts["base_pos"].append(data.sensordata[:3].copy())
                    sim_ts["base_vel"].append(data.sensordata[3:6].copy())
                    sim_ts["base_acc"].append(data.sensordata[6:9].copy())
                    sim_ts["base_force"].append(data.sensordata[9:12].copy())
                    sim_ts["base_torque"].append(data.sensordata[12:15].copy())
                    sim_ts["q"].append(data.qpos.copy())
                    sim_ts["qvel"].append(data.qvel.copy())
                    sim_ts["ctrl"].append(data.ctrl.copy())
                    sim_ts["actuator_force"].append(data.actuator_force.copy())
                    sim_ts["qfrc_fluid"].append(data.qfrc_fluid.copy())
                    sim_ts["q_des"].append(q_des.copy())

                    V_log.append(V)
                    if u is not None:
                        u_log.append(u.squeeze())
                    else:
                        u_log.append(np.full(9, np.nan))
                    task_error_log.append(task_error)
                    q_vel.append(dq.squeeze().copy())
                    q_pos.append(q.squeeze().copy())
                    time_log.append(t)
                
                step_count += 1
                
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                t += dt

                # Rudimentary time keeping, will drift relative to wall clock.
                # Removed sleep to run as fast as possible for 1 second of sim time
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)
    print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")
    return V_log, task_error_log, q_vel, q_pos, time_log, sim_ts, u_log

def plot_helpers(time_log, V_log=None, task_error_log=None, q_pos=None, q_vel=None, u_log=None, control_limit=25, show=True):
    if V_log is not None:
        plt.figure()
        plt.plot(time_log, V_log)
        plt.title("Lyapunov Function V over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("V")
        plt.grid()

    if task_error_log is not None:
        plt.figure()
        plt.plot(time_log, task_error_log)
        plt.title("Task Error over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Task Error (norm)")
        plt.grid()

    # if q_pos is not None:
    #     plt.figure()
    #     for i in range(q_pos.shape[1]):
    #         plt.plot(time_log, q_pos[:, i], label=f"q{i}")
    #     plt.title("Joint Positions over Time")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Joint Positions (rad)")
    #     plt.legend()
    #     plt.grid()

    # if q_vel is not None:
    #     plt.figure()
    #     for i in range(q_vel.shape[1]):
    #         plt.plot(time_log, q_vel[:, i], label=f"dq{i}")
    #     plt.title("Joint Velocities over Time")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Joint Velocities (rad/s)")
    #     plt.legend()
    #     plt.grid()

    # if u_log is not None:
    #     plt.figure()
    #     for i in range(u_log.shape[1]):
    #         plt.plot(time_log, u_log[:, i], label=f"u{i}")
    #     plt.title("Control Inputs over Time")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Control Inputs")
    #     plt.ylim([-control_limit, control_limit])
    #     plt.legend()
    #     plt.grid()

    if show:
        plt.show()

import numpy as np
import pandas as pd

def save_data(time_log, q_vel, q_pos, V_log, task_error_log, u_log):
    """
    Saves simulation logs to CSV.
    - Handles V_log = None (for non-CLF controllers)
    - Prompts user for filename before saving
    """

    # Convert to numpy arrays
    time_log = np.array(time_log)
    q_vel = np.array(q_vel)
    q_pos = np.array(q_pos)
    u_log = np.array(u_log)
    error_log = np.array(task_error_log)

    # Create base DataFrame with control inputs
    df = pd.DataFrame(
        u_log,
        columns=[f"u{i}" for i in range(u_log.shape[1])]
    )

    # Insert common columns
    df.insert(0, "time", time_log)
    df.insert(1, "task_error", error_log)

    # Insert Lyapunov column only if available
    if V_log is not None:
        V_log = np.array(V_log)
        df.insert(2, "V", V_log)

    # Ask user for filename
    default_name = "simulation_log.csv"
    filename = input(f"Enter CSV filename (default: {default_name}): ").strip()

    if filename == "":
        filename = default_name

    if not filename.endswith(".csv"):
        filename += ".csv"

    # Save
    df.to_csv(filename, index=False)
    print(f"Saved CSV to {filename}")
