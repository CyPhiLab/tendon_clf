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
import pandas as pd
from utils import *

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

def controller(experiment, control_scheme, model_name, target, model, data, invariants, previous_solution=None):
    jac = np.zeros((6, model.nv))
    m = invariants['m']
    twist = np.zeros(m)

    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    site_name = "ee"
    site_id = model.site(site_name).id

    if experiment == 'tracking':
        if m == 3:
            data.mocap_pos[mocap_id] = target['pos']
            target_vel = target['vel']
            target_acc = target['acc']
        elif m == 6:
            data.mocap_pos[mocap_id] = target['pos']
            target_vel = np.hstack([target['vel'], [0, 0, 0]])
            target_acc = np.hstack([target['acc'], [0, 0, 0]])

    elif experiment == 'set':
            data.mocap_pos[mocap_id] = target
            target_vel = np.zeros(m)
            target_acc = np.zeros(m)

    twist[:3] = data.mocap_pos[mocap_id] - data.site(site_id).xpos
    task_error = np.linalg.norm(twist[:3])

    if model_name != 'tendon':
        twist[3:] *= Kori 
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        twist[3:] = 0.0
    

    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    
    # Compute the Jacobian.
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # Compute the inertia matrix.
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M * np.random.normal(1.0, 0.01, size=M.shape)  # Add some noise to the inertia matrix
    dJ_dt = compute_jacobian_derivative(model, data, site_id)
    dJ_dt = dJ_dt[:m,:]
    jac = jac[:m,:]

    if control_scheme == 'id_clf_qp':
        # error and Lyapunov function for CLF
        eta = np.concatenate((-twist,jac @ data.qvel - target_vel), axis=0)
        V, previous_solution, u = id_clf_qp_control(model_name, model, data, invariants, eta, 
                                                    target_vel, target_acc, twist, jac, M, dJ_dt, previous_solution)
        return V, task_error, previous_solution, u
    elif control_scheme == 'impedance':
        u = impedance_control(model_name, model, data, invariants, target_vel, target_acc, twist, jac, M_inv, dJ_dt)
        return task_error, u
    elif control_scheme == 'mpc':
        previous_solution, u = mpc_control(model_name, model, data, invariants, 
                                           target_vel, target_acc, twist, jac, M, dJ_dt, previous_solution=None)
        return task_error, previous_solution, u



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized tendon control simulation')
    parser.add_argument('--headless', action='store_true', help='Run simulation without GUI for maximum performance')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots at the end')
    parser.add_argument('--control_scheme', type=str, default='id_clf_qp', choices=['id_clf_qp', 'impedance', 'mpc', 'osc'], help='Controller type to use')
    parser.add_argument('--robot', type=str, default='helix', choices=['helix', 'tendon','spirob'], help='Robot to simulate')
    parser.add_argument('--experiment', type=str, default='set', choices=['set', 'tracking'], help='Experiment name')
    parser.add_argument('--target_pos', type=str, default='pos4', choices=['pos1', 'pos2', 'pos3', 'pos4'], help='Target position for the end-effector')
    args = parser.parse_args()
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Simulate the model
    if args.control_scheme == 'id_clf_qp':
        V_log, task_error_log, time_log, sim_ts, u_log = simulate_model(headless=args.headless,
                                                                        control_scheme=args.control_scheme, 
                                                                        target_pos=args.target_pos, 
                                                                        controller=controller, 
                                                                        experiment=args.experiment, 
                                                                        model_name=args.robot)
    else:
        task_error_log, time_log, sim_ts, u_log = simulate_model(headless=args.headless,
                                                                 control_scheme=args.control_scheme, 
                                                                 target_pos=args.target_pos, 
                                                                 controller=controller, 
                                                                 experiment=args.experiment, 
                                                                 model_name=args.robot)

    # Record end time
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds (wall clock time)")
    print(f"Simulated {sim_ts['ts'][-1]:.3f} seconds of physics time")
    print(f"Performance ratio: {sim_ts['ts'][-1] / (end_time - start_time):.2f}x real-time")
    
    if args.no_plots:
        print("Skipping plot generation")
        exit(0)
        


    # Save control inputs and task error to CSV
    if args.control_scheme == 'id_clf_qp':
        V_log = np.array(V_log)
    u_log = np.array(u_log)
    time_log_csv = np.array(time_log)
    sim_time_csv = np.array(sim_ts['ts'])
    error_log = np.array(task_error_log)
    df = pd.DataFrame(
        u_log,
        columns=[f"u{i}" for i in range(u_log.shape[1])]
    )
    df.insert(0, "time", time_log_csv)
    df.insert(1, "sim_time", sim_time_csv)
    if args.control_scheme == 'id_clf_qp':
        df.insert(2, "lyapunov_V", V_log)
    df.insert(3, "task_error", error_log)
    # Create results directory if it doesn't exist
    if args.experiment == 'set':
        os.makedirs(f"results/{args.robot}/{args.control_scheme}", exist_ok=True)
    if args.experiment == 'tracking':
        os.makedirs(f"results/{args.robot}/{args.control_scheme}", exist_ok=True)
    if args.experiment == 'set':
        csv_path = f"results/{args.robot}/{args.control_scheme}/{args.experiment}_{args.control_scheme}_{args.target_pos}.csv"
    if args.experiment == 'tracking':
        csv_path = f"results/{args.robot}/{args.control_scheme}/{args.experiment}_{args.control_scheme}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # #Lyapunov Function Over Time – shows how stability evolves.
    # plt.figure()
    # plt.plot(time_log, V_log, label="Lyapunov Function V")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Lyapunov Function V")
    # plt.title("Lyapunov Function Over Time")
    # plt.grid(True)
    # plt.legend()  # <--- Add legend

    # #End-Effector Position Error – checks task-space convergence.
    # plt.figure()
    # plt.plot(time_log, task_error_log, label="‖x_desired - x_actual‖")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Task-Space Position Error (m)")
    # plt.title("End-Effector Position Error Over Time")
    # plt.grid(True)
    # plt.legend()

    # # # Control inputs plot
    # # actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
    # # ctrl = np.array(sim_ts["ctrl"])  # shape: (timesteps, nu)
    # # num_actuators = ctrl.shape[1]
    # # control_limit = 25

    # # fig, axs = plt.subplots(actuated_indices, 1, figsize=(10, 8), sharex=True)

    # # for i in range(actuated_indices):
    # #     axs[i].plot(time_log, ctrl[:, i], label=f"Actuator {i}")
    # #     axs[i].axhline(control_limit, color='r', linestyle='--', linewidth=1)
    # #     axs[i].axhline(-control_limit, color='r', linestyle='--', linewidth=1)
    # #     axs[i].set_ylabel("Torque (Nm)")
    # #     axs[i].legend()
    # #     axs[i].grid(True)

    # # axs[-1].set_xlabel("Time (s)")
    # # fig.suptitle("Control Inputs at Actuated Joints")
    # # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

