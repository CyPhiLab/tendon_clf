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
from ctrl_utils import *

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


def controller(model, data, invariants, previous_solution=None):
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Extract pre-computed values
    F = invariants['F']
    G = invariants['G']
    Kp = invariants['Kp']
    Kd = invariants['Kd']
    Pe = invariants['Pe']
    pinv_B = invariants['pinv_B']
    sel = invariants['sel']
    e = invariants['e']

    site_name = "ee"
    site_id = model.site(site_name).id

    L = 0.435/2
    R = L/2
    h = 0.7 - 2*L
    data.mocap_pos[mocap_id] = np.array([L-R, 0.0, L+h])
    # data.mocap_pos[mocap_id] = np.array([L+R, 0.0, L+h])
    # data.mocap_pos[mocap_id] = np.array([L, 0.0, L+R+h])
    # data.mocap_pos[mocap_id] = np.array([L, 0.0, L-R+h])

    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos

    twist[:3] = dx 
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori 
    twist[3:] = 0.0
    q = data.qpos
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M * np.random.normal(1.0, 0.01, size=M.shape)  # Add some noise to the inertia matrix
    dJ_dt = compute_jacobian_derivative(model, data, site_id)

    # define decision variables (create fresh variables each time)
    nu = 9
    nq = model.nq
    u = cp.Variable(shape=(nu, 1))
    qdd = cp.Variable(shape=(nq, 1))
    dl = cp.Variable(shape=(1, 1))


    # error and Lyapunov function for CLF
    eta = np.concatenate((-twist,jac @ data.qvel))
    V = eta.T @ Pe @ eta

    # Use original constraint formulation
    dq = data.qvel.reshape(-1,1)
    
    N = np.eye(model.nv) - np.linalg.pinv(jac) @ jac
    qdd_null = N @ qdd

    # Vdot for our main CLF
    dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd)
    
    
    objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - (Kp * twist.reshape(-1,1) - Kd * (jac @ dq)))) + 0.2 * cp.square(cp.norm(qdd))  
                            + 0.5 * cp.square(cp.norm(u)) + 1000 * cp.square(dl) + 0.5 * cp.square(cp.norm(qdd_null - 0*dq.reshape(-1,1))))

    constraints = [ dV <= - 1/e * V + dl, 
                    pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) + data.qfrc_passive.reshape(-1,1)) == u,
                    -25*sel <= u,
                    25*np.ones((nu,1)) >= u]

    prob = cp.Problem(objective=objective, constraints=constraints)
    
    # Warm start with previous solution if available
    if previous_solution is not None:
        try:
            u.value = previous_solution['u']
            qdd.value = previous_solution['qdd'] 
            dl.value = previous_solution['dl']
        except:
            pass  # If warm start fails, proceed without it

    task_error = np.linalg.norm(dx)
    q = data.qpos
    dq = data.qvel
    
    try:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        if u.value is not None:
            data.ctrl = np.squeeze(B @ u.value)
            
            # Cache solution for next iteration
            current_solution = {
                'u': u.value.copy(),
                'qdd': qdd.value.copy(),
                'dl': dl.value.copy()
            }
            # print(f"converged\n")
            return V, task_error, q, dq, current_solution, u.value.copy()
        else:
            print(f"failed convergence - no solution\n")
            return V, task_error, q, dq, previous_solution
    except Exception as e:
        print(f"failed convergence - exception: {e}\n")
        return V, task_error, q, dq, previous_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized tendon control simulation')
    parser.add_argument('--headless', action='store_true', help='Run simulation without GUI for maximum performance')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots at the end')
    parser.add_argument('--controller', type=str, default='id_clf_qp', choices=['id_clf_qp', 'impedance', 'mpc', 'osc'], help='Controller type to use')
    parser.add_argument('--environment', type=str, default='helix', choices=['helix', 'finger'], help='Environment to simulate')
    args = parser.parse_args()
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Simulate the model
    V_log, task_error_log, q_vel, q_pos, time_log, sim_ts, u_log = simulate_model(controller, headless=args.headless)
    
    # Record end time
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds (wall clock time)")
    print(f"Simulated {sim_ts['ts'][-1]:.3f} seconds of physics time")
    print(f"Performance ratio: {sim_ts['ts'][-1] / (end_time - start_time):.2f}x real-time")
    
    if args.no_plots:
        print("Skipping plot generation")
        exit(0)

    if not args.no_plots:
        plot_helpers(time_log, V_log, task_error_log, q_pos, q_vel, u_log)

    save_data(time_log, q_vel, q_pos, V_log, task_error_log, u_log)
   