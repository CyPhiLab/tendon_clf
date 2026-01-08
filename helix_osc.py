
import argparse
from ast import Lambda
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

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


model_name = f"helix_control"

# ----------------------------
# "Santina-style" task-space impedance (acc-level) parameters
# ----------------------------
# Keep the same variable naming style as your template
TASK_DIM = 6  # 3D tip-position (m=3)

# Gains for the twist computation (kept from your template)
Kori: float = 0.95
stiffness = 0.01

# Damped least-squares in synergy space (Santina projector solve)
SOLVE_LAM = 1e-6


# (Kept from your template; not used directly here but preserved)
f_ctrl = 2000.0
T = 1
w = 2 * np.pi / T


B = np.zeros((36, 9))
for i in range(3):  # Iterate over u1, u2, u3 blocks
    for j in range(4):  # Repeat each block 4 times
        row_start = i * 12 + j * 3  # Compute row index
        col_start = i * 3  # Compute column index
        B[row_start:row_start + 3, col_start:col_start + 3] = np.eye(3)  # Assign identity

print(B)


def get_coriolis_and_gravity(model, data):
    """
    Calculate the Coriolis matrix and gravity vector for a MuJoCo model.

    NOTE:
    This matches the "expensive but consistent" finite-difference approach:
    - g is from qfrc_bias at current state
    - C is estimated by perturbing qvel and re-running RNE
    """
    nv = model.nv
    dummy = np.zeros(nv,)

    # Make sure kinematics are current
    mujoco.mj_forward(model, data)

    # Gravity/bias at current qvel (includes C(q,qd)qd + g(q) + passive internal terms)
    mujoco.mj_rne(model, data, 0, dummy)
    g = data.qfrc_bias.copy()

    # Coriolis matrix by finite differences over qvel
    C = np.zeros((nv, nv))
    q_vel_orig = data.qvel.copy()

    eps = 1e-6
    for i in range(nv):
        data.qvel[:] = q_vel_orig
        data.qvel[i] += eps

        mujoco.mj_forward(model, data)
        mujoco.mj_rne(model, data, 0, dummy)
        tau_plus = data.qfrc_bias.copy()

        data.qvel[:] = q_vel_orig
        mujoco.mj_forward(model, data)
        mujoco.mj_rne(model, data, 0, dummy)
        tau_base = data.qfrc_bias.copy()

        C[:, i] = (tau_plus - tau_base) / eps

    # restore
    data.qvel[:] = q_vel_orig
    mujoco.mj_forward(model, data)

    return C, g


def compute_jacobian_derivative(model, data, site_id, h=1e-6):
    """
    Jdot via finite differences:
      Jdot ≈ (J(q + h*qdot) - J(q)) / h

    This mirrors your template function ordering and style, but includes the
    necessary mj_forward calls to keep everything consistent.
    """
    mujoco.mj_forward(model, data)

    # Base Jacobian
    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)

    # Backup
    qpos_backup = np.copy(data.qpos)
    qvel_backup = np.copy(data.qvel)

    # Integrate forward
    mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
    mujoco.mj_forward(model, data)

    # New Jacobian
    Jh = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, Jh[:3], Jh[3:], site_id)

    # Restore
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)

    return (Jh - J) / h


def compute_fixed_S_from_qbar(model, data, site_id, task_dim, qbar=None):
    """
    Compute a *fixed* synergy matrix S (nv x m) at reference posture qbar.
    Paper-consistent: S is constant.
    Practical choice: S = Jm(qbar)^T.
    """
    # Backup state
    state_sz = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    backup = np.zeros(state_sz)
    mujoco.mj_getState(model, data, backup, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    # Set qbar
    if qbar is None:
        qbar = data.qpos.copy()
    data.qpos[:] = qbar
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    jac = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    S = jac.T.copy()  # (nv x m)

    # Restore
    mujoco.mj_setState(model, data, backup, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    return S

def precompute_invariants(model, data):
    """Pre-compute matrices that don't change during simulation"""
    # Input matrix pseudoinverse
    pinv_B = np.linalg.pinv(B)

    # IDs
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    site_name = "ee"
    site_id = model.site(site_name).id

    # Fixed S (computed ONCE)
    S = compute_fixed_S_from_qbar(model, data, site_id, task_dim=TASK_DIM, qbar=None)

    return {
        "pinv_B": pinv_B,
        "site_id": site_id,
        "mocap_id": mocap_id,
        "S": S,
    }


def controller(model, data, invariants, previous_solution=None, S=None):
    """
    Replicates your controller() function style as closely as possible, but
    implements the Santina fixed-S projection pipeline from your earlier script.
    """
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))

    mocap_id = invariants["mocap_id"]
    site_id = invariants["site_id"]
    pinv_B = invariants["pinv_B"]

    # Fixed S: prefer passed-in S (your style), else invariants S
    if S is None:
        S = invariants["S"]


    # --- Target pose (same naming / style)
    # data.mocap_pos[mocap_id] = np.array([0.2, 0.2, 0.3])
    # data.mocap_pos[mocap_id] = np.array([0.23312815, -0.31631513, 0.44791383])
    # data.mocap_pos[mocap_id] = np.array([0.27083678, 0.00194196, 0.58488434])
    data.mocap_pos[mocap_id] = ([-0.055,   0.06,  0.45])

    # --- Pose error -> twist (same as your template)
    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
    twist[:3] = dx

    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori

    # for TASK_DIM=3, match your behavior
    twist[3:] = 0.0

    # --- Jacobian (match your kinematics calls)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # --- Mass matrix inverse
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M * np.random.normal(1.0, 0.01, size=M.shape)

    # --- Jdot
    dJ_dt = compute_jacobian_derivative(model, data, site_id)

    # --- Use original constraint formulation variables
    dq = data.qvel.reshape(-1, 1)
    task_error = np.linalg.norm(dx)
    q = data.qpos

    # Task-space inertia 
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    Jbar = M_inv @ jac.T @ Mx
    C, g = get_coriolis_and_gravity(model, data)
    ydd = 500 * twist - 20 * (jac @ data.qvel)
    Cy = (Jbar.T @ C @ data.qvel) - (Mx @ (dJ_dt @ data.qvel))
    f = Mx @ ydd + Cy
    # f = Mx @ ydd + Cy + Jbar.T @ (g + data.qfrc_passive)

    # Santina synergy-projected torque
    sigma = np.linalg.pinv(jac @ M_inv @ S, rcond=SOLVE_LAM) @ (jac @ M_inv @ jac.T @ f)  # (m,)
    tau_synergy = S @ sigma  # (nv,)

    # Add gravity + passive 
    tau = tau_synergy + g - data.qfrc_passive
    # tau = B @ np.linalg.pinv(jac @ M_inv @ B) @ jac @ M_inv @ jac.T @ f


    u = tau

    try:
        data.ctrl = u
    except Exception as e:
        print(f"failed convergence: {e}\n")
        pass

    return task_error, q, dq, previous_solution


def simulate_model(headless=False):
    model_path = Path("mujoco_models/helix") / (str(model_name) + str(".xml"))

    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    model.jnt_stiffness[:] = stiffness
    model.dof_damping[:] = 0.2
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    # init
    if data.qpos.shape[0] > 2:
        data.qpos[2] = 0.0

    # keep your template edits
    model.jnt_range[range(2, len(data.qpos), 3)] = [[-0.001, 0.03 / 2] for _ in range(2, len(data.qpos), 3)]
    model.jnt_stiffness[range(2, len(data.qpos), 3)] = 50

    mujoco.mj_forward(model, data)

    q0 = data.qpos.copy()  # at startup

    # Pre-compute invariant matrices
    print("Pre-computing invariant matrices...")
    invariants = precompute_invariants(model, data)

    # Keep your S call visible in this style
    S = compute_fixed_S_from_qbar(model, data, model.site("ee").id, task_dim=TASK_DIM, qbar=None)

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
    q_des = np.ones(data.qpos.shape[0]) * 0.2

    task_error_log = []
    q_vel = []
    q_pos = []
    time_log = []
    t = 0.0
    dt = model.opt.timestep

    max_sim_time = 50.0
    log_frequency = 5
    step_count = 0
    threshold  = 0.001

    if headless:
        print("Running headless simulation...")
        while (len(task_error_log) < 2 or
            abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold ):
            task_error, q, dq, previous_solution = controller(model, data, invariants, previous_solution, S)
            mujoco.mj_step(model, data)

            if step_count % log_frequency == 0:
                sim_ts["ts"].append(data.time)

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

                task_error_log.append(task_error)
                q_vel.append(dq.squeeze().copy())
                q_pos.append(q.squeeze().copy())
                time_log.append(t)

            step_count += 1
            t += dt
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and (
                len(task_error_log) < 2 or
                abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold 
            ):
                task_error, q, dq, previous_solution = controller(model, data, invariants, previous_solution, S)
                mujoco.mj_step(model, data)

                if step_count % log_frequency == 0:
                    print(data.time)

                    sim_ts["ts"].append(data.time)

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

                    task_error_log.append(task_error)
                    q_vel.append(dq.squeeze().copy())
                    q_pos.append(q.squeeze().copy())
                    time_log.append(t)

                step_count += 1
                viewer.sync()
                t += dt

    print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")
    return task_error_log, q_vel, q_pos, time_log, sim_ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized tendon control simulation")
    parser.add_argument("--headless", action="store_true", help="Run simulation without GUI for maximum performance")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots at the end")
    args = parser.parse_args()

    # Record start time for performance measurement
    start_time = time.time()

    # Simulate the model
    task_error_log, q_vel, q_pos, time_log, sim_ts = simulate_model(headless=args.headless)

    # Record end time
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds (wall clock time)")
    print(f"Simulated {sim_ts['ts'][-1]:.3f} seconds of physics time")
    print(f"Performance ratio: {sim_ts['ts'][-1] / (end_time - start_time):.2f}x real-time")

    if args.no_plots:
        print("Skipping plot generation")
        exit(0)

    q_vel = np.array(q_vel)
    q_pos = np.array(q_pos)

    # End-Effector Position Error – checks task-space convergence.
    plt.figure()
    plt.plot(time_log, task_error_log, label="‖x_desired - x_actual‖")
    plt.xlabel("Time (s)")
    plt.ylabel("Task-Space Position Error (m)")
    plt.title("End-Effector Position Error Over Time (Santina Fixed-S Projection)")
    plt.grid(True)
    plt.legend()

    plt.show()

