import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# Configure MuJoCo to use EGL (GPU) if available
os.environ.setdefault("MUJOCO_GL", "egl")


# ----------------------------
# User parameters
# ----------------------------
MODEL_NAME = "helix_control"
MOCAP_BODY_NAME = "target"
EE_SITE_NAME = "ee"

# Task dimension: set to 3 (translation) or 6 (translation+orientation)
TASK_DIM = 3  # 3 matches Santina's 3D tip-position example (m=3)

# "Impedance" gains in task space (acceleration-level form)
KP_TASK = 500.0
KD_TASK = 20.0

# Twist scaling for pose error -> desired twist
KPOS = 1.0
KORI = 0.95

# Pseudoinverse and solve damping
DET_THRESH = 1e-10
PINV_RCOND = 1e-8

# Numerical damping for solving in synergy space
SOLVE_LAM = 1e-6  # used in least-squares (A^T A + lam I)

# Rank / conditioning checks
SVD_TOL = 1e-9
WARN_IF_RANK_DEFICIENT = True

# Joint stiffness/damping config (your original choices)
JOINT_STIFFNESS = 0.01
DOF_DAMPING = 0.2
GRAVITY = (0, 0, -9.81)

MAX_SIM_TIME = 25.0
LOG_EVERY_N_STEPS = 5


def build_B():
    B = np.zeros((36, 9))
    for i in range(3):      # 3 blocks
        for j in range(4):  # repeat each block 4 times
            row_start = i * 12 + j * 3
            col_start = i * 3
            B[row_start:row_start + 3, col_start:col_start + 3] = np.eye(3)
    return B


def inv_or_pinv(A: np.ndarray, det_thresh: float = DET_THRESH, rcond: float = PINV_RCOND):
    if A.shape[0] == A.shape[1]:
        detA = np.linalg.det(A)
        if np.isfinite(detA) and abs(detA) >= det_thresh:
            return np.linalg.inv(A)
    return np.linalg.pinv(A, rcond=rcond)


def get_coriolis_and_gravity(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Approximate C(q,qdot) and g(q) via RNE + finite differences on qvel.
    Preserved from your original code (expensive but consistent).
    """
    nv = model.nv
    dummy = np.zeros(nv)

    # Ensure derived quantities are current
    mujoco.mj_forward(model, data)

    # Bias at current qvel -> includes C(q,qd)qd + g(q) + passive internal terms
    mujoco.mj_rne(model, data, 0, dummy)
    g = data.qfrc_bias.copy()

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

    data.qvel[:] = q_vel_orig
    mujoco.mj_forward(model, data)
    return C, g


def compute_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int):
    """Return 6xnv Jacobian for the site."""
    mujoco.mj_forward(model, data)
    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)
    return J


def compute_jacobian_derivative(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, h: float = 1e-6):
    """
    Jdot via finite differences:
      Jdot ≈ (J(q + h*qdot) - J(q)) / h
    FIX: we must call mj_forward after changing qpos, and restore state cleanly.
    """
    mujoco.mj_forward(model, data)
    J0 = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J0[:3], J0[3:], site_id)

    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()

    mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
    mujoco.mj_forward(model, data)

    Jh = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, Jh[:3], Jh[3:], site_id)

    # Restore
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)

    return (Jh - J0) / h


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

    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)
    Jm = J[:task_dim, :]          # (m x nv)

    S = Jm.T.copy()               # (nv x m), fixed

    # Restore
    mujoco.mj_setState(model, data, backup, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    return S


def check_rank_condition(Jm, Minv, S, m, tol=SVD_TOL):
    A = Jm @ Minv @ S
    svals = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(svals > tol))
    return rank, svals


# ----------------------------
# Santina synergy projection (Eq. 18 form)
# ----------------------------
def santina_project_tau(Jm: np.ndarray, Minv: np.ndarray, S: np.ndarray, f_task: np.ndarray, m: int):
    """
    Paper projector form:
      tau = S (J Minv S)^(-1) J Minv J^T f

    We solve sigma robustly with damped least squares:
      sigma = argmin ||A sigma - b||^2 + lam ||sigma||^2
      => (A^T A + lam I) sigma = A^T b
    """
    A = Jm @ Minv @ S                  # (m x m)
    b = Jm @ Minv @ (Jm.T @ f_task)    # (m,)

    # Damped least squares solve (robust)
    sigma = np.linalg.solve(A.T @ A + SOLVE_LAM * np.eye(m), A.T @ b)  # (m,)
    tau = S @ sigma  # (nv,)
    return tau


# ----------------------------
# Controller setup / invariants
# ----------------------------
def precompute_invariants(model: mujoco.MjModel, data: mujoco.MjData, task_dim: int):
    B = build_B()
    pinv_B = np.linalg.pinv(B)

    # IDs
    site_id = model.site(EE_SITE_NAME).id
    mocap_id = model.body(MOCAP_BODY_NAME).mocapid[0]

    # FIX: compute S ONCE and keep it fixed (do NOT overwrite later)
    S = compute_fixed_S_from_qbar(model, data, site_id, task_dim, qbar=None)

    return {
        "B": B,
        "pinv_B": pinv_B,
        "site_id": site_id,
        "mocap_id": mocap_id,
        "S": S,
        "task_dim": task_dim,
    }


# ----------------------------
# Santina-style controller (operational-space impedance + synergy projection)
# ----------------------------
def controller_step(model: mujoco.MjModel, data: mujoco.MjData, inv: dict):
    nv = model.nv
    site_id = inv["site_id"]
    mocap_id = inv["mocap_id"]
    S = inv["S"]
    m = inv["task_dim"]
    B = inv["B"]
    pinv_B = inv["pinv_B"]

    # Ensure forward kinematics are current
    mujoco.mj_forward(model, data)

    # --- target position (you can replace with trajectory later)
    data.mocap_pos[mocap_id] = ([-0.0553837,   0.05965076,  0.45050877])

    # data.mocap_pos[mocap_id] = np.array([0.23312815, -0.31631513, 0.44791383])
    # data.mocap_pos[mocap_id] = np.array([0.27083678, 0.00194196, 0.58488434])

    # --- Jacobian
    jac = np.zeros((6, nv))
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # --- error -> twist (pose error)
    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
    twist = np.zeros(6)
    twist[:3] = KPOS * dx

    # orientation error (only used if m=6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= KORI

    if m == 3:
        twist[3:] = 0.0

    # --- Minv
    Minv = np.zeros((nv, nv))
    mujoco.mj_solveM(model, data, Minv, np.eye(nv))

    # --- Jdot
    Jdot = compute_jacobian_derivative(model, data, site_id)

    # --- Select task Jacobian
    if m == 3:
        Jm = jac[:3, :]
        Jdot_m = Jdot[:3, :]
        # acceleration-level impedance in task space
        ydd = KP_TASK * twist[:3] - KD_TASK * (Jm @ data.qvel)
    elif m == 6:
        Jm = jac
        Jdot_m = Jdot
        ydd = KP_TASK * twist - KD_TASK * (Jm @ data.qvel)
    else:
        raise ValueError("TASK_DIM must be 3 or 6")

    # --- Lambda
    Lambda_inv = Jm @ Minv @ Jm.T
    Lambda = inv_or_pinv(Lambda_inv)

    # --- Jbar (dynamically consistent inverse)
    Jbar = Minv @ Jm.T @ Lambda  # (nv x m)

    # --- C and g (expensive but consistent with your approach)
    C, g = get_coriolis_and_gravity(model, data)

    # --- Operational-space bias term
    Cy = (Jbar.T @ C @ data.qvel) - (Lambda @ (Jdot_m @ data.qvel))

    # --- Task-space command "force"
    f_task = Lambda @ ydd + Cy

    # --- Rank check (paper condition: rank(J Minv S) = m)
    if WARN_IF_RANK_DEFICIENT:
        r, svals = check_rank_condition(Jm, Minv, S, m)
        if r < m:
            print(f"[WARN] rank(J Minv S)={r} < {m}. svals={np.array2string(svals, precision=2)}")

    # --- Santina projector
    tau_synergy = santina_project_tau(Jm, Minv, S, f_task, m=m)

    # Add gravity + passive
    tau_cmd = tau_synergy + g + data.qfrc_passive

    # Map to actuators (keep your pipeline)
    # If dimensions mismatch here, your B mapping doesn't match model.nu.
    u = (B @ (pinv_B @ tau_cmd)).reshape(-1)
    data.ctrl[:] = u[: model.nu]  # guard in case model.nu < len(u)

    task_error = float(np.linalg.norm(dx))
    return task_error


# ----------------------------
# Simulation driver
# ----------------------------
def simulate(headless: bool, no_plots: bool):
    model_path = Path("mujoco_models/helix") / f"{MODEL_NAME}.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    data = mujoco.MjData(model)

    # model params
    model.jnt_stiffness[:] = JOINT_STIFFNESS
    model.dof_damping[:] = DOF_DAMPING
    model.opt.gravity = GRAVITY

    # init tweaks
    if data.qpos.shape[0] > 2:
        data.qpos[2] = 0.0

    mujoco.mj_forward(model, data)

    inv = precompute_invariants(model, data, TASK_DIM)

    task_error_log = []
    time_log = []
    step_count = 0

    def do_step():
        nonlocal step_count
        err = controller_step(model, data, inv)
        mujoco.mj_step(model, data)
        if step_count % LOG_EVERY_N_STEPS == 0:
            task_error_log.append(err)
            time_log.append(data.time)
        step_count += 1

    start = time.time()

    if headless:
        while data.time < MAX_SIM_TIME:
            do_step()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < MAX_SIM_TIME:
                do_step()
                viewer.sync()

    end = time.time()
    print(f"Simulated {data.time:.3f}s in {end-start:.2f}s wall time.")
    if time_log:
        print(f"Final task error: {task_error_log[-1]:.6f} m")

    if (not no_plots) and time_log:
        plt.figure()
        plt.plot(time_log, task_error_log, label="‖x_des - x‖")
        plt.xlabel("Time (s)")
        plt.ylabel("Task position error (m)")
        plt.title("Task-Space Position Error (Fixed-S Santina Projection)")
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    simulate(headless=args.headless, no_plots=args.no_plots)


if __name__ == "__main__":
    main()
