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
import imageio
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

os.environ["MUJOCO_GL"] = "egl"

model_name = "spirob_control"

gear = 435.0

# Cartesian gains
Kpos = 0.95
Kori = 0.95

damping_ratio = 1.0

f_ctrl = 2000.0

# ============================================================
# CBF PARAMETERS
# ============================================================

# Minimum acceptable directional actuation authority.
#
# h(q) = phi(q) - phi_min
#
# where
#
# phi(q) = || A(q)^T d ||^2
#
# and
#
# A(q) = J M^{-1} B
#
PHI_MIN = 1e-4

# HOCBF gains
ALPHA1 = 10.0
ALPHA2 = 20.0

# Slack penalty
CBF_SLACK_WEIGHT = 5000.0

# Finite difference step used for derivatives of h(q)
GRAD_EPS = 1e-5
HESS_EPS = 5e-4

# Regularization for numerical pseudoinverses
PINV_RCOND = 1e-2

# Control limits
U_MIN = -1.0
U_MAX = 0.0

# CLF parameters
CLF_K = 500.0
CLF_EPS = 0.05
CLF_SLACK_WEIGHT = 100.0

# QP objective weights
W_TASK = 0.02
W_QDD = 0.02
W_U = 0.05

# Global input matrix
B = None


# ============================================================
# INPUT MATRIX
# ============================================================

def calculate_input_matrix(model):
    """
    Calculate the actuator-to-generalized-force matrix B.

    B maps actuator controls u to generalized actuator forces:

        tau_act = B u

    This assumes the actuator transmission is locally linear
    in ctrl, which is appropriate for the actuator model being used.
    """

    nv = model.nv
    nu = model.nu

    print(f"Model dimensions: nv={nv}, nu={nu}")

    if nu == 0:
        return np.zeros((nv, 0))

    B = np.zeros((nv, nu))

    data_temp = mujoco.MjData(model)

    data_temp.qpos[:] = 0.0
    data_temp.qvel[:] = 0.0

    mujoco.mj_forward(model, data_temp)

    for i in range(nu):

        data_temp.ctrl[:] = 0.0
        data_temp.ctrl[i] = 1.0

        mujoco.mj_forward(model, data_temp)

        B[:, i] = data_temp.qfrc_actuator.copy()

    data_temp.ctrl[:] = 0.0

    return B


def calculate_input_matrix_at_state(model, data):
    """
    Calculate local actuator mapping B(q).

    B[:,i] is the generalized force produced by unit actuator i.
    """

    nv = model.nv
    nu = model.nu

    Bp = np.zeros((nv, nu))

    data_temp = mujoco.MjData(model)

    data_temp.qpos[:] = data.qpos
    data_temp.qvel[:] = data.qvel

    mujoco.mj_forward(model, data_temp)

    for i in range(nu):

        data_temp.ctrl[:] = 0.0
        data_temp.ctrl[i] = 1.0

        mujoco.mj_forward(model, data_temp)

        Bp[:, i] = data_temp.qfrc_actuator.copy()

    return Bp


# ============================================================
# JACOBIAN
# ============================================================

def get_site_jacobian(model, data, site_id):
    """
    Return the 6 x nv spatial Jacobian.
    """

    J = np.zeros((6, model.nv))

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    mujoco.mj_jacSite(
        model,
        data,
        J[:3],
        J[3:],
        site_id
    )

    return J


def compute_jacobian_derivative(model, data, site_id, h=1e-6):
    """
    Numerically compute Jdot using qpos integration.
    """

    qpos_backup = data.qpos.copy()

    # Current Jacobian
    J0 = get_site_jacobian(model, data, site_id)

    # Integrate qpos forward
    qpos_temp = data.qpos.copy()

    mujoco.mj_integratePos(
        model,
        qpos_temp,
        data.qvel,
        h
    )

    data.qpos[:] = qpos_temp

    J1 = get_site_jacobian(model, data, site_id)

    # Restore
    data.qpos[:] = qpos_backup

    mujoco.mj_forward(model, data)

    return (J1 - J0) / h


# ============================================================
# MASS MATRIX
# ============================================================

def get_mass_matrix(model, data):
    """
    Return full generalized mass matrix M.
    """

    M = np.zeros((model.nv, model.nv))

    mujoco.mj_fullM(
        model,
        M,
        data.qM
    )

    return M


def get_mass_inverse(model, data):
    """
    Compute M^{-1}.
    """

    M = get_mass_matrix(model, data)

    return np.linalg.pinv(
        M,
        rcond=PINV_RCOND
    )


# ============================================================
# END-EFFECTOR TARGET DIRECTION
# ============================================================

def get_target_direction(model, data, site_id, mocap_id):
    """
    Compute

        d = (y_target - y) / ||y_target-y||

    Only translational direction is used.
    """

    dx = (
        data.mocap_pos[mocap_id]
        - data.site(site_id).xpos
    )

    norm_dx = np.linalg.norm(dx)

    if norm_dx < 1e-9:
        return np.zeros((3, 1)), dx

    d = (dx / norm_dx).reshape(3, 1)

    return d, dx


# ============================================================
# DIRECTIONAL ACTUATION AUTHORITY
# ============================================================

def compute_authority(
    model,
    data,
    site_id,
    mocap_id
):
    """
    Compute

        A(q) = J(q) M(q)^(-1) B(q)

    and

        phi(q) = || A(q)^T d ||^2

    where d is the direction from the end-effector to the target.

    phi measures how strongly the actuators can affect
    acceleration in the target direction.
    """

    # Jacobian
    J = get_site_jacobian(
        model,
        data,
        site_id
    )

    # Mass inverse
    M_inv = get_mass_inverse(
        model,
        data
    )

    # Local actuator map
    Bp = calculate_input_matrix_at_state(
        model,
        data
    )

    # Target direction
    d, dx = get_target_direction(
        model,
        data,
        site_id,
        mocap_id
    )

    # Translational actuation-to-acceleration map
    A = (
        J[:3, :]
        @ M_inv
        @ Bp
    )

    # Directional authority
    authority_vector = A.T @ d

    phi = np.dot(
        authority_vector.ravel(),
        authority_vector.ravel()
    ).item()

    return (
        phi,
        A,
        d,
        dx,
        J,
        M_inv,
        Bp
    )


# ============================================================
# AUTHORITY FUNCTION AT AN ARBITRARY CONFIGURATION
# ============================================================

def authority_from_qpos(
    model,
    data,
    qpos,
    site_id,
    mocap_id
):
    """
    Evaluate

        phi(q) = ||A(q)^T d(q)||^2

    at an arbitrary qpos.

    Used for numerical gradient/Hessian calculation.
    """

    temp = mujoco.MjData(model)

    temp.qpos[:] = qpos

    # Velocity is irrelevant for phi(q), but MuJoCo needs
    # a valid state for forward kinematics.
    temp.qvel[:] = data.qvel

    temp.mocap_pos[:] = data.mocap_pos
    temp.mocap_quat[:] = data.mocap_quat

    mujoco.mj_forward(
        model,
        temp
    )

    phi, _, _, _, _, _, _ = compute_authority(
        model,
        temp,
        site_id,
        mocap_id
    )

    return phi


# ============================================================
# NUMERICAL GRADIENT OF AUTHORITY
# ============================================================

def compute_authority_gradient(
    model,
    data,
    site_id,
    mocap_id,
    eps=GRAD_EPS
):
    """
    Compute

        grad_phi = d phi / d q

    using central finite differences.
    """

    nq = model.nq

    q0 = data.qpos.copy()

    grad = np.zeros(nq)

    for i in range(nq):

        q_plus = q0.copy()
        q_minus = q0.copy()

        q_plus[i] += eps
        q_minus[i] -= eps

        phi_plus = authority_from_qpos(
            model,
            data,
            q_plus,
            site_id,
            mocap_id
        )

        phi_minus = authority_from_qpos(
            model,
            data,
            q_minus,
            site_id,
            mocap_id
        )

        grad[i] = (
            phi_plus - phi_minus
        ) / (2.0 * eps)

    return grad


# ============================================================
# NUMERICAL HESSIAN OF AUTHORITY
# ============================================================

def compute_authority_hessian(
    model,
    data,
    site_id,
    mocap_id,
    eps=HESS_EPS
):
    """
    Compute

        Hess_phi = d^2 phi / dq^2

    using finite differences of the gradient.

    WARNING:
        This is expensive for high-dimensional continuum robots.
        It is intended as a reference implementation.
    """

    nq = model.nq

    q0 = data.qpos.copy()

    H = np.zeros((nq, nq))

    for i in range(nq):

        q_plus = q0.copy()
        q_minus = q0.copy()

        q_plus[i] += eps
        q_minus[i] -= eps

        temp_plus = mujoco.MjData(model)
        temp_minus = mujoco.MjData(model)

        temp_plus.qpos[:] = q_plus
        temp_minus.qpos[:] = q_minus

        temp_plus.qvel[:] = data.qvel
        temp_minus.qvel[:] = data.qvel

        temp_plus.mocap_pos[:] = data.mocap_pos
        temp_plus.mocap_quat[:] = data.mocap_quat

        temp_minus.mocap_pos[:] = data.mocap_pos
        temp_minus.mocap_quat[:] = data.mocap_quat

        mujoco.mj_forward(model, temp_plus)
        mujoco.mj_forward(model, temp_minus)

        grad_plus = compute_authority_gradient(
            model,
            temp_plus,
            site_id,
            mocap_id,
            eps=GRAD_EPS
        )

        grad_minus = compute_authority_gradient(
            model,
            temp_minus,
            site_id,
            mocap_id,
            eps=GRAD_EPS
        )

        H[:, i] = (
            grad_plus - grad_minus
        ) / (2.0 * eps)

    # Numerical symmetrization
    H = 0.5 * (H + H.T)

    return H


# ============================================================
# CLF MATRICES
# ============================================================

def precompute_invariants(model):

    Kp_null = np.ones(model.nv)

    Kd_null = (
        damping_ratio
        * 2.0
        * np.sqrt(Kp_null)
    )

    m = 6

    F = np.zeros((2 * m, 2 * m))

    F[:m, m:] = np.eye(m)

    G = np.zeros((2 * m, m))

    G[m:, :] = np.eye(m)

    e = CLF_EPS

    Q = np.eye(2 * m)

    R = np.eye(m)

    Pe = (
        linalg.block_diag(
            np.eye(m) / e,
            np.eye(m)
        ).T
        @ linalg.solve_continuous_are(
            F,
            G,
            Q,
            R
        )
        @ linalg.block_diag(
            np.eye(m) / e,
            np.eye(m)
        )
    )

    pinv_B = np.linalg.pinv(
        B,
        rcond=PINV_RCOND
    )

    nu = model.nu

    sel = np.ones((nu, 1))

    return {
        "Kp_null": Kp_null,
        "Kd_null": Kd_null,
        "F": F,
        "G": G,
        "Pe": Pe,
        "pinv_B": pinv_B,
        "sel": sel,
        "e": e
    }


# ============================================================
# CONTROLLER
# ============================================================

def controller(
    model,
    data,
    invariants,
    sigma,
    previous_solution=None,
    pos_noise_std=0.0,
    vel_noise_std=0.0,
    ctrl_noise_std=0.0
):

    # --------------------------------------------------------
    # Noisy observation
    # --------------------------------------------------------

    obs = mujoco.MjData(model)

    obs.qpos[:] = data.qpos
    obs.qvel[:] = data.qvel
    obs.ctrl[:] = data.ctrl

    obs.mocap_pos[:] = data.mocap_pos
    obs.mocap_quat[:] = data.mocap_quat

    if pos_noise_std > 0.0:
        obs.qpos[:] += np.random.normal(
            0.0,
            pos_noise_std,
            size=obs.qpos.shape
        )

    if vel_noise_std > 0.0:
        obs.qvel[:] += np.random.normal(
            0.0,
            vel_noise_std,
            size=obs.qvel.shape
        )

    mujoco.mj_forward(
        model,
        obs
    )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    site_name = "ee"
    site_id = model.site(site_name).id

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # --------------------------------------------------------
    # Kinematics
    # --------------------------------------------------------

    jac = get_site_jacobian(
        model,
        obs,
        site_id
    )

    dJ_dt = compute_jacobian_derivative(
        model,
        obs,
        site_id
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    M = get_mass_matrix(
        model,
        obs
    )

    M_inv = np.linalg.pinv(
        M,
        rcond=PINV_RCOND
    )

    # IMPORTANT:
    #
    # qfrc_bias = C(q,dq)dq + g(q)
    #
    # We do NOT need to separately construct C.
    #
    bias = obs.qfrc_bias.reshape(
        -1, 1
    )

    # --------------------------------------------------------
    # Local actuator mapping
    # --------------------------------------------------------

    Bp = calculate_input_matrix_at_state(
        model,
        obs
    )

    pinv_Bp = np.linalg.pinv(
        Bp,
        rcond=PINV_RCOND
    )

    Ip = Bp @ pinv_Bp

    I = np.eye(model.nv)

    # --------------------------------------------------------
    # Target error
    # --------------------------------------------------------

    dx = (
        obs.mocap_pos[mocap_id]
        - obs.site(site_id).xpos
    )

    dx_norm = np.linalg.norm(dx)

    if dx_norm > 1e-9:
        d = (
            dx / dx_norm
        ).reshape(3, 1)
    else:
        d = np.zeros((3, 1))

    # --------------------------------------------------------
    # Twist
    # --------------------------------------------------------

    twist = np.zeros(6)

    twist[:3] = dx

    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    mujoco.mju_mat2Quat(
        site_quat,
        obs.site(site_id).xmat
    )

    mujoco.mju_negQuat(
        site_quat_conj,
        site_quat
    )

    mujoco.mju_mulQuat(
        error_quat,
        obs.mocap_quat[mocap_id],
        site_quat_conj
    )

    mujoco.mju_quat2Vel(
        twist[3:],
        error_quat,
        1.0
    )

    twist[3:] *= Kori

    # Only position control
    twist[3:] = 0.0

    dq = obs.qvel.reshape(
        -1, 1
    )

    # --------------------------------------------------------
    # CVXPY variables
    # --------------------------------------------------------

    nu = model.nu

    u = cp.Variable(
        (nu, 1)
    )

    qdd = cp.Variable(
        (model.nv, 1)
    )

    dclf = cp.Variable(
        (1, 1),
        nonneg=True
    )

    dcbf = cp.Variable(
        (1, 1),
        nonneg=True
    )

    # --------------------------------------------------------
    # Task acceleration
    # --------------------------------------------------------

    ydd = (
        jac @ qdd
        + dJ_dt @ dq
    )

    # --------------------------------------------------------
    # Operational-space inverse dynamics
    # --------------------------------------------------------

    Mx_inv = (
        jac
        @ M_inv
        @ jac.T
    )

    if abs(np.linalg.det(Mx_inv)) >= 1e-2:

        Mx = np.linalg.inv(
            Mx_inv
        )

    else:

        Mx = np.linalg.pinv(
            Mx_inv,
            rcond=PINV_RCOND
        )

    Jbar = (
        M_inv
        @ jac.T
        @ Mx
    )

    N = (
        I
        - jac.T
        @ Jbar.T
    )

    # --------------------------------------------------------
    # Null-space torque
    # --------------------------------------------------------

    tau_task = (
        jac.T
        @ Mx
        @ (
            ydd
            - dJ_dt @ dq
        )
        + jac.T
        @ Jbar.T
        @ (
            obs.qfrc_bias.reshape(-1, 1)
            - obs.qfrc_passive.reshape(-1, 1)
        )
    )

    tau_null = (
        np.linalg.pinv(
            (Ip - I) @ N,
            rcond=PINV_RCOND
        )
        @ (I - Ip)
        @ tau_task
    )

    tau = (
        tau_task
        + N @ tau_null
    )

    # --------------------------------------------------------
    # CLF
    # --------------------------------------------------------

    F = invariants["F"]
    G = invariants["G"]
    Pe = invariants["Pe"]
    e = invariants["e"]

    eta = np.concatenate(
        (
            -twist,
            (jac @ obs.qvel).reshape(-1)
        )
    ).reshape(-1, 1)

    V = (
        eta.T
        @ Pe
        @ eta
    )

    dV = (
        eta.T
        @ (
            F.T @ Pe
            + Pe @ F
        )
        @ eta
        +
        2.0
        * eta.T
        @ Pe
        @ G
        @ (
            dJ_dt @ dq
            + jac @ qdd
        )
    )

    # ========================================================
    # CORRECT CBF
    # ========================================================
    #
    # A(q) = J M^{-1} B
    #
    # phi(q) = || A(q)^T d ||^2
    #
    # h(q) = phi(q) - phi_min
    #
    # Since h depends on q only:
    #
    # hdot = grad_h^T qdot
    #
    # hddot =
    #       grad_h^T qdd
    #       + qdot^T Hess_h qdot
    #
    # HOCBF:
    #
    # hddot + alpha1 hdot + alpha2 h >= -delta
    #
    # ========================================================

    phi, A_authority, _, _, _, _, _ = compute_authority(
        model,
        obs,
        site_id,
        mocap_id
    )

    h = phi - PHI_MIN

    # --------------------------------------------------------
    # Numerical gradient
    # --------------------------------------------------------

    grad_phi = compute_authority_gradient(
        model,
        obs,
        site_id,
        mocap_id,
        eps=GRAD_EPS
    )

    grad_phi = grad_phi.reshape(
        -1, 1
    )

    # --------------------------------------------------------
    # Numerical Hessian
    # --------------------------------------------------------

    Hess_phi = compute_authority_hessian(
        model,
        obs,
        site_id,
        mocap_id,
        eps=HESS_EPS
    )

    # --------------------------------------------------------
    # HOCBF terms
    # --------------------------------------------------------

    hdot = (
        grad_phi.T
        @ dq
    )

    hddot = (
        grad_phi.T
        @ qdd
        +
        dq.T
        @ Hess_phi
        @ dq
    )

    cbf_constraint = (
        hddot
        + ALPHA1 * hdot
        + ALPHA2 * h
    )

    # --------------------------------------------------------
    # PD acceleration reference
    # --------------------------------------------------------

    ydd_des = (
        CLF_K * twist.reshape(-1, 1)
        - 2.0
        * np.sqrt(CLF_K)
        * (jac @ dq)
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = cp.Minimize(

        # Main task acceleration tracking
        W_TASK
        * cp.sum_squares(
            ydd - ydd_des
        )

        +

        # Regularize qdd
        W_QDD
        * cp.sum_squares(
            qdd
        )

        +

        # Regularize actuator effort
        W_U
        * cp.sum_squares(
            u
        )

        +

        # CLF relaxation
        CLF_SLACK_WEIGHT
        * cp.sum_squares(
            dclf
        )

        +

        # CBF relaxation
        CBF_SLACK_WEIGHT
        * cp.sum_squares(
            dcbf
        )
    )

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    constraints = [

        # CLF
        dV
        <=
        -1.0 / e * V
        + dclf,

        # Map operational-space inverse-dynamics torque
        # to actuator control
        pinv_Bp @ tau
        == u,

        # Actuator limits
        u >= U_MIN,
        u <= U_MAX,

        # CBF
        cbf_constraint
        >=
        -dcbf,
    ]

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    prob = cp.Problem(
        objective,
        constraints
    )

    # Warm start
    if previous_solution is not None:

        try:

            u.value = previous_solution["u"]
            qdd.value = previous_solution["qdd"]
            dclf.value = previous_solution["dclf"]
            dcbf.value = previous_solution["dcbf"]

        except Exception:
            pass

    task_error = np.linalg.norm(dx)

    q_return = data.qpos.copy()
    dq_return = data.qvel.copy()

    try:

        prob.solve(
            solver=cp.SCS,
            verbose=False,
            warm_start=True
        )

        if u.value is None:

            print("QP failed: no solution")

            return (
                V,
                task_error,
                q_return,
                dq_return,
                previous_solution,
                None
            )

        # ----------------------------------------------------
        # Apply control
        # ----------------------------------------------------

        u_command = np.squeeze(
            u.value
        )

        if ctrl_noise_std > 0.0:

            u_command += np.random.normal(
                0.01,
                ctrl_noise_std,
                size=model.nu
            )

        u_command = np.clip(
            u_command,
            U_MIN,
            U_MAX
        )

        data.ctrl[:] = u_command

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        current_solution = {

            "u":
                u.value.copy(),

            "qdd":
                qdd.value.copy(),

            "dclf":
                dclf.value.copy(),

            "dcbf":
                dcbf.value.copy(),

            "phi":
                phi,

            "h":
                h,

            "hdot":
                float(hdot),

            "cbf_value":
                float(
                    cbf_constraint.value
                )
                if cbf_constraint.value is not None
                else np.nan,
        }

        return (
            V,
            task_error,
            q_return,
            dq_return,
            current_solution,
            u.value.copy()
        )

    except Exception as exc:

        print(
            f"QP exception: {exc}"
        )

        return (
            V,
            task_error,
            q_return,
            dq_return,
            previous_solution,
            None
        )


# ============================================================
# SIMULATION
# ============================================================

def simulate_model(
    headless=False,
    record_video=False,
    video_fps=30,
    pos_noise_std=0.0,
    vel_noise_std=0.0,
    ctrl_noise_std=0.0
):

    global B

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_path = (
        Path("mujoco_models/spirob")
        / f"{model_name}.xml"
    )

    model = mujoco.MjModel.from_xml_path(
        str(model_path.absolute())
    )

    data = mujoco.MjData(model)

    print(
        f"Model loaded: "
        f"{model.nq} positions, "
        f"{model.nv} velocities, "
        f"{model.nu} actuators"
    )

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------

    model.jnt_stiffness[:] = 0.3
    model.dof_damping[:] = 0.1

    model.opt.gravity = (
        0,
        0,
        -9.81
    )

    # --------------------------------------------------------
    # Initial configuration
    # --------------------------------------------------------

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data
    )

    # --------------------------------------------------------
    # Input matrix
    # --------------------------------------------------------

    print("Calculating input matrix B...")

    B = calculate_input_matrix(
        model
    )

    # --------------------------------------------------------
    # Invariants
    # --------------------------------------------------------

    print(
        "Pre-computing controller invariants..."
    )

    invariants = precompute_invariants(
        model
    )

    previous_solution = None

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    frames = []

    renderer = None
    camera = None

    if record_video:

        video_filename = (
            f"spirob_cbf_"
            f"{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        camera = mujoco.MjvCamera()

        mujoco.mjv_defaultCamera(
            camera
        )

        camera.distance = 0.8
        camera.lookat[:] = [
            0.2,
            0.0,
            0.2
        ]

        camera.azimuth = 90
        camera.elevation = 0

        renderer = mujoco.Renderer(
            model,
            height=1080,
            width=1920
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    sim_ts = {
        "ts": [],
        "q": [],
        "qvel": [],
        "ctrl": [],
        "actuator_force": [],
        "task_error": [],
        "V": [],
        "phi": [],
        "h": [],
        "hdot": [],
        "cbf_value": [],
        "dcbf": [],
    }

    V_log = []
    task_error_log = []
    q_vel = []
    q_pos = []
    time_log = []
    u_log = []

    t = 0.0

    dt = model.opt.timestep

    max_sim_time = 125.0

    log_frequency = 5

    step_count = 0

    # --------------------------------------------------------
    # Simulation loop helper
    # --------------------------------------------------------

    def run_step():

        nonlocal previous_solution
        nonlocal step_count
        nonlocal t

        V, task_error, q, dq, solution, u = controller(
            model,
            data,
            invariants,
            sigma=10,
            previous_solution=previous_solution,
            pos_noise_std=pos_noise_std,
            vel_noise_std=vel_noise_std,
            ctrl_noise_std=ctrl_noise_std
        )

        previous_solution = solution

        mujoco.mj_step(
            model,
            data
        )

        if (
            step_count
            % log_frequency
            == 0
        ):

            sim_ts["ts"].append(
                data.time
            )

            sim_ts["q"].append(
                data.qpos.copy()
            )

            sim_ts["qvel"].append(
                data.qvel.copy()
            )

            sim_ts["ctrl"].append(
                data.ctrl.copy()
            )

            sim_ts["actuator_force"].append(
                data.actuator_force.copy()
            )

            sim_ts["task_error"].append(
                task_error
            )

            sim_ts["V"].append(
                float(V)
            )

            if solution is not None:

                sim_ts["phi"].append(
                    solution.get(
                        "phi",
                        np.nan
                    )
                )

                sim_ts["h"].append(
                    solution.get(
                        "h",
                        np.nan
                    )
                )

                sim_ts["hdot"].append(
                    solution.get(
                        "hdot",
                        np.nan
                    )
                )

                sim_ts["cbf_value"].append(
                    solution.get(
                        "cbf_value",
                        np.nan
                    )
                )

                dcbf_value = solution.get(
                    "dcbf",
                    np.array([[np.nan]])
                )

                sim_ts["dcbf"].append(
                    float(
                        np.squeeze(
                            dcbf_value
                        )
                    )
                )

            else:

                sim_ts["phi"].append(
                    np.nan
                )

                sim_ts["h"].append(
                    np.nan
                )

                sim_ts["hdot"].append(
                    np.nan
                )

                sim_ts["cbf_value"].append(
                    np.nan
                )

                sim_ts["dcbf"].append(
                    np.nan
                )

            V_log.append(
                float(V)
            )

            task_error_log.append(
                task_error
            )

            q_vel.append(
                dq.copy()
            )

            q_pos.append(
                q.copy()
            )

            time_log.append(
                t
            )

            if u is not None:

                u_log.append(
                    np.squeeze(u)
                )

            else:

                u_log.append(
                    np.zeros(model.nu)
                )

        step_count += 1

        t += dt

    # --------------------------------------------------------
    # Headless
    # --------------------------------------------------------

    if headless:

        print(
            "Running headless simulation..."
        )

        while data.time < max_sim_time:

            run_step()

            if record_video:

                frame_interval = max(
                    1,
                    int(
                        1.0
                        /
                        (
                            video_fps
                            * dt
                        )
                    )
                )

                if (
                    step_count
                    % frame_interval
                    == 0
                ):

                    renderer.update_scene(
                        data,
                        camera=camera
                    )

                    frames.append(
                        renderer.render()
                    )

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    else:

        with mujoco.viewer.launch_passive(
            model,
            data
        ) as viewer:

            while (
                viewer.is_running()
                and data.time < max_sim_time
            ):

                run_step()

                viewer.sync()

    # --------------------------------------------------------
    # Save video
    # --------------------------------------------------------

    if (
        record_video
        and len(frames) > 0
    ):

        print(
            f"Saving {len(frames)} frames..."
        )

        with imageio.get_writer(
            video_filename,
            fps=video_fps,
            codec="libx264"
        ) as writer:

            for frame in frames:

                writer.append_data(
                    frame
                )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return (
        V_log,
        task_error_log,
        q_vel,
        q_pos,
        time_log,
        sim_ts,
        u_log
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "SpiRob CLF-QP with directional-authority CBF"
    )

    parser.add_argument(
        "--headless",
        action="store_true"
    )

    parser.add_argument(
        "--no-plots",
        action="store_true"
    )

    parser.add_argument(
        "--record-video",
        action="store_true"
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=30
    )

    parser.add_argument(
        "--pos-noise-std",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--vel-noise-std",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--ctrl-noise-std",
        type=float,
        default=0.0
    )

    args = parser.parse_args()

    start_time = time.time()

    (
        V_log,
        task_error_log,
        q_vel,
        q_pos,
        time_log,
        sim_ts,
        u_log
    ) = simulate_model(
        headless=args.headless,
        record_video=args.record_video,
        video_fps=args.video_fps,
        pos_noise_std=args.pos_noise_std,
        vel_noise_std=args.vel_noise_std,
        ctrl_noise_std=args.ctrl_noise_std
    )

    end_time = time.time()

    print(
        f"Simulation completed in "
        f"{end_time - start_time:.2f} s"
    )

    if len(sim_ts["ts"]) == 0:
        print("No simulation data logged.")
        exit()

    print(
        f"Simulated "
        f"{sim_ts['ts'][-1]:.3f} s"
    )

    print(
        f"Performance ratio: "
        f"{sim_ts['ts'][-1] / max(end_time - start_time, 1e-9):.2f}x"
    )

    if args.no_plots:
        exit(0)

    # ========================================================
    # Convert logs
    # ========================================================

    V_log = np.asarray(
        V_log
    )

    task_error_log = np.asarray(
        task_error_log
    )

    q_vel = np.asarray(
        q_vel
    )

    q_pos = np.asarray(
        q_pos
    )

    time_log = np.asarray(
        time_log
    )

    u_log = np.asarray(
        u_log
    )

    # ========================================================
    # CSV
    # ========================================================

    df = pd.DataFrame(
        u_log,
        columns=[
            f"u{i}"
            for i in range(
                u_log.shape[1]
            )
        ]
    )

    df.insert(
        0,
        "time",
        time_log
    )

    df.insert(
        1,
        "V",
        V_log
    )

    df.insert(
        2,
        "task_error",
        task_error_log
    )

    df.insert(
        3,
        "phi",
        np.asarray(
            sim_ts["phi"]
        )
    )

    df.insert(
        4,
        "h",
        np.asarray(
            sim_ts["h"]
        )
    )

    df.insert(
        5,
        "hdot",
        np.asarray(
            sim_ts["hdot"]
        )
    )

    df.insert(
        6,
        "cbf_value",
        np.asarray(
            sim_ts["cbf_value"]
        )
    )

    df.insert(
        7,
        "dcbf",
        np.asarray(
            sim_ts["dcbf"]
        )
    )

    csv_path = (
        "spirob_authority_cbf.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Saved CSV to {csv_path}"
    )

    # ========================================================
    # Lyapunov function
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        V_log,
        label="Lyapunov Function V"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "V"
    )

    plt.title(
        "Lyapunov Function"
    )

    plt.grid(True)
    plt.legend()

    # ========================================================
    # Task error
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        task_error_log,
        label="Task-space error"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Position Error (m)"
    )

    plt.title(
        "End-Effector Position Error"
    )

    plt.grid(True)
    plt.legend()

    # ========================================================
    # Directional authority
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        sim_ts["phi"],
        label=r"$\phi(q)=\|A^Td\|^2$"
    )

    plt.axhline(
        PHI_MIN,
        linestyle="--",
        label=r"$\phi_{\min}$"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Directional Authority"
    )

    plt.title(
        "Target-Directional Actuation Authority"
    )

    plt.grid(True)
    plt.legend()

    # ========================================================
    # Barrier function
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        sim_ts["h"],
        label=r"$h(q)$"
    )

    plt.axhline(
        0.0,
        linestyle="--",
        label="CBF boundary"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "h"
    )

    plt.title(
        "Directional Authority CBF"
    )

    plt.grid(True)
    plt.legend()

    # ========================================================
    # CBF constraint
    # ========================================================

    plt.figure()

    plt.plot(
        time_log,
        sim_ts["cbf_value"],
        label="HOCBF LHS"
    )

    plt.axhline(
        0.0,
        linestyle="--",
        label="Constraint boundary"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        r"$\ddot h+\alpha_1\dot h+\alpha_2h$"
    )

    plt.title(
        "HOCBF Constraint"
    )

    plt.grid(True)
    plt.legend()

    # ========================================================
    # Control
    # ========================================================

    ctrl = np.asarray(
        sim_ts["ctrl"]
    )

    fig, axs = plt.subplots(
        ctrl.shape[1],
        1,
        figsize=(10, 8),
        sharex=True
    )

    if ctrl.shape[1] == 1:
        axs = [axs]

    for i in range(
        ctrl.shape[1]
    ):

        axs[i].plot(
            time_log,
            ctrl[:, i],
            label=f"Actuator {i}"
        )

        axs[i].axhline(
            U_MIN,
            linestyle="--"
        )

        axs[i].axhline(
            U_MAX,
            linestyle="--"
        )

        axs[i].set_ylabel(
            f"u{i}"
        )

        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        "Control Inputs"
    )

    fig.tight_layout()

    plt.show()