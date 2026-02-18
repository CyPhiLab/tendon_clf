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
import imageio
import csv
import pandas as pd

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


model_name = f"spirob_control"

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
stiffness = 0.01


f_ctrl = 2000.0 
T = 1
w = 2 * np.pi / T

# B matrix will be calculated from the MuJoCo model after loading
B = None

def calculate_input_matrix(model):
    """
    Calculate the input matrix B from the MuJoCo model.
    
    For MuJoCo, the actuator transmission matrix can be extracted from the model.
    This maps actuator forces/torques to joint torques.
    
    Parameters:
        model: MuJoCo model object
    
    Returns:
        B: Input matrix (nv x nu) mapping control inputs to joint torques
    """
    nv = model.nv  # number of velocity coordinates (DOFs)
    nu = model.nu  # number of actuators
    
    print(f"Model dimensions: nv={nv}, nu={nu}")
    
    if nu == 0:
        print("Warning: No actuators found in model. Creating default B matrix.")
        # If no actuators, create a simple identity-like mapping
        B = np.eye(nv, min(nv, 3))  # Assume 3 control inputs max
        return B
    
    # Get actuator transmission matrix from MuJoCo
    # This is stored in model.actuator_moment which gives the moment arm matrix
    B = np.zeros((nv, nu))
    
    # Extract the transmission matrix from MuJoCo's internal representation
    data_temp = mujoco.MjData(model)
    mujoco.mj_step1(model, data_temp)  # Initialize to get proper matrices
    
    # The transmission matrix is implicitly defined by how actuators affect joints
    # We can extract it by looking at the actuator moment arms
    for i in range(nu):
        # Create unit actuator force
        ctrl_test = np.zeros(nu)
        ctrl_test[i] = 1.0
        data_temp.ctrl[:] = ctrl_test
        
        # Compute actuator forces and map to joint torques
        mujoco.mj_forward(model, data_temp)
        B[:, i] = data_temp.qfrc_actuator.copy()
        
        # Reset
        data_temp.ctrl[:] = 0.0
    
    print(f"Calculated B matrix shape: {B.shape}")
    return B

def calculate_input_matrix_at_state(model, data):
    nv, nu = model.nv, model.nu
    B = np.zeros((nv, nu))

    data_temp = mujoco.MjData(model)
    data_temp.qpos[:] = data.qpos
    data_temp.qvel[:] = data.qvel  # optional; usually 0 is fine too

    mujoco.mj_forward(model, data_temp)

    for i in range(nu):
        data_temp.ctrl[:] = 0.0
        data_temp.ctrl[i] = 1.0
        mujoco.mj_forward(model, data_temp)
        B[:, i] = data_temp.qfrc_actuator.copy()

    return B


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
    e = 0.05
    Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))
    
    # Input matrix pseudoinverse
    pinv_B = np.linalg.pinv(B)

    # Selection matrix for control inputs
    nu = model.nu  # Use actual number of actuators from model
    sel = np.ones((nu, 1))  # All control inputs are active
    
    return {
        'Kp_null': Kp_null,
        'Kd_null': Kd_null,
        'F': F,
        'G': G,
        'Pe': Pe,
        'pinv_B': pinv_B,
        'sel': sel,
        'e': e
    }

def grad_wd(model, data, site_id, d):

    nv = model.nv
    grad = np.zeros(nv)

    # --- current Jacobian ---
    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)

    y = Jp.T @ d

    qvel0 = data.qvel.copy()

    for i in range(nv):
        # set unit joint velocity
        data.qvel[:] = 0.0
        data.qvel[i] = 1.0

        mujoco.mj_forward(model, data)

        # compute Jdot
        Jdot = compute_jacobian_derivative(model, data, site_id, h=1e-6) # you likely already have this
        Jdot = Jdot[:3, :]

        grad[i] = 2.0 * d.T @ (Jdot @ y)

    # restore
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    return grad

import numpy as np
import cvxpy as cp
import mujoco

# Assumes you already have:
# - Kori (float)
# - compute_jacobian_derivative(model, data, site_id)  -> (6,nv) or (6,nv) consistent with jac
# - calculate_input_matrix_at_state(model, data)       -> Bp (nv,nu)
# - grad_wd(model, data, site_id, d)                   -> (nv,) gradient of w_d wrt q
#
# Notes:
# - This rewrite fixes the big bug where you used qdd.value BEFORE solving
# - It also avoids overwriting cvxpy variable `u` with a numpy array
# - tau_align is treated as a CONSTANT joint torque computed at current state
#   and included consistently in the dynamics/actuator mapping constraint.

def controller(model, data, invariants, sigma=None, previous_solution=None):
    nv = model.nv
    nu = model.nu

    # ---------------------------------------------------------------------
    # Allocate working arrays
    # ---------------------------------------------------------------------
    jac = np.zeros((6, nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((nv, nv))
    M = np.zeros((nv, nv))

    # ---------------------------------------------------------------------
    # IDs
    # ---------------------------------------------------------------------
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    site_name = "ee"
    site_id = model.site(site_name).id

    # ---------------------------------------------------------------------
    # Invariants (from your CLF construction)
    # ---------------------------------------------------------------------
    F  = invariants["F"]
    G  = invariants["G"]
    Pe = invariants["Pe"]
    # pinv_B = invariants["pinv_B"]   # not used in this version (we use state-dependent Bp)
    e_clf = float(invariants.get("e", 1.0))  # avoid collision with error norm variable

    # ---------------------------------------------------------------------
    # Input matrix at current state (actuation mapping)
    # ---------------------------------------------------------------------
    Bp = calculate_input_matrix_at_state(model, data)        # (nv, nu)
    pinv_Bp = np.linalg.pinv(Bp)                             # (nu, nv)

    # ---------------------------------------------------------------------
    # Set target (your current choice)
    # ---------------------------------------------------------------------
    # data.mocap_pos[mocap_id] = np.array([0.15, 0.0, 0.2])
    # data.mocap_pos[mocap_id] = np.array([0.1, 0.0, 0.1])
    # data.mocap_pos[mocap_id] = np.array([0.2, 0.0, 0.2])
    data.mocap_pos[mocap_id] = np.array([0.16, 0.0, 0.35])
    # data.mocap_pos[mocap_id] = np.array([0.2, 0, 0.3])

    # ---------------------------------------------------------------------
    # Task-space error and twist command (position + orientation)
    # ---------------------------------------------------------------------
    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
    twist[:3] = dx

    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori

    # If you want ONLY position tracking for now:
    twist[3:] = 0.0

    # ---------------------------------------------------------------------
    # Kinematics + Jacobian
    # ---------------------------------------------------------------------
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)   # jac: (6,nv)

    # ---------------------------------------------------------------------
    # Dynamics matrices
    # ---------------------------------------------------------------------
    mujoco.mj_solveM(model, data, M_inv, np.eye(nv))
    mujoco.mj_fullM(model, M, data.qM)

    # Optional: inertia noise (if you truly want it)
    # M = M * np.random.normal(1.0, 0.01, size=M.shape)

    # Jdot (your helper)
    dJ_dt = compute_jacobian_derivative(model, data, site_id)   # shape must match jac

    # ---------------------------------------------------------------------
    # Bookkeeping
    # ---------------------------------------------------------------------
    task_error = float(np.linalg.norm(dx))
    q = data.qpos.copy()
    dq = data.qvel.reshape(-1, 1).copy()

    # Desired direction d for directional manipulability (use position direction)
    v = twist[:3]
    vnorm = np.linalg.norm(v)
    if vnorm > 1e-6:
        d_dir = v / vnorm
    else:
        d_dir = np.zeros_like(v)

    # Nullspace projector (based on current jacobian)
    # NOTE: pseudoinverse on 6xnv is OK; for better conditioning you might damp it.
    Jpinv = np.linalg.pinv(jac)
    N = np.eye(nv) - Jpinv @ jac   # (nv,nv)

    # ---------------------------------------------------------------------
    # Compute tau_align (CONSTANT joint torque at current state)
    # ---------------------------------------------------------------------
    # grad_wd should return shape (nv,) or (nv,1)
    g = grad_wd(model, data, site_id, d_dir).reshape(-1)
    g /= (np.linalg.norm(g) + 1e-8)

    k_a = 200.0
    tau_align = (k_a * (N.T @ g)).reshape(-1, 1)  # (nv,1)

    # ---------------------------------------------------------------------
    # CVXPY variables
    # ---------------------------------------------------------------------
    qdd = cp.Variable((nv, 1))
    u   = cp.Variable((nu, 1))
    dl  = cp.Variable((1, 1))

    # ---------------------------------------------------------------------
    # CLF terms
    # eta = [-twist; J dq]
    # V = eta^T Pe eta
    # dV = eta^T (F^T Pe + Pe F) eta + 2 eta^T Pe G (Jdot dq + J qdd)
    # ---------------------------------------------------------------------
    eta = np.concatenate((-twist.reshape(-1, 1), (jac @ dq)), axis=0)  # (12,1) if twist is 6 and jac@dq is 6
    V = float(eta.T @ Pe @ eta)

    # Expression for dV (CVXPY)
    Aclf = (F.T @ Pe + Pe @ F)
    dV_expr = (
        eta.T @ Aclf @ eta
        + 2.0 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd)
    )  # (1,1) expression

    # ---------------------------------------------------------------------
    # Tracking accel objective term (your original form)
    # ---------------------------------------------------------------------
    K = 500.0

    # Desired operational accel: K*twist - 2*sqrt(K)*J*dq (your original)
    # a_task = Jdot*dq + J*qdd
    a_task = dJ_dt @ dq + jac @ qdd
    a_des  = (K * twist.reshape(-1, 1) - 2.0 * np.sqrt(K) * (jac @ dq))

    # ---------------------------------------------------------------------
    # Constraints
    # Dynamics / actuation consistency WITH tau_align:
    #   M qdd + bias - passive = B u + tau_align
    # => B u = M qdd + bias - passive - tau_align
    # Using pinv_Bp as your actuator-space map (consistent with your previous style).
    # ---------------------------------------------------------------------
    bias = data.qfrc_bias.reshape(-1, 1)
    passive = data.qfrc_passive.reshape(-1, 1)

    constraints = [
        dV_expr <= (-1.0 / max(e_clf, 1e-8)) * V + dl,   # CLF constraint
        pinv_Bp @ (M @ qdd + bias - passive - tau_align) == u,
        u <= 0.0,          # enforce unilateral (negative-only) actuation
        u >= -200.0        # lower bound
    ]

    # ---------------------------------------------------------------------
    # Objective
    # ---------------------------------------------------------------------
    # NOTE: cp.norm(u,1) is fine; cp.square(cp.norm(u,1)) is convex but can be stiff.
    # Keeping your weighting structure.
    obj = cp.Minimize(
        1 * cp.square(cp.norm(a_task - a_des))
        + 0.05 * cp.square(cp.norm(qdd))
        + 0.5  * cp.norm1(u)
        + 1000.0 * cp.square(dl)
        # + 0.0 * cp.square(cp.norm(N @ qdd))   # if you want to penalize nullspace accel
    )

    prob = cp.Problem(obj, constraints)

    # Warm start
    if previous_solution is not None:
        try:
            if "u" in previous_solution and previous_solution["u"] is not None:
                u.value = previous_solution["u"]
            if "qdd" in previous_solution and previous_solution["qdd"] is not None:
                qdd.value = previous_solution["qdd"]
            if "dl" in previous_solution and previous_solution["dl"] is not None:
                dl.value = previous_solution["dl"]
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------
    try:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)

        if u.value is None:
            print("failed convergence - no u.value")
            return V, task_error, q, dq, previous_solution, None

        # Apply control
        u_cmd = np.squeeze(u.value).clip(-200.0, 0.0)
        data.ctrl[:] = u_cmd

        current_solution = {
            "u": u.value.copy(),
            "qdd": qdd.value.copy(),
            "dl": dl.value.copy() if dl.value is not None else None,
        }

        return V, task_error, q, dq, current_solution, u.value.copy()

    except Exception as ex:
        print(f"failed convergence - exception: {ex}")
        return V, task_error, q, dq, previous_solution, None


def simulate_model(headless=False, record_video=False, video_fps=30):
    global B  # Make B accessible globally
    
    model_path = Path("mujoco_models/spirob") / (str(model_name) + str(".xml"))
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    
    print(f"Model loaded: {model.nq} positions, {model.nv} velocities, {model.nu} actuators")
    
    # Calculate the input matrix B from the model
    B = calculate_input_matrix(model)

    
    # Set joint properties for soft robot
    model.jnt_stiffness[:] = 1
    model.dof_damping[:] = 0.1
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    # Initialize joint positions to neutral configuration
    # For the SpiRob, this should be a straight configuration
    data.qpos[:] = 0.0
    # model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Pre-compute invariant matrices
    print("Pre-computing invariant matrices...")
    invariants = precompute_invariants(model)
    
    # Initialize solution cache
    previous_solution = None

    # Video recording setup
    frames = []
    video_writer = None
    video_filename = None
    renderer = None
    camera = None
    
    if record_video:
        video_filename = f"spirob_simulation_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        print(f"Video recording enabled. Output file: {video_filename}")
        
        # Set up camera for a zoomed-in view of the robot using proper MuJoCo camera API
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 0.8  # Closer distance for zoom
        camera.lookat[:] = [0.2, 0.0, 0.4]  # Look at robot center
        camera.azimuth = 160  # Side angle view
        camera.elevation = -10  # Slightly above
        
        # Set up rendering for video capture
        if headless:
            # For headless rendering, we need to create a renderer with high quality dimensions
            renderer = mujoco.Renderer(model, height=1080, width=1920)
            renderer.update_scene(data, camera=camera)
        else:
            renderer = None  # Will use viewer's rendering

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
    # print(np.shape(q_des))

    V_log = []
    u_log = []
    task_error_log = []
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
    sigma = 10

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        video_frame_interval = max(1, int(1.0 / (video_fps * dt))) if record_video else float('inf')
        
        while (len(task_error_log) < 2 or
            abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold ):
            step_start = time.time()
            V, task_error, q, dq, previous_solution, u = controller(model, data, invariants, sigma, previous_solution)
            mujoco.mj_step(model, data)
            
            # Capture video frame if recording
            if record_video and step_count % video_frame_interval == 0:
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                frames.append(frame)
            
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
                q_vel.append(dq.squeeze().copy())
                q_pos.append(q.squeeze().copy())
                time_log.append(t)
                u_log.append(u.squeeze().copy())
            
            step_count += 1
            t += dt
    else:
        # Run simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_start = time.time()
            video_frame_interval = max(1, int(1.0 / (video_fps * dt))) if record_video else float('inf')
            
            if record_video:
                # Create renderer for video capture from viewer
                renderer = mujoco.Renderer(model, height=1080, width=1920)
            
            while viewer.is_running() and (
                len(task_error_log) < 2 or
                abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold 
            ):
                step_start = time.time()
                first_time = time.time()
                V, task_error, q, dq, previous_solution, u = controller(model, data, invariants, sigma, previous_solution)
                mujoco.mj_step(model, data)
                
                # Capture video frame if recording
                if record_video and step_count % video_frame_interval == 0:
                    renderer.update_scene(data, camera=camera)
                    frame = renderer.render()
                    frames.append(frame)
                
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
                    task_error_log.append(task_error)
                    q_vel.append(dq.squeeze().copy())
                    q_pos.append(q.squeeze().copy())
                    time_log.append(t)
                    u_log.append(u.squeeze().copy())
                
                step_count += 1
                
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                t += dt

    # Save video if recording was enabled
    if record_video and frames:
        print(f"Saving video with {len(frames)} frames...")
        try:
            with imageio.get_writer(video_filename, fps=video_fps, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Video saved successfully: {video_filename}")
        except Exception as e:
            print(f"Error saving video: {e}")
    
    print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")
    return V_log, task_error_log, q_vel, q_pos, time_log, sim_ts, u_log



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized tendon control simulation')
    parser.add_argument('--headless', action='store_true', help='Run simulation without GUI for maximum performance')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots at the end')
    parser.add_argument('--record-video', action='store_true', help='Record simulation video to MP4 file')
    parser.add_argument('--video-fps', type=int, default=30, help='Video frame rate (default: 30)')
    args = parser.parse_args()
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Simulate the model
    V_log, task_error_log, q_vel, q_pos, time_log, sim_ts, u_log = simulate_model(
        headless=args.headless, 
        record_video=args.record_video, 
        video_fps=args.video_fps
    )
    
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


    # Save control inputs and task error to CSV
    V_log = np.array(V_log)
    u_log = np.array(u_log)
    time_log_csv = np.array(time_log)
    error_log = np.array(task_error_log)
    df = pd.DataFrame(
        u_log,
        columns=[f"u{i}" for i in range(u_log.shape[1])]
    )
    df.insert(0, "time", time_log_csv)
    df.insert(1, "V", V_log)
    df.insert(2, "task_error", error_log)

    csv_path = "spirob_id_clf_qp_pos5.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    #Lyapunov Function Over Time – shows how stability evolves.
    plt.figure()
    plt.plot(time_log, V_log, label="Lyapunov Function V")
    plt.xlabel("Time (s)")
    plt.ylabel("Lyapunov Function V")
    plt.title("Lyapunov Function Over Time")
    plt.grid(True)
    plt.legend()  # <--- Add legend

    #End-Effector Position Error – checks task-space convergence.
    plt.figure()
    plt.plot(time_log, task_error_log, label="‖x_desired - x_actual‖")
    plt.xlabel("Time (s)")
    plt.ylabel("Task-Space Position Error (m)")
    plt.title("End-Effector Position Error Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

    # # Joint Angles (qq) 
    # actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
    # fig, axs = plt.subplots(len(actuated_indices), 1, figsize=(10, 8), sharex=True)

    # for i, idx in enumerate(actuated_indices):
    #     axs[i].plot(time_log, q_pos[:, idx], label=f"Joint {idx}")
    #     axs[i].legend()
    #     axs[i].grid(True)

    # axs[-1].set_xlabel("Time (s)")
    # fig.suptitle("Actuated Joint Angles Over Time (Rad)")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # # Joint Velocities (q̇) 
    # actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
    # fig, axs = plt.subplots(len(actuated_indices), 1, figsize=(10, 8), sharex=True)

    # for i, idx in enumerate(actuated_indices):
    #     axs[i].plot(time_log, q_vel[:, idx], label=f"Joint {idx}")
    #     axs[i].legend()
    #     axs[i].grid(True)

    # axs[-1].set_xlabel("Time (s)")
    # fig.suptitle("Actuated Joint Velocities Over Time (Rad/s)")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Control inputs plot
    actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
    ctrl = np.array(sim_ts["ctrl"])  # shape: (timesteps, nu)
    num_actuators = ctrl.shape[1]
    control_limit = 0.0

    fig, axs = plt.subplots(len(actuated_indices), 1, figsize=(10, 8), sharex=True)

    for i in range(len(actuated_indices)):
        axs[i].plot(time_log, ctrl[:, i], label=f"Actuator {i}")
        axs[i].axhline(control_limit, color='r', linestyle='--', linewidth=1)
        axs[i].axhline(-control_limit, color='r', linestyle='--', linewidth=1)
        axs[i].set_ylabel("Tension (N)")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Control Inputs at Actuated Joints")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
