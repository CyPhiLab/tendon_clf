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

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


model_name = f"scene"

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

def precompute_invariants(model):
    """Pre-compute matrices that don't change during simulation"""

    # ---------- nullspace gains (unchanged) ----------
    Kp_null = np.asarray([1] * model.nv)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # ---------- wrapping task setup ----------
    # 5 points total: end-effector + 4 other body sites
    # IMPORTANT: replace these with your actual site names you want to control.
    # site_names = ["ee", "150", "148", "146"]   # <-- EDIT THESE
    # site_names = ["ee", "149"]   # <-- EDIT THESE
    # site_names = ["ee", "148"]   # <-- EDIT THESE
    # site_names = ["ee", "150", "149", "148"]   # <-- EDIT THESE
    # site_names = ["ee"]   # <-- EDIT THESE
    # site_names = ["148"]   # <-- EDIT THESE
    site_names = ["ee", "150", "149","148", "147","146"]   # <-- EDIT THESE
    # site_names = ["ee", "150", "149","148", "147","146","145","144","143","142","141","140",
                #   "139"]   # <-- EDIT THESE
    # site_names = ["ee", "140"]   # <-- EDIT THESE
    # site_names = ["ee",  "150"]   # <-- EDIT THESE
    # site_names = ["148", "147","146"]   # <-- EDIT THESE

    n_sites = len(site_names)
    m = 3 * n_sites  # position-only task dimension (x,y,z for each site)

    # ---------- CLF matrices for stacked double integrators ----------
    # eta = [ -e ; v ]  with e in R^m, v in R^m
    # e_dot = v
    # v_dot = mu  (mu = Jdot*qdot + J*qdd)
    F = np.zeros((2*m, 2*m))
    F[:m, m:] = np.eye(m)

    G = np.zeros((2*m, m))
    G[m:, :] = np.eye(m)

    # CLF shaping (same idea as your original, but now dimensioned for 5 sites)
    e_scale = 0.2
    Q = np.eye(2*m)
    R = np.eye(m)
    P0 = linalg.solve_continuous_are(F, G, Q, R)
    S = linalg.block_diag(np.eye(m) / e_scale, np.eye(m))
    Pe = S.T @ P0 @ S

    # Input matrix pseudoinverse (optional; you may not need it anymore)
    pinv_B = np.linalg.pinv(B)

    # Selection matrix for control inputs
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
        "e_scale": e_scale,
        "site_names": site_names,
        "m": m,
        "n_sites": n_sites,
    }


def controller(model, data, invariants, sigma, previous_solution=None):
    # ----------------------------
    # Unpack invariants
    # ----------------------------
    F        = invariants["F"]
    G        = invariants["G"]
    Pe       = invariants["Pe"]
    site_names = invariants["site_names"]
    n_sites  = invariants["n_sites"]
    m        = invariants["m"]

    # ----------------------------
    # Build desired positions for 5 sites (simple "wrap along backbone")
    # ----------------------------
    # Choose a "base anchor" point to interpolate from.
    # Here: use first body segment origin as anchor (or replace with a site/body you prefer).
    base_anchor = data.body("segment_1__configuration_default").xpos.copy()

    cube_body_id = model.body("cube_body").id
    target_pos = data.xpos[cube_body_id].copy()
    # data.mocap_pos[mocap_id] = ([-0.2,   0.0,  0.5])


    # For i-th site: desired position is along the line base->target
    # alpha goes from small (near base) to 1 (tip)
    # This creates a distributed shape (a first step toward wrapping).
    desired_positions = []
    for i in range(n_sites):
        alpha = (i + 1) / n_sites  # 0.2, 0.4, 0.6, 0.8, 1.0
        # p_des = (1 - alpha) * base_anchor + alpha * target_pos
        p_des = target_pos
        desired_positions.append(p_des)

    # Ensure the last one (ee) is exactly target_pos (optional but nice)
    desired_positions[-1] = target_pos


    # ----------------------------
    # Stack Jacobians, Jdot, errors, velocities
    # ----------------------------
    dq = data.qvel.reshape(-1, 1)

    J_stack = np.zeros((m, model.nv))
    Jdot_stack = np.zeros((m, model.nv))

    e_stack = np.zeros((m, 1))
    v_stack = np.zeros((m, 1))

    # Refresh kinematics
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    for i, site_name in enumerate(site_names):
        site_id = model.site(site_name).id

        # current site pos
        p = data.site(site_id).xpos.copy()
        p_des = desired_positions[i]
        e_i = (p_des - p).reshape(3, 1)  # position error

        # Jacobian (position only)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        # Jdot (position only)
        Jdot6 = compute_jacobian_derivative(model, data, site_id)  # (6,nv)
        Jdotp = Jdot6[:3, :]

        # Stack
        r0 = 3 * i
        r1 = r0 + 3
        J_stack[r0:r1, :] = jacp
        Jdot_stack[r0:r1, :] = Jdotp
        e_stack[r0:r1, :] = e_i
        v_stack[r0:r1, :] = jacp @ dq

    # ----------------------------
    # Build CLF state eta = [ -e ; v ]
    # ----------------------------
    eta = np.vstack((-e_stack, v_stack))  # (2m,1)

    # Lyapunov terms (make scalar floats)
    V = float((eta.T @ Pe @ eta).item())

    # ----------------------------
    # Mass matrix + dynamics
    # ----------------------------
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)

    # Optional inertia noise (you had this)
    # M = M * np.random.normal(1.0, 0.01, size=M.shape)

    # Bias forces already include gravity/Coriolis/etc
    bias = data.qfrc_bias.reshape(-1, 1)
    passive = data.qfrc_passive.reshape(-1, 1)
    contact = data.qfrc_constraint.reshape(-1,1)

    # State-dependent actuator map Bp
    Bp = calculate_input_matrix_at_state(model, data)  # (nv,nu)

    # ----------------------------
    # QP variables
    # ----------------------------
    nu = model.nu
    K = 500
    u   = cp.Variable((nu, 1))
    qdd = cp.Variable((model.nv, 1))
    dl  = cp.Variable((1, 1))

    # ------------------------------------------
    # Per-site stiffness schedule (EE high -> base low)
    # ------------------------------------------
    K_tip  = 2000.0   # highest gain at EE
    K_base = 200.0    # lowest gain near base

    # Your stacking order is i = 0..n_sites-1 (likely base->tip)
    # We want K_i increasing with i, so EE (i=n_sites-1) gets K_tip.
    Ks = np.linspace(K_base, K_tip, n_sites)  # base->tip ramp

    # Expand to 3D blocks (x,y,z for each site)
    K_vec = np.repeat(Ks, 3).reshape(-1, 1)   # shape (m,1) because m=3*n_sites
    sqrtK_vec = np.sqrt(K_vec)


    # ----------------------------
    # Desired task acceleration (stacked)
    # mu = Jdot*qdot + J*qdd  should track  K*e - 2*sqrt(K)*v
    # ----------------------------
    mu = Jdot_stack @ dq + J_stack @ qdd
    mu_des = K * e_stack - 2.0 * np.sqrt(K) * v_stack
    # mu_des = K_vec * e_stack - 2.0 * sqrtK_vec * v_stack

    # ----------------------------
    # CLF derivative
    # eta_dot = F*eta + G*mu
    # dV = eta^T (F^T P + P F) eta + 2 eta^T P G mu
    # ----------------------------
    dV = (eta.T @ (F.T @ Pe + Pe @ F) @ eta) + 2.0 * (eta.T @ Pe @ G @ mu)

    # ----------------------------
    # Objective
    # ----------------------------
    # Track mu_des + regularize qdd and u; keep your nullspace term (optional)
    # For nullspace: define with respect to the stacked Jacobian (position only)
    N = np.eye(model.nv) - np.linalg.pinv(J_stack) @ J_stack
    qdd_null = N @ qdd

    # ============================================================
    # Null reference (stabilizing)
    dx = target_pos - data.site(model.site("ee").id).xpos
    Knull = 100.0
    Dnull = 20 * np.sqrt(Knull)
    q = data.qpos.reshape(-1,1)
    delta_q = cp.Variable(shape=(model.nq, 1))
    objective = cp.Minimize(cp.sum_squares(J_stack @ delta_q - e_stack))
    constraints = [delta_q <= 0.175, delta_q >= -0.175]
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    q_ref = q + delta_q.value
    qdd_0 = -Knull*(q - q_ref) - Dnull*dq
    # qdd_0 = 0
    # ============================================================
    print(contact)

    objective = cp.Minimize(
        cp.sum_squares(mu - mu_des)
        + 1.5 * cp.sum_squares(qdd)
        + 0.5 * cp.sum_squares(u)  # try to match contact forces
        + 1000.0 * cp.sum_squares(dl)
        + 0.5 * cp.sum_squares(qdd_null - qdd_0)
    )

    # ----------------------------
    # Constraints
    # ----------------------------
    # CLF: dV <= -c V + dl
    c = 0.050 / invariants["e_scale"]

    constraints = [
        dV <= -c * V + dl,
        M @ qdd + (bias - passive - 1.25*contact) == Bp @ u,
        # M @ qdd + (bias - passive) == Bp @ u,
        u <= 0.0,
        u >= -20.0,
        dl >= 0.0,
    ]

    prob = cp.Problem(objective, constraints)

    # ----------------------------
    # Warm start
    # ----------------------------
    if previous_solution is not None:
        try:
            u.value = previous_solution.get("u", None)
            qdd.value = previous_solution.get("qdd", None)
            dl.value = previous_solution.get("dl", None)
        except:
            pass

    # Metrics for logging
    task_error = float(np.linalg.norm(e_stack) / n_sites)

    # ----------------------------
    # Solve
    # ----------------------------
    try:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)

        if u.value is not None:
            data.ctrl[:] = np.squeeze(u.value)

            current_solution = {
                "u": u.value.copy(),
                "qdd": qdd.value.copy(),
                "dl": dl.value.copy(),
            }

            return V, task_error, data.qpos.copy(), data.qvel.copy(), current_solution
        else:
            print("failed convergence - no solution")
            return V, task_error, data.qpos.copy(), data.qvel.copy(), previous_solution

    except Exception as e:
        print(f"failed convergence - exception: {e}")
        return V, task_error, data.qpos.copy(), data.qvel.copy(), previous_solution



def simulate_model(headless=False, record_video=False, video_fps=30):
    global B  # Make B accessible globally
    
    model_path = Path("mujoco_models/spirob") / (str(model_name) + str(".xml"))
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    
    print(f"Model loaded: {model.nq} positions, {model.nv} velocities, {model.nu} actuators")
    
    # Calculate the input matrix B from the model
    B = calculate_input_matrix(model)

    
    # Set joint properties for soft robot
    model.jnt_stiffness[:] = stiffness
    model.dof_damping[:] = 0.01
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
    threshold  = 0.0000000000005
    sigma = 10

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        video_frame_interval = max(1, int(1.0 / (video_fps * dt))) if record_video else float('inf')
        
        while (len(task_error_log) < 2 or
            abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold ):
            step_start = time.time()
            V, task_error, q, dq, previous_solution = controller(model, data, invariants, sigma, previous_solution)
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
                V, task_error, q, dq, previous_solution = controller(model, data, invariants, sigma, previous_solution)
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
    return V_log, task_error_log, q_vel, q_pos, time_log, sim_ts



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
    V_log, task_error_log, q_vel, q_pos, time_log, sim_ts = simulate_model(
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

    # #Lyapunov Function Over Time – shows how stability evolves.
    # plt.figure()
    # plt.plot(time_log, V_log, label="Lyapunov Function V")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Lyapunov Function V")
    # plt.title("Lyapunov Function Over Time")
    # plt.grid(True)
    # plt.legend()  # <--- Add legend

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
