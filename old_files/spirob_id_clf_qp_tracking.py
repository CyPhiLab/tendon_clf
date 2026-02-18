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
w = np.pi/4

# B matrix will be calculated from the MuJoCo model after loading
B = None



def circular_trajectory(t):
    """
    Circular trajectory through the 4 given points.
    One full revolution in time T.
    """

    # Circle parameters
    L = 0.45/2
    R = L/2

    cx, cy, cz = L, 0.0, L
    r = R

    # Angle
    omega =  np.pi/5 
    theta = omega * t

    # Position
    x = cx + r * np.cos(theta)
    y = cy
    z = cz + r * np.sin(theta)

    # Velocity
    xd = -r * omega * np.sin(theta)
    yd = 0.0
    zd =  r * omega * np.cos(theta)

    # Acceleration
    xdd = -r * omega**2 * np.cos(theta)
    ydd = 0.0
    zdd = -r * omega**2 * np.sin(theta)

    pos = np.array([x, y, z])
    vel = np.array([xd, yd, zd])
    acc = np.array([xdd, ydd, zdd])

    return {"pos": pos, "vel": vel, "acc": acc}

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
    e = 0.01
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

def controller(model, data, invariants, previous_solution=None, trajectory=None):

    nv, nu = model.nv, model.nu

    # =========================
    # IDs
    # =========================
    mocap_id = model.body("target").mocapid[0]
    site_id  = model.site("ee").id

    # =========================
    # Desired trajectory (6D)
    # =========================
    pos = np.hstack([trajectory["pos"], [0,0,0]])
    vel = np.hstack([trajectory["vel"], [0,0,0]])
    acc = np.hstack([trajectory["acc"], [0,0,0]])

    data.mocap_pos[mocap_id] = pos[:3]

    # =========================
    # Kinematics
    # =========================
    jac = np.zeros((6, nv))
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    dq = data.qvel.reshape(-1,1)
    xdot = jac @ dq

    # =========================
    # Position error
    # =========================
    dx = pos[:3] - data.site(site_id).xpos
    twist = np.zeros((6,1))
    twist[:3,0] = dx

    # =========================
    # CLF state (12D — DO NOT CHANGE)
    # =========================
    eta = np.vstack([
        -twist,
        xdot - vel.reshape(6,1)
    ])

    F  = invariants["F"]
    G  = invariants["G"]
    Pe = invariants["Pe"]
    e_clf = invariants["e"]

    V = float(eta.T @ Pe @ eta)

    # =========================
    # Dynamics
    # =========================
    M = np.zeros((nv,nv))
    mujoco.mj_fullM(model, M, data.qM)

    bias = data.qfrc_bias.reshape(-1,1)
    passive = data.qfrc_passive.reshape(-1,1)

    Bp = calculate_input_matrix_at_state(model, data)
    pinv_Bp = np.linalg.pinv(Bp)

    dJ_dt = compute_jacobian_derivative(model, data, site_id)

    # =========================
    # Decision variables
    # =========================
    qdd = cp.Variable((nv,1))
    u   = cp.Variable((nu,1))
    dl  = cp.Variable((1,1))

    # =========================
    # Task acceleration
    # =========================
    xdd = dJ_dt @ dq + jac @ qdd
    xdd_des = acc.reshape(6,1)

    dV = (
        eta.T @ (F.T @ Pe + Pe @ F) @ eta
        + 2 * eta.T @ Pe @ G @ (xdd - xdd_des)
    )

    # =========================
    # Directional manipulability alignment
    # (POSITION ONLY)
    # =========================
    Jp = jac[:3,:]

    # direction = desired velocity direction (for trajectory tracking)
    v = trajectory["vel"]
    if np.linalg.norm(v) > 1e-6:
        d_dir = v / np.linalg.norm(v)
    else:
        d_dir = dx / (np.linalg.norm(dx)+1e-8)

    Jpinv = np.linalg.pinv(Jp)
    N = np.eye(nv) - Jpinv @ Jp

    g = grad_wd(model, data, site_id, d_dir).reshape(-1)
    g /= (np.linalg.norm(g)+1e-8)

    k_align = 30.0
    tau_align = (k_align * (N.T @ g)).reshape(-1,1)

    # =========================
    # Tracking objective
    # =========================
    Kp = 2500.0
    Kd = 2*np.sqrt(Kp)

    xdd_ref = xdd_des + Kp*twist + Kd*(vel.reshape(6,1)-xdot)

    objective = cp.Minimize(
        cp.sum_squares(xdd - xdd_ref)
        + 0.05*cp.sum_squares(qdd)
        + 0.5*cp.norm1(u)
        + 1000*cp.sum_squares(dl)
    )

    # =========================
    # Constraints
    # =========================
    constraints = [
        dV <= -1/max(e_clf,1e-6)*V + dl,
        pinv_Bp @ (M@qdd + bias - passive - tau_align) == u,
        u <= 0,
        u >= -100,
        dl >= 0
    ]

    prob = cp.Problem(objective, constraints)

    # warm start
    if previous_solution is not None:
        try:
            u.value   = previous_solution["u"]
            qdd.value = previous_solution["qdd"]
        except:
            pass

    # =========================
    # Solve
    # =========================
    try:
        prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

        if u.value is None:
            return V, np.linalg.norm(dx), data.qpos.copy(), data.qvel.copy(), previous_solution, None

        data.ctrl = np.squeeze(u.value).clip(-100,0)

        current_solution = {
            "u":u.value.copy(),
            "qdd":qdd.value.copy()
        }

        return V, np.linalg.norm(dx), data.qpos.copy(), data.qvel.copy(), current_solution, u.value.copy()

    except Exception as ex:
        print("QP fail:",ex)
        return V, np.linalg.norm(dx), data.qpos.copy(), data.qvel.copy(), previous_solution, None


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
    period = 4 * np.pi/w

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        while t < max_sim_time:
            step_start = time.time()
            trajectory = circular_trajectory(t)

            V, task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution, trajectory)
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
            sim_start = time.time()
            while t < period and viewer.is_running():
                step_start = time.time()
                first_time = time.time()
                trajectory = circular_trajectory(t)
                V, task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution, trajectory)
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
                    task_error_log.append(task_error)
                    if u is not None:
                        u_log.append(u.squeeze())
                    else:
                        u_log.append(np.full(9, np.nan))
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
    df.insert(1, "task_error", error_log)
    df.insert(2, "lyapunov_function", V_log)
    csv_path = "spirob_id_clf_qp_tracking.csv"
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
