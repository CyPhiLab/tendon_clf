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
    e = 0.02
    Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))
    
    # Selection matrix for control inputs
    nu = model.nu  # Use actual number of actuators from model
    sel = np.ones((nu, 1))  # All control inputs are active
    
    return {
        'Kp_null': Kp_null,
        'Kd_null': Kd_null,
        'F': F,
        'G': G,
        'Pe': Pe,
        'e': e
    }

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
    Pe = invariants['Pe']
    Bp = calculate_input_matrix_at_state(model, data)  # or cached update every N steps
    pinv_Bp = np.linalg.pinv(Bp)

    site_name = "ee"
    site_id = model.site(site_name).id

    # Set target position for SpiRob end-effector
    # data.mocap_pos[mocap_id] = np.array([0.3, 0.0, 0.2])
    # # data.mocap_pos[mocap_id] = np.array([0.2, 0.2, 0.3])
    # data.mocap_pos[mocap_id] = np.array([0.1, 0.1, 0.3])
    # data.mocap_pos[mocap_id] = np.array([0.2, 0.0, 0.3])
    # data.mocap_pos[mocap_id] = np.array([0.1, 0.0, 0.1])
    # data.mocap_pos[mocap_id] = np.array([0.2, 0.0, 0.2])
    # data.mocap_pos[mocap_id] = np.array([0.15, 0.0, 0.15])
    # data.mocap_pos[mocap_id] = np.array([0.15, 0.0, 0.2])
    # data.mocap_pos[mocap_id] = np.array([0.15, 0.0, 0.25])
    # data.mocap_pos[mocap_id] = np.array([0.16, 0.0, 0.35])



    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos

    twist[:3] = dx 

    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori 

    q = data.qpos
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M * np.random.normal(1.0, 0.05, size=M.shape)  # Add some noise to the inertia matrix
    dJ_dt = compute_jacobian_derivative(model, data, site_id)


    # Use original constraint formulation
    dq = data.qvel.reshape(-1,1)
    C, g = get_coriolis_and_gravity(model, data)


    # eta_0 (numeric)
    eta0 = np.concatenate((-twist, jac @ data.qvel)).reshape(12, 1)

    N = 10  # prediction horizon
    gamma = 0.9
    dt = 1.0 / f_ctrl

    # Define decision variables (create fresh variables each time)
    nu = model.nu  # Use actual number of actuators from model
    nq = model.nq
    qdd_k  = cp.Variable((nq, N))  # joint accelerations
    mu   = cp.Variable((6, N))      # mu[:,k] = mu_k
    u_k  = cp.Variable((nu, N))     # single applied control (keep as you had)
    eta_k = cp.Variable((12, N))    # eta_k[:,k] = eta at step k

    # Since eta is already an error-state [-twist; J qdot], the goal is eta -> 0
    eta_target = np.zeros((12, 1))

    # Initial condition constraint
    constraints = []
    constraints += [eta_k[:, 0:1] == eta0]

    objective = 0

    # Linear discrete dynamics rollout as constraints
    for k in range(N - 1):
        constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
        constraints += [mu[:, k:k+1] == jac @ qdd_k[:, k:k+1] + dJ_dt @ dq]
        dq = dq + dt * qdd_k[:, k:k+1]
        constraints += [pinv_Bp @ (M @ qdd_k[:, k:k+1] + C @ dq + g.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u_k[:, k:k+1]]
        constraints += [np.zeros((nu,1)) >= u_k[:, k:k+1]]
        eta_k1 = eta_k[:, k+1:k+2]
        mu_k1  = mu[:, k:k+1]
        mu_des_k = -500 * eta_k[0:6, k:k+1] - 2*np.sqrt(500) * eta_k[6:12, k:k+1]
        objective += (gamma**k) * (cp.sum_squares(mu_k1 - mu_des_k) + cp.sum_squares(eta_k1 - eta_target) + 0.1 * cp.sum_squares(u_k[:, k:k+1]) + 0.1 * cp.sum_squares(qdd_k[:, k:k+1]))

    # Terminal penalty (use eta_k at terminal, not eta_next)
    eta_N = eta_k[:, N-1:N]
    objective += 10*cp.quad_form(eta_N, Pe)
    objective = cp.Minimize(objective)
    
    prob = cp.Problem(objective, constraints)
    # Warm start with previous solution if available
    if previous_solution is not None:
        try:
            u_k.value[:, 0] = previous_solution['u']
        except:
            pass  # If warm start fails, proceed without it

    twist[3:] = 0

    task_error = np.linalg.norm(dx)
    q = data.qpos
    dq = data.qvel
    
    try:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        if u_k.value is not None:
            data.ctrl = np.squeeze(u_k.value[:, 0]) 
            # Cache solution for next iteration
            current_solution = {
                'u': u_k.value[:, 0].copy()
            }
            # print(f"converged\n")
            return task_error, q, dq, current_solution
        else:
            print(f"failed convergence - no solution\n")
            return task_error, q, dq, previous_solution
    except Exception as e:
        print(f"failed convergence - exception: {e}\n")
        return task_error, q, dq, previous_solution

def simulate_model(headless=False, record_video=False, video_fps=30):
    global B  # Make B accessible globally
    
    model_path = Path("mujoco_models/spirob") / (str(model_name) + str(".xml"))
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    
    print(f"Model loaded: {model.nq} positions, {model.nv} velocities, {model.nu} actuators")
    
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

    task_error_log = []
    q_vel = []
    q_pos = []
    time_log = []
    t = 0.0
    dt = model.opt.timestep

    log_frequency = 5  # Log every 5 steps instead of every step
    step_count = 0
    threshold  = 0.00005

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        video_frame_interval = max(1, int(1.0 / (video_fps * dt))) if record_video else float('inf')
        
        while (len(task_error_log) < 2 or
            abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold ):
            task_error, q, dq, previous_solution = controller(model, data, invariants, previous_solution)
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

                task_error_log.append(task_error)
                q_vel.append(dq.squeeze().copy())
                q_pos.append(q.squeeze().copy())
                time_log.append(t)
            
            step_count += 1
            t += dt
    else:
        # Run simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            video_frame_interval = max(1, int(1.0 / (video_fps * dt))) if record_video else float('inf')
            
            if record_video:
                # Create renderer for video capture from viewer
                renderer = mujoco.Renderer(model, height=1080, width=1920)
            
            while viewer.is_running() and (
                len(task_error_log) < 2 or
                abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold 
            ):
                task_error, q, dq, previous_solution = controller(model, data, invariants, previous_solution)
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
    return task_error_log, q_vel, q_pos, time_log, sim_ts



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
    task_error_log, q_vel, q_pos, time_log, sim_ts = simulate_model(
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

    # # Control inputs plot
    # actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
    # ctrl = np.array(sim_ts["ctrl"])  # shape: (timesteps, nu)
    # num_actuators = ctrl.shape[1]
    # control_limit = 25

    # fig, axs = plt.subplots(actuated_indices, 1, figsize=(10, 8), sharex=True)

    # for i in range(actuated_indices):
    #     axs[i].plot(time_log, ctrl[:, i], label=f"Actuator {i}")
    #     axs[i].axhline(control_limit, color='r', linestyle='--', linewidth=1)
    #     axs[i].axhline(-control_limit, color='r', linestyle='--', linewidth=1)
    #     axs[i].set_ylabel("Torque (Nm)")
    #     axs[i].legend()
    #     axs[i].grid(True)

    # axs[-1].set_xlabel("Time (s)")
    # fig.suptitle("Control Inputs at Actuated Joints")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])


