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

def circular_trajectory(t):
    """
    Circular trajectory through the 4 given points.
    One full revolution in time T.
    """

    # Circle parameters
    L = 0.5/2
    R = L/4

    cx, cy, cz = L, 0.0, L
    r = R

    # Angle
    omega =  np.pi/4 
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

def controller(model, data, invariants, previous_solution=None, trajectory=None):
    # ---------- setup ----------
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    site_name = "ee"
    site_id = model.site(site_name).id

    # ---------- precomputed ----------
    F  = invariants["F"]
    G  = invariants["G"]
    Pe = invariants["Pe"]

    # input map (state-dependent)
    Bp = calculate_input_matrix_at_state(model, data)  # or cached update every N steps
    pinv_Bp = np.linalg.pinv(Bp)

    # ---------- MPC params ----------
    N = 10
    dt = 1.0 / f_ctrl
    gamma = 0.9

    # ---------- trajectory preview (tracking) ----------
    # Expect circular_trajectory(t) -> dict with keys: 'pos','vel','acc' (each 3,)
    traj_seq = [circular_trajectory(data.time + k * dt) for k in range(N)]
    pos_seq = [np.hstack([tr["pos"], np.zeros(3)]) for tr in traj_seq]  # 6-dim (pos + ori-vel placeholder)
    vel_seq = [np.hstack([tr["vel"], np.zeros(3)]) for tr in traj_seq]  # 6-dim
    acc_seq = [np.hstack([tr["acc"], np.zeros(3)]) for tr in traj_seq]  # 6-dim

    acc_seq_cp = [cp.Constant(acc_seq[k].reshape(6, 1)) for k in range(N)]

    # visualize current target
    data.mocap_pos[mocap_id] = pos_seq[0][:3]

    # ---------- kinematics ----------
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # ---------- dynamics ----------
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M * np.random.normal(1.0, 0.05, size=M.shape)  # optional inertia noise
    dJ_dt = compute_jacobian_derivative(model, data, site_id)

    C, g = get_coriolis_and_gravity(model, data)

    # ---------- error state eta0 ----------
    # position/orientation tracking error (twist)
    # position error uses previewed reference (k=0)
    dx = pos_seq[0][:3] - data.site(site_id).xpos
    twist[:3] = dx

    # orientation error: use mocap_quat as desired (no preview here)
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori

    # task-space velocity error uses previewed v_ref (k=0)
    ydot = (jac @ data.qvel).reshape(6,)
    vel_err = ydot - vel_seq[0]

    eta0 = np.concatenate([(-twist).reshape(6,), vel_err.reshape(6,)]).reshape(12, 1)

    # ---------- decision variables ----------
    nu = model.nu
    nq = model.nq

    qdd_k  = cp.Variable((nq, N))   # joint accelerations
    mu     = cp.Variable((6, N))    # task accelerations
    u_k    = cp.Variable((nu, N))   # actuator commands
    eta_k  = cp.Variable((12, N))   # error state

    constraints = []
    constraints += [eta_k[:, 0:1] == eta0]

    objective = 0

    dq_expr = data.qvel.reshape(-1, 1)  # affine expression rollout

    # ---------- rollout ----------
    for k in range(N - 1):
        # error dynamics with reference acceleration preview
        constraints += [
            eta_k[:, k+1:k+2] ==
            eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ (mu[:, k:k+1] - acc_seq_cp[k]))
        ]

        # kinematic relation: mu = J qdd + dJ dt * qd
        constraints += [
            mu[:, k:k+1] == jac @ qdd_k[:, k:k+1] + dJ_dt @ dq_expr
        ]

        # velocity rollout
        dq_expr = dq_expr + dt * qdd_k[:, k:k+1]

        # inverse dynamics (mapped into actuator space)
        constraints += [
            pinv_Bp @ (
                M @ qdd_k[:, k:k+1] +
                C @ dq_expr +
                g.reshape(-1, 1) -
                data.qfrc_passive.reshape(-1, 1)
            ) == u_k[:, k:k+1]
        ]

        # control input constraint: u <= 0 (and bounded below)
        constraints += [
            u_k[:, k:k+1] <= np.zeros((nu, 1))
        ]

        # CLF-style desired task acceleration (tracking form)
        mu_des_k = (
            acc_seq_cp[k]
            - 500.0 * eta_k[0:6, k:k+1]
            - 2.0 * np.sqrt(500.0) * eta_k[6:12, k:k+1]
        )

        objective += (gamma**k) * (
            cp.sum_squares(mu[:, k:k+1] - mu_des_k)
            + 0.2 * cp.sum_squares(u_k[:, k:k+1])
            + 0.2 * cp.sum_squares(qdd_k[:, k:k+1])
        )

    # terminal penalty
    eta_N = eta_k[:, N-1:N]
    objective += cp.quad_form(eta_N, Pe)

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # warm start
    if previous_solution is not None:
        try:
            u_k.value[:, 0] = previous_solution["u"]
        except:
            pass

    task_error = float(np.linalg.norm(dx))

    try:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)

        if u_k.value is not None:
            data.ctrl = np.squeeze(u_k.value[:, 0])
            current_solution = {"u": u_k.value[:, 0].copy()}
            return task_error, data.qpos, data.qvel, current_solution, u_k.value[:, 0].copy()
        else:
            return task_error, data.qpos, data.qvel, previous_solution, None

    except Exception as e:
        print(f"failed convergence - exception: {e}\n")
        return task_error, data.qpos, data.qvel, previous_solution, None



def simulate_model(headless=False, record_video=False, video_fps=30):
    global B  # Make B accessible globally
    
    model_path = Path("mujoco_models/spirob") / (str(model_name) + str(".xml"))
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    
    print(f"Model loaded: {model.nq} positions, {model.nv} velocities, {model.nu} actuators")
    
    # # Calculate the input matrix B from the model
    # B = calculate_input_matrix_at_state(model,data)

    
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
    period = 16

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        while t < max_sim_time:
            step_start = time.time()
            trajectory = circular_trajectory(t)

            task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution, trajectory)
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
                task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution, trajectory)
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
    return task_error_log, q_vel, q_pos, time_log, sim_ts, u_log


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
    task_error_log, q_vel, q_pos, time_log, sim_ts, u_log = simulate_model(
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
    u_log = np.array(u_log)
    time_log_csv = np.array(time_log)
    error_log = np.array(task_error_log)
    df = pd.DataFrame(
        u_log,
        columns=[f"u{i}" for i in range(u_log.shape[1])]
    )
    df.insert(0, "time", time_log_csv)
    df.insert(1, "task_error", error_log)
    csv_path = "spirob_mpc_tracking.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


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
    # control_limit = 0.0

    # fig, axs = plt.subplots(len(actuated_indices), 1, figsize=(10, 8), sharex=True)

    # for i in range(len(actuated_indices)):
    #     axs[i].plot(time_log, ctrl[:, i], label=f"Actuator {i}")
    #     axs[i].axhline(control_limit, color='r', linestyle='--', linewidth=1)
    #     axs[i].axhline(-control_limit, color='r', linestyle='--', linewidth=1)
    #     axs[i].set_ylabel("Tension (N)")
    #     axs[i].legend()
    #     axs[i].grid(True)

    # axs[-1].set_xlabel("Time (s)")
    # fig.suptitle("Control Inputs at Actuated Joints")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plt.show()
