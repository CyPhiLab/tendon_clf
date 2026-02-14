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

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


# MJCF XML as string
mjcf = """
<mujoco model="4link_tendon_planar">
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81" integrator="implicitfast"/>

    <default>
        <joint type="hinge" axis="0 1 0" limited="true" range="-1.57 1.57" damping="0.01"/>
        <geom type="capsule" size="0.005 0.03" rgba="0.5 0.5 0.5 1" mass="0.01"/>
    </default>


    <worldbody>
        <camera name="side_view" pos="0 0.1 0.05" xyaxes="1 0 0  0 0 1"/>
        <body name="link1" pos="0 0 0">
            <joint name="joint1" />
            <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
            <body name="link2" pos="0.06 0 0">
                <joint name="joint2"/>
                <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
                <body name="link3" pos="0.06 0 0">
                    <joint name="joint3"/>
                    <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
                    <body name="link4" pos="0.06 0 0">
                        <joint name="joint4"/>
                        <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
                        <body name="attachment" pos="0.03 0.0 0.0">
                              <site name="ee" rgba="1 0 0 1" size="0.001" group="1"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <worldbody>
        <body name="target" pos="0.1 0.0 -0.15" quat="0 1 0 0" mocap="true">
        <site type="sphere" size="0.01" rgba="0 0 1 1" group="2"/>
        </body>
    </worldbody>

    <!-- Tendon-based antagonistic actuation -->
    <tendon>
        <!-- Tendon pair A -->
        <fixed name="tendon_a_flex">
            <joint joint="joint1" coef="1"/>
            <joint joint="joint2" coef="1"/>
        </fixed>
        <fixed name="tendon_a_ext">
            <joint joint="joint1" coef="-1"/>
            <joint joint="joint2" coef="-1"/>
        </fixed>

        <!-- Tendon pair B -->
        <fixed name="tendon_b_flex">
            <joint joint="joint3" coef="1"/>
            <joint joint="joint4" coef="1"/>
        </fixed>
        <fixed name="tendon_b_ext">
            <joint joint="joint3" coef="-1"/>
            <joint joint="joint4" coef="-1"/>
        </fixed>
    </tendon>


    <!-- Actuators for tendons -->
    <actuator>
        <motor tendon="tendon_a_flex" ctrlrange="0 1" gear="0.1"/>
        <motor tendon="tendon_a_ext" ctrlrange="0 1" gear="0.1"/>
        <motor tendon="tendon_b_flex" ctrlrange="0 1" gear="0.1"/>
        <motor tendon="tendon_b_ext" ctrlrange="0 1" gear="0.1"/>
    </actuator>
</mujoco>
"""  # Replace with the MJCF content provided above


# Load model from string
model = mujoco.MjModel.from_xml_string(mjcf)
data = mujoco.MjData(model)

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

def circular_trajectory(t):
    """
    Circular trajectory through the 4 given points.
    One full revolution in time T.
    """

    L = 0.24/2
    R = L/2
    h = 0

    # Circle parameters
    cx, cy, cz = L, 0, -L+h
    r = R

    # Angle
    omega = 0.5 * np.pi 
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


# # Print actuator matrix
def compute_B_matrix(model, data):
    nv, nu = model.nv, model.nu
    B = np.zeros((nv, nu))

    ctrl_backup = data.ctrl.copy()

    for i in range(nu):
        data.ctrl[:] = 0.0
        data.ctrl[i] = 1.0
        mujoco.mj_forward(model, data)
        B[:, i] = data.qfrc_actuator

    data.ctrl[:] = ctrl_backup  # Restore control inputs
    return B
B = compute_B_matrix(model, data)
# B = np.array([[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1]])

print("B is", B)


# print(B)

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

def compute_fixed_S_from_qbar(model, data, site_id, qbar=None):
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

    jac = jac[:3,:]

    S = jac.T.copy()  # (nv x m)

    # Restore
    mujoco.mj_setState(model, data, backup, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    return S

# Pre-compute invariant matrices (computed once, used throughout simulation)
def precompute_invariants(model, data):
    """Pre-compute matrices that don't change during simulation"""
    Kp_null = np.asarray([1] * model.nv)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)
    
    # Input matrix pseudoinverse
    pinv_B = np.linalg.pinv(B)
    
    site_name = "ee"
    site_id = model.site(site_name).id

    # Fixed S (computed ONCE)
    S = compute_fixed_S_from_qbar(model, data, site_id, qbar=None)
    
    return {
        'Kp_null': Kp_null,
        'Kd_null': Kd_null,
        'pinv_B': pinv_B,
        'S': S
    }

def controller(model, data, invariants, previous_solution=None, S = None, trajectory=None):
    jac = np.zeros((6, model.nv))
    twist = np.zeros(3)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))


    # Track desired trajectory 
    pos = trajectory["pos"]   # (3,)
    vel = trajectory["vel"]   # (3,)
    acc = trajectory["acc"]   # (3,)


    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    data.mocap_pos[mocap_id] = pos

    site_name = "ee"
    site_id = model.site(site_name).id
    dx = pos - data.site(site_id).xpos
    twist = dx 

    q = data.qpos
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M = M * np.random.normal(1.0, 0.01, size=M.shape)  # Add some noise to the inertia matrix
    dJ_dt_full = compute_jacobian_derivative(model, data, site_id)
    dJ_dt = dJ_dt_full[:3,:]
    jac = jac[:3,:]

    # Use original constraint formulation
    task_error = np.linalg.norm(dx)
    q = data.qpos
    dq = data.qvel

    # Task-space inertia 
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    Jbar = M_inv @ jac.T @ Mx
    C, g = get_coriolis_and_gravity(model, data)
    ydd = acc + 500 * twist  - 20 * (jac @ dq - vel)
    Cy = (Jbar.T @ C @ data.qvel) - (Mx @ (dJ_dt @ data.qvel))
    f = Mx @ ydd + Cy

    # Santina synergy-projected torque
    sigma = np.linalg.pinv(jac @ M_inv @ S, rcond=1e-6) @ (jac @ M_inv @ jac.T @ f)  # (m,)
    tau_synergy = S @ sigma  # (nv,)

    # Add gravity + passive 
    tau = tau_synergy + g - data.qfrc_passive

    try:
        data.ctrl = np.linalg.pinv(B, rcond=1e-6) @ tau

    except:
        print(f"failed convergence\n")
        pass
    return task_error, q, dq, previous_solution, u.copy()

def simulate_model(headless=False):
    model.jnt_stiffness[:] = stiffness
    
    # model.dof_damping[:] = 0.2
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    data.qpos[2] = 0.0
    model.jnt_range[range(2,len(data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(data.qpos),3)]
    # model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    # Pre-compute invariant matrices
    print("Pre-computing invariant matrices...")
    invariants = precompute_invariants(model, data)
    
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
    q_des = np.ones(data.qpos.shape[0])*0.2
    # print(np.shape(q_des))

    task_error_log = []
    u_log = []
    q_vel = []
    q_pos = []
    time_log = []
    t = 0.0
    dt = model.opt.timestep


    last_ctrl = time.time()
    max_sim_time = 25.0  # Run for 1 second of simulation time
    log_frequency = 5  # Log every 5 steps instead of every step
    step_count = 0
    period = 8  # Two full revolutions

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        while t < period and viewer.is_running():
            step_start = time.time()
            trajectory = circular_trajectory(t)
            task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution, trajectory=trajectory)
            mujoco.mj_step(model, data)
            
            # Only log every N steps to reduce overhead
            if step_count % log_frequency == 0:
                # print(f"Sim time: {data.time:.3f}s")
                
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
                u_log.append(u.squeeze().copy())
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
                task_error, q, dq, previous_solution, u = controller(model, data, invariants, previous_solution, trajectory=trajectory)
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
                    q_vel.append(dq.squeeze().copy())
                    q_pos.append(q.squeeze().copy())
                    u_log.append(u.squeeze().copy())
                    time_log.append(t)
                
                step_count += 1
                
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                t += dt

                # Rudimentary time keeping, will drift relative to wall clock.
                # Removed sleep to run as fast as possible for 1 second of sim time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")
    return task_error_log, q_vel, q_pos, time_log, sim_ts, u_log



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized tendon control simulation')
    parser.add_argument('--headless', action='store_true', help='Run simulation without GUI for maximum performance')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots at the end')
    args = parser.parse_args()
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Simulate the model
    task_error_log, q_vel, q_pos, time_log, sim_ts, u_log = simulate_model(headless=args.headless)
    
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

    csv_path = "tendon_mpc_tracking.csv"
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

    plt.show()

