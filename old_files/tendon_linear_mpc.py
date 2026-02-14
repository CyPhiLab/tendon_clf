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


# # Print actuator matrix
# B = compute_B_matrix(model, data)
B = np.array([[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1]])


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

# Pre-compute invariant matrices (computed once, used throughout simulation)
def precompute_invariants(model):
    """Pre-compute matrices that don't change during simulation"""
    Kp_null = np.asarray([1] * model.nv)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)
    
    # CLF matrices
    m = 3
    F = np.zeros((2*m, 2*m))
    F[:m, m:] = np.eye(m, m)
    G = np.zeros((2*m, m))
    G[m:, :] = np.eye(m)
    e = 0.05
    Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))
    
    # Input matrix pseudoinverse
    pinv_B = np.linalg.pinv(B)
    
    # Selection matrix for compression/extension actuators
    nu = 2
    
    return {
        'Kp_null': Kp_null,
        'Kd_null': Kd_null,
        'F': F,
        'G': G,
        'Pe': Pe,
        'pinv_B': pinv_B,
        'e': e
    }

def controller(model, data, invariants, previous_solution=None):
    jac = np.zeros((6, model.nv))
    twist = np.zeros(3)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    # data.mocap_pos[mocap_id] = ([0.15,   0.0,  -0.1])
    # data.mocap_pos[mocap_id] = ([0.05,   0.0,  -0.1])
    # data.mocap_pos[mocap_id] = ([0.15,   0.0,  -0.15])
    # data.mocap_pos[mocap_id] = ([0.15,   0.0,  -0.05])

    # Extract pre-computed values
    F = invariants['F']
    G = invariants['G']
    Pe = invariants['Pe']
    pinv_B = invariants['pinv_B']
    e = invariants['e']

    site_name = "ee"
    site_id = model.site(site_name).id
    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
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
    e = e

    # Use original constraint formulation
    dq = data.qvel.reshape(-1,1)

    # eta_0 (numeric)
    eta0 = np.concatenate((-twist, jac @ data.qvel)).reshape(6, 1)

    N = 10  # prediction horizon
    gamma = 0.9
    dt = 1.0 / f_ctrl

    # Define decision variables (create fresh variables each time)
    nu = 2
    mu = cp.Variable(shape=(3, N))      # mu[:,k] = mu_k
    u_k  = cp.Variable((nu, 1))     # single applied control (keep as you had)
    eta_k = cp.Variable((6, N))    # eta_k[:,k] = eta at step k

    # Since eta is already an error-state [-twist; J qdot], the goal is eta -> 0
    eta_target = np.zeros((6, 1))

    # qdd must be a CVXPY expression (and keep only k=0 since you apply u_k once)
    Jpinv = cp.Constant(np.linalg.pinv(jac))
    qdd = Jpinv @ (mu[:, 0:1] - dJ_dt @ dq)   # (nv x 1)

    # Initial condition constraint
    constraints = []
    constraints += [eta_k[:, 0:1] == eta0]
    objective = 0.0

    # Linear discrete dynamics rollout as constraints
    for k in range(N - 1):
        constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
        eta_k1 = eta_k[:, k+1:k+2]
        mu_k1  = mu[:, k:k+1]
        mu_des_k = -500 * eta_k[0:3, k:k+1] - 2 * np.sqrt(500) * eta_k[3:6, k:k+1]
        objective += (gamma**k) * (cp.sum_squares(mu_k1 - mu_des_k) + (gamma**k)*cp.sum_squares(eta_k1 - eta_target))

    # Terminal penalty (use eta_k at terminal, not eta_next)
    eta_N = eta_k[:, N-1:N]
    objective += 10*cp.quad_form(eta_N, Pe)
    objective = cp.Minimize(objective + 0.2 * cp.sum_squares(u_k) )# + 0.2 * cp.square(cp.norm(qdd)))

    # Inverse dynamics constraint
    constraints += [pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u_k,
        -1.0 <= u_k,
        1.0 >= u_k,
    ]

    prob = cp.Problem(objective=objective, constraints=constraints)
    
    # Warm start with previous solution if available
    if previous_solution is not None:
        try:
            u_k.value = previous_solution['u']
            qdd.value = previous_solution['qdd'] 
        except:
            pass  # If warm start fails, proceed without it


    task_error = np.linalg.norm(dx)
    q = data.qpos
    dq = data.qvel
    
    try:
        prob.solve(solver=cp.SCS, verbose=False, warm_start=True)

        if u_k.value is not None:
            u_opt = u_k.value.copy()

            u_tendon = np.array([
                max(u_opt[0, 0], 0.0),
                -min(u_opt[0, 0], 0.0),
                max(u_opt[1, 0], 0.0),
                -min(u_opt[1, 0], 0.0)
            ])

            # data.ctrl[:] = np.squeeze(u_tendon)
            data.ctrl[:] = u_tendon

            current_solution = {
                'u': u_opt,
            }

            return task_error, q, dq, current_solution

        else:
            print(f"failed convergence - no solution\n")
            return task_error, q, dq, previous_solution
    except Exception as e:
        print(f"failed convergence - exception: {e}\n")
        return task_error, q, dq, previous_solution

def simulate_model(headless=False):
    model.jnt_stiffness[:] = stiffness
    
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    # Pre-compute invariant matrices
    print("Pre-computing invariant matrices...")
    invariants = precompute_invariants(model)
    
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

    task_error_log = []
    q_vel = []
    q_pos = []
    time_log = []
    t = 0.0
    dt = model.opt.timestep


    last_ctrl = time.time()
    max_sim_time = 25.0  # Run for 1 second of simulation time
    log_frequency = 5  # Log every 5 steps instead of every step
    step_count = 0
    threshold  = 1e-3

    if headless:
        # Run simulation without viewer for maximum performance
        print("Running headless simulation...")
        while (len(task_error_log) < 2 or
            abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold ):
            step_start = time.time()
            V, task_error, q, dq, previous_solution = controller(model, data, invariants, previous_solution)
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
                time_log.append(t)
            
            step_count += 1
            t += dt
    else:
        # Run simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_start = time.time()
            while viewer.is_running() and (
                len(task_error_log) < 2 or
                abs(task_error_log[-1] - task_error_log[-2]) / dt > threshold 
            ):
                step_start = time.time()
                first_time = time.time()
                task_error, q, dq, previous_solution = controller(model, data, invariants, previous_solution)
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
    return task_error_log, q_vel, q_pos, time_log, sim_ts



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized tendon control simulation')
    parser.add_argument('--headless', action='store_true', help='Run simulation without GUI for maximum performance')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots at the end')
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

