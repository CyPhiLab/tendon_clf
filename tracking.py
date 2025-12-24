import argparse
import matplotlib.pyplot as plt
import os
os.environ["MUJOCO_GL"] = "egl"   # or comment this line out entirely
import mujoco
import mujoco.viewer
import numpy as np
import os
from pathlib import Path
import time
from scipy import linalg
import cvxpy as cp
from scipy.linalg import eigh
import pandas as pd
from matplotlib import rcParams




# # Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


model_name = f"helix_control"

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
w = 5 * np.pi / T

B = np.zeros((36, 9))
for i in range(3):  # Iterate over u1, u2, u3 blocks
    for j in range(4):  # Repeat each block 4 times
        row_start = i * 12 + j * 3  # Compute row index
        col_start = i * 3  # Compute column index
        B[row_start:row_start+3, col_start:col_start+3] = np.eye(3)  # Assign identity

print(B)

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


def finite_difference_Mdot(model, data, delta_t=1e-6):
    # Backup current state
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()

    # Compute M at current q
    M0 = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M0, data.qM)

    # Integrate qpos forward by dq * delta_t
    mujoco.mj_integratePos(model, data.qpos, data.qvel, delta_t)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_forward(model, data)

    # Compute M at new q
    M1 = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M1, data.qM)

    # Restore qpos
    data.qpos[:] = qpos_orig
    data.qvel[:] = qvel_orig
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_forward(model, data)

    # Approximate Mdot
    Mdot = (M1 - M0) / delta_t
    return Mdot

# def pos_traj_circle_xy(t):
#     """Circle in XY, fixed Z."""
#     center = np.array([0, 0.00, 0.7])   # near your current target
#     Ax, Ay, Az = 0.435, 0, 0.435           # radii (m)
#     theta = np.pi/6 * 1                       
#     return center + np.array([Ax*np.cos(theta), 0, Ay*np.sin(theta)])

# def pos_traj_circle_xy(t):
#     """Circle in XY, fixed Z."""
#     # center = np.array([0, 0.00, 0.35])   # near your current target
#     center = np.array([0, 0.00, 0.265])   # near your current target
#     Ax, Ay, Az = 0.1, 0.1, 0.1           # radii (m); Az=0 keeps Z fixed
#     wx = 0.5 * np.pi           
#     wy = 2 * np.pi                            # rad/s
#     return center + np.array([Ax*np.cos(wx*t), Ay*np.sin(wy*t), 0*Az*np.sin(w*t)])

# def vel_traj_circle_xy(t):
#     """Velocity for circular XY trajectory, fixed Z."""
#     Ax, Ay, Az = 0.1, 0.1, 0.1
#     wx = 0.5 * np.pi           
#     wy = 2 * np.pi  
#     return np.array([
#         -Ax * wx * np.sin(wx * t),
#          Ay * wy * np.cos(wy * t),
#          0*Az * w * np.cos(w * t)
#     ])

# def acc_traj_circle_xy(t):
#     """Acceleration for circular XY trajectory, fixed Z."""
#     Ax, Ay, Az = 0.1, 0.1, 0.1
#     wx = 0.5 * np.pi           
#     wy = 2 * np.pi  
#     return np.array([
#         -Ax * wx**2 * np.cos(wx*t),
#         -Ay * wy**2 * np.sin(wy*t),
#         -0*Az * w**2 * np.sin(w*t)
#     ])


TIP = np.array([0.0, 0.0, 0.265])

Ax = 0.10        # meters (X amplitude)
Az = 0.06        # meters (Z amplitude)
w  = 2*np.pi     # rad/s

def pos_traj_circle_xy(t):
    x0, y0, z0 = TIP
    x = x0 + Ax * np.sin(w*t)            # x(0)=x0
    z = z0 + Az * (1.0 - np.cos(w*t))    # z(0)=z0
    return np.array([x, y0, z])

def vel_traj_circle_xy(t):
    xdot = Ax * w * np.cos(w*t)
    zdot = Az * w * np.sin(w*t)
    return np.array([xdot, 0.0, zdot])

def acc_traj_circle_xy(t):
    xddot = -Ax * w**2 * np.sin(w*t)
    zddot =  Az * w**2 * np.cos(w*t)
    return np.array([xddot, 0.0, zddot])



def controller(model, data,vel_d,acc_d):
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))
    Kp_null = np.asarray([1] * model.nv)

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)
    n = 6
    F = np.zeros((2*n,2*n))
    m = 6
    F = np.zeros((2*m,2*m))
    F[:m,m:] = np.eye(m,m)
    G = np.zeros((2*m,m))
    G[m:,:] = np.eye(m)
    e = 0.05
    Pe = linalg.block_diag(np.eye(m) / e, np.eye(m) ).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m) )

    
    pinv_B = np.linalg.pinv(B)
    site_name = "ee"
    site_id = model.site(site_name).id


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
    M = M * np.random.normal(1.0, 0.01, size=M.shape)  # Add some noise to the inertia matrix
    dJ_dt = compute_jacobian_derivative(model, data, site_id)

    Mx_inv = jac @ M_inv @ jac.T
    e = e
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    Jbar = M_inv @ jac.T @ Mx

    # define decision variables
    nu = 9
    nq = model.nq
    u = cp.Variable(shape=(nu, 1))
    qdd = cp.Variable(shape=(nq, 1))
    dl = cp.Variable(shape=(1, 1))
    mu = cp.Variable(shape=(6, 1))
    twist[3:] = 0.0


    # error and Lyapunov function for CLF
    eta = np.concatenate((-twist,jac @ data.qvel - vel_d))
    V = eta.T @ Pe @ eta

    C, g = get_coriolis_and_gravity(model, data)

    # static forces
    statics = (g + data.qfrc_spring)
    dq = data.qvel.reshape(-1,1)
    # stat = cp.square(cp.norm(B @ u - statics.reshape(-1,1)))

    # selection matrix for compression/extension actuators
    sel = np.ones((nu,1))
    sel[[2,5,8]] = 0.0

    # # nullspace projection
    # ddq = Kp_null * (- data.qpos) - 4*Kd_null * data.qvel
    # null = 0.1*B @ pinv_B @ ddq.reshape(-1,1)
    # damp = cp.square(cp.norm(B @ u - null))
    


    Jbar = M_inv @ jac.T @ Mx

    # Vdot for our main CLF
    dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd - acc_d)
    # dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd)


    yd = np.copy(jac @ data.qvel)

    vel_d = vel_d.flatten()
    acc_d = acc_d.flatten()

    # twist = y_d - y
    # objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - (K_task @ twist  - D_task @ (jac @ data.qvel)))) + 0.2 * cp.square(cp.norm(qdd)) + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl)) 

    objective = cp.Minimize(cp.square(cp.norm((dJ_dt @ dq + jac @ qdd - acc_d) - (1500 * twist  + 20 * (vel_d - jac @ data.qvel)))) + 0.2 * cp.square(cp.norm(qdd)) + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl) ) 

    # objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - (acc_d + K_task @ twist  + D_task @ (vel_d - jac @ data.qvel)))) + 0.2 * cp.square(cp.norm(qdd))  + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl)) 
#
    # objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - (1000 * twist  - 20 * (jac @ data.qvel)))) + 0.2 * cp.square(cp.norm(qdd))  + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl)) 


    constraints = [ dV <= - 2/0.05 * V + 0.01*dl, 
                      pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u,
                    -25*sel <= u,
                    25*np.ones((nu,1)) >= u]
    

    prob = cp.Problem(objective=objective, constraints=constraints)
    


    twist[3:] = 0

    task_error = np.linalg.norm(dx)
        
    
    try:
        prob.solve(verbose=False)
        data.ctrl = np.squeeze(B @ u.value)
        V = V.value
        
        # C, g = get_coriolis_and_gravity(model, data)
        # Cy = Jbar.T @ C @ data.qvel - Mx @ dJ_dt @ data.qvel
        # # ydd = acc_d + 1000 * twist + 20 * (vel_d - jac @ data.qvel)
        # ydd = 500 * twist - 20 * (jac @ data.qvel)

        # tau = jac.T @ (Mx @ ydd + Cy) + g + data.qfrc_passive
        # data.ctrl = B @ pinv_B @ tau

        
    except:
        # print(f"failed convergence\n")
        pass

    return V, task_error, q, dq, yd



def simulate_model():
    model_path = Path("./mujoco_models/helix") / (str(model_name) + str(".xml"))
    print(f"Loading model from {model_path}")
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    model.jnt_stiffness[:] = stiffness
    
    model.dof_damping[:] = 0.2
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    data.qpos[2] = 0.0
    model.jnt_range[range(2,len(data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(data.qpos),3)]
    model.jnt_stiffness[range(2,len(data.qpos),3)] = 50
    # model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

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

    y_dd_real = []
    y_dd_des = []

    mocap_id = model.body("target").mocapid[0]
    


    last_ctrl = time.time()


    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start = time.time()
        while viewer.is_running():
            step_start = time.time()
            first_time = time.time()
            # controller(model,data,acc_d)
            mujoco.mj_step(model, data)

            
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

            # # Trajectory tracking
            xd = pos_traj_circle_xy(t)          # pick one: circle_xy / line / lissajous / spline
            data.mocap_pos[mocap_id] = xd
            # print("xd is: ",xd)
            v_lin = vel_traj_circle_xy(t)   
            a_lin = acc_traj_circle_xy(t)        # shape (3,)
            vel_d = np.hstack([v_lin, [0, 0, 0]]) 
            acc_d = np.hstack([a_lin, [0, 0, 0]])  # shape (6,)


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            V, task_error, q, dq, yd = controller(model, data,vel_d,acc_d)

            # # ---- Early-stop convergence check ----
            # # ---- Convergence debug ----
            # if len(q_vel) > 1:
            #     dq_diff_norm = np.linalg.norm(q_vel[-1] - q_vel[-2]) / dt
            #     print(f"t={data.time:.3f} s | ||dq - dq_prev||/dt = {dq_diff_norm:.3e}")

            #     if dq_diff_norm < 1e-5:   # your 10e-6 threshold
            #         print("✅ Converged, stopping simulation")
            #         break


            # print("ydd is: ",np.shape(twist))
            V_log.append(V)
            task_error_log.append(task_error)
            q_vel.append(dq.squeeze().copy())
            q_pos.append(q.squeeze().copy())
            y_dd_des.append
            time_log.append(t)
            t += dt


            # ---- Convergence debug ----
            if t > 2*np.pi/(0.5*np.pi):   # your 10e-6 threshold
                    print("✅ Stopping simulation")
                    break

                
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                # print(time_until_next_step)
                time.sleep(time_until_next_step)
            # print(time.time() - first_time)
    print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")
    return V_log, task_error_log, q_vel, q_pos, time_log, sim_ts


if __name__ == "__main__":
    # Simulate the model
    V_log, task_error_log, q_vel, q_pos, time_log, sim_ts = simulate_model()
    q_vel = np.array(q_vel)
    q_pos = np.array(q_pos)

    # print(task_error_log[-1])
    # --- Save data to CSV ---
    # Flatten q_pos and q_vel into labeled columns
    q_pos_df = pd.DataFrame(q_pos, columns=[f"q{i}" for i in range(q_pos.shape[1])])
    q_vel_df = pd.DataFrame(q_vel, columns=[f"qdot{i}" for i in range(q_vel.shape[1])])

    # # Build main DataFrame
    # df = pd.DataFrame({
    #     "time": time_log,
    #     "Lyapunov": V_log,
    #     "task_error": task_error_log
    # })
    # # Save to CSV
    # df.to_csv("simulation_results.csv", index=False)
    # print("Saved all results to simulation_results.csv")


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

 


    # Control inputs plot (recover 9 actuator commands from 36 generalized torques)
    ctrl = np.array(sim_ts["ctrl"])                  # shape: (T, 36)
    u_rec = (np.linalg.pinv(B) @ ctrl.T).T           # shape: (T, 9)  approximate actuator commands
    num_actuators = u_rec.shape[1]
    control_limit = 20.0

    # fig, axs = plt.subplots(num_actuators, 1, figsize=(10, 8), sharex=True)
    # if num_actuators == 1:
    #     axs = [axs]  # make iterable

    # for i in range(num_actuators):
    #     axs[i].plot(time_log, u_rec[:, i], label=f"u{i+1}")
    #     axs[i].axhline(control_limit,  linestyle='--', linewidth=3)
    #     axs[i].axhline(-control_limit, linestyle='--', linewidth=3)
    #     axs[i].legend()
    #     axs[i].grid(True)

    # axs[-1].set_xlabel("Time (s)")
    # fig.suptitle("Control Effort (Nm)")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])


    plt.show()

        # Build main DataFrame
    df = pd.DataFrame({
        "time": time_log,
        "Lyapunov": V_log,
        "task_error": task_error_log,
    })

    # Add nine control inputs as separate columns
    for i in range(u_rec.shape[1]):
        df[f"u{i+1}"] = u_rec[:, i]

    # Save to CSV
    df.to_csv("simulation_results.csv", index=False)
    print("Saved all results to simulation_results.csv")


    
#-----------------------Target Tracking---------------------------------

# import argparse
# import matplotlib.pyplot as plt
# import os
# os.environ["MUJOCO_GL"] = "egl"   # or comment this line out entirely
# import mujoco
# import mujoco.viewer
# import numpy as np
# import os
# from pathlib import Path
# import time
# from scipy import linalg
# import cvxpy as cp
# from scipy.linalg import eigh
# import pandas as pd
# from matplotlib import rcParams




# # # Configure MuJoCo to use the EGL rendering backend (requires GPU)
# os.environ["MUJOCO_GL"] = "egl"


# model_name = f"helix_control"

# # Cartesian impedance control gains.
# impedance_pos = np.asarray([50.0, 50.0, 50.0])  # [N/m]
# impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# # Joint impedance control gains.


# # Damping ratio for both Cartesian and joint impedance control.
# damping_ratio = 1.0

# # Gains for the twist computation. These should be between 0 and 1. 0 means no
# # movement, 1 means move the end-effector to the target in one integration step.
# Kpos: float = 0.95

# # Gain for the orientation component of the twist computation. This should be
# # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# # orientation in one integration step.
# Kori: float = 0.95
# stiffness = 0.01


# f_ctrl = 2000.0 
# T = 1
# w = 2 * np.pi / T

# B = np.zeros((36, 9))
# for i in range(3):  # Iterate over u1, u2, u3 blocks
#     for j in range(4):  # Repeat each block 4 times
#         row_start = i * 12 + j * 3  # Compute row index
#         col_start = i * 3  # Compute column index
#         B[row_start:row_start+3, col_start:col_start+3] = np.eye(3)  # Assign identity

# print(B)

# def get_coriolis_and_gravity(model, data):
#     """
#     Calculate the Coriolis matrix and gravity vector for a MuJoCo model

#     Parameters:
#         model: MuJoCo model object
#         data: MuJoCo data object

#     Returns:
#         C: Coriolis matrix (nv x nv)
#         g: Gravity vector (nv,)
#     """
#     nv = model.nv  # number of degrees of freedom

#     # Calculate gravity vector
#     g = np.zeros(nv)
#     dummy = np.zeros(nv,)
#     mujoco.mj_factorM(model, data)  # Compute sparse M factorization
#     mujoco.mj_rne(model, data, 0, dummy)  # Run RNE with zero acceleration and velocity
#     g = data.qfrc_bias.copy()

#     # Calculate Coriolis matrix
#     C = np.zeros((nv, nv))
#     q_vel = data.qvel.copy()

#     # Compute each column of C using finite differences
#     eps = 1e-6
#     for i in range(nv):
#         # Save current state
#         vel_orig = q_vel.copy()

#         # Perturb velocity
#         q_vel[i] += eps
#         data.qvel = q_vel

#         # Calculate forces with perturbed velocity
#         mujoco.mj_rne(model, data, 0, dummy)
#         tau_plus = data.qfrc_bias.copy()

#         # Restore original velocity
#         q_vel = vel_orig
#         data.qvel = q_vel

#         # Compute column of C using finite difference
#         C[:, i] = (tau_plus - data.qfrc_bias) / eps

#     return C, g


# def compute_jacobian_derivative(model, data, site_id, h=1e-6):
#     """
#     Compute the time derivative of the Jacobian in MuJoCo.
    
#     Parameters:
#     - model: The MuJoCo model (mjModel).
#     - data: The MuJoCo data structure (mjData).
#     - jac_func: Function to compute the Jacobian (e.g., mj_jacBody or mj_jacSite).
#     - h: Small positive step for numerical differentiation.
    
#     Returns:
#     - Jdot: The time derivative of the Jacobian.
#     """
#     # Step 1: Update kinematics
#     mujoco.mj_kinematics(model, data)
#     mujoco.mj_comPos(model, data)
    
#     # Step 2: Compute the initial Jacobian
#     J = np.zeros((6, model.nv))  # Assuming a 6xnv Jacobian for full spatial representation
#     mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)
    
#     # Step 3: Integrate position using velocity
#     qpos_backup = np.copy(data.qpos)  # Backup original qpos
#     mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
    
#     # Step 4: Update kinematics again
#     mujoco.mj_kinematics(model, data)
#     mujoco.mj_comPos(model, data)
    
#     # Step 5: Compute the new Jacobian
#     Jh = np.zeros((6, model.nv))
#     mujoco.mj_jacSite(model, data, Jh[:3], Jh[3:], site_id)
    
#     # Step 6: Compute Jdot
#     Jdot = (Jh - J) / h
    
#     # Step 7: Restore qpos
#     data.qpos[:] = qpos_backup
    
#     return Jdot


# def finite_difference_Mdot(model, data, delta_t=1e-6):
#     # Backup current state
#     qpos_orig = data.qpos.copy()
#     qvel_orig = data.qvel.copy()

#     # Compute M at current q
#     M0 = np.zeros((model.nv, model.nv))
#     mujoco.mj_fullM(model, M0, data.qM)

#     # Integrate qpos forward by dq * delta_t
#     mujoco.mj_integratePos(model, data.qpos, data.qvel, delta_t)
#     mujoco.mj_kinematics(model, data)
#     mujoco.mj_comPos(model, data)
#     mujoco.mj_forward(model, data)

#     # Compute M at new q
#     M1 = np.zeros((model.nv, model.nv))
#     mujoco.mj_fullM(model, M1, data.qM)

#     # Restore qpos
#     data.qpos[:] = qpos_orig
#     data.qvel[:] = qvel_orig
#     mujoco.mj_kinematics(model, data)
#     mujoco.mj_comPos(model, data)
#     mujoco.mj_forward(model, data)

#     # Approximate Mdot
#     Mdot = (M1 - M0) / delta_t
#     return Mdot

# def pos_traj_circle_xy(t):
#     """Circle in XY, fixed Z."""
#     center = np.array([0, 0.00, 0.7])   # near your current target
#     Ax, Ay, Az = 0.435, 0, 0.435           # radii (m)
#     theta = np.pi/6 * 3                       
#     return center + np.array([Ax*np.cos(theta), 0, Ay*np.sin(theta)])

# # def pos_traj_circle_xy(t):
# #     """Circle in XY, fixed Z."""
# #     center = np.array([0, 0.00, 0.35])   # near your current target
# #     Ax, Ay, Az = 0.1, 0.1, 0.1           # radii (m); Az=0 keeps Z fixed
# #     wx = 0.5 * np.pi           
# #     wy = 2 * np.pi                            # rad/s
# #     return center + np.array([Ax*np.cos(wx*t), Ay*np.sin(wy*t), 0*Az*np.sin(w*t)])

# # def vel_traj_circle_xy(t):
# #     """Velocity for circular XY trajectory, fixed Z."""
# #     Ax, Ay, Az = 0.1, 0.1, 0.1
# #     wx = 0.5 * np.pi           
# #     wy = 2 * np.pi  
# #     return np.array([
# #         -Ax * wx * np.sin(wx * t),
# #          Ay * wy * np.cos(wy * t),
# #          0*Az * w * np.cos(w * t)
# #     ])

# # def acc_traj_circle_xy(t):
# #     """Acceleration for circular XY trajectory, fixed Z."""
# #     Ax, Ay, Az = 0.1, 0.1, 0.1
# #     wx = 0.5 * np.pi           
# #     wy = 2 * np.pi  
# #     return np.array([
# #         -Ax * wx**2 * np.cos(wx*t),
# #         -Ay * wy**2 * np.sin(wy*t),
# #         -0*Az * w**2 * np.sin(w*t)
# #     ])



# def controller(model, data):
#     jac = np.zeros((6, model.nv))
#     twist = np.zeros(6)
#     site_quat = np.zeros(4)
#     site_quat_conj = np.zeros(4)
#     error_quat = np.zeros(4)
#     M_inv = np.zeros((model.nv, model.nv))
#     M = np.zeros((model.nv, model.nv))
#     # Kp_null = np.asarray([1] * model.nv)

#     mocap_name = "target"
#     mocap_id = model.body(mocap_name).mocapid[0]

#     # Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)
#     m = 6
#     F = np.zeros((2*m,2*m))
#     F[:m,m:] = np.eye(m,m)
#     G = np.zeros((2*m,m))
#     G[m:,:] = np.eye(m)
#     e = 0.05
#     Pe = linalg.block_diag(np.eye(m) / e, np.eye(m) ).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m) )

    
#     pinv_B = np.linalg.pinv(B)
#     site_name = "ee"
#     site_id = model.site(site_name).id


#     dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
#     twist[:3] = dx 
#     mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
#     mujoco.mju_negQuat(site_quat_conj, site_quat)
#     mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
#     mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
#     twist[3:] *= Kori 

#     q = data.qpos
#     mujoco.mj_kinematics(model,data)
#     mujoco.mj_comPos(model,data)
#     mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
#     # Compute the task-space inertia matrix.
#     mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
#     mujoco.mj_fullM(model, M, data.qM)
#     M = M * np.random.normal(1.0, 0.01, size=M.shape)  # Add some noise to the inertia matrix
#     dJ_dt = compute_jacobian_derivative(model, data, site_id)

    

#     # define decision variables
#     nu = 9
#     nq = model.nq
#     u = cp.Variable(shape=(nu, 1))
#     qdd = cp.Variable(shape=(nq, 1))
#     dl = cp.Variable(shape=(1, 1))
#     # mu = cp.Variable(shape=(6, 1))
#     twist[3:] = 0.0


#     # error and Lyapunov function for CLF
#     eta = np.concatenate((-twist,jac @ data.qvel))
#     V = eta.T @ Pe @ eta

#     C, g = get_coriolis_and_gravity(model, data)

#     # static forces
#     # statics = (g + data.qfrc_spring)
#     dq = data.qvel.reshape(-1,1)
#     # stat = cp.square(cp.norm(B @ u - statics.reshape(-1,1)))

#     # selection matrix for compression/extension actuators
#     sel = np.ones((nu,1))
#     sel[[2,5,8]] = 0.0

#     # # nullspace projection
#     # ddq = Kp_null * (- data.qpos) - 4*Kd_null * data.qvel
#     # null = 0.1*B @ pinv_B @ ddq.reshape(-1,1)
#     # damp = cp.square(cp.norm(B @ u - null))
    
#     # Vdot for our main CLF
#     dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd)
#     # dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd)


#     yd = np.copy(jac @ data.qvel)

#     # ---------------------------------------------------------
#     # Task critical damping https://www.sciencedirect.com/topics/engineering/critical-damping
#     Mx_inv = jac @ M_inv @ jac.T
#     if abs(np.linalg.det(Mx_inv)) >= 1e-2:
#         M_task = np.linalg.inv(Mx_inv)
#     else:
#         M_task = np.linalg.pinv(Mx_inv, rcond=1e-2)
    
#     # Choose desired modal frequencies (eigenvalues of M^{-1} K)

#     omega_sq = np.diag([100, 100, 100, 100, 100, 100])  * 10 # omega^2

#     # Compute eigenvectors of M (to use as modal basis)
#     eigvals, P = eigh(M_task)  # P: eigenvectors of M
#     P_inv = np.linalg.inv(P)

#     # Construct K in modal coordinates, then transform to original coordinates
#     K_modal = omega_sq
#     K_task = P_inv.T @ K_modal @ P_inv  # K = P^{-T} * Omega^2 * P^{-1}

#     # Choose damping ratios (zeta = 1 for critical damping)
#     zeta = 1.0
#     D_modal = 2 * zeta * np.sqrt(omega_sq)  # 2ζω

#     # Step 5: Construct C in modal coordinates and transform back
#     D_task = P_inv.T @ D_modal @ P_inv
#     # ---------------------------------------------------------

#     objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - (K_task @ twist  - D_task @ (jac @ data.qvel)))) + 0.2 * cp.square(cp.norm(qdd)) + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl)) 
#     # objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - (500 * twist  - 20 * (jac @ data.qvel)))) + 0.2 * cp.square(cp.norm(qdd))  + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl))

#     constraints = [ dV <= - 2/e * V + 0.01*dl, 
#                       pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u,
#                     -20*sel <= u,
#                     20*np.ones((nu,1)) >= u]
    

#     prob = cp.Problem(objective=objective, constraints=constraints)
    


#     twist[3:] = 0

#     task_error = np.linalg.norm(dx)
        
    
#     try:
#         prob.solve(verbose=False)
#         data.ctrl = np.squeeze(B @ u.value)
#         V = V.value
        
#         # Mx_inv = jac @ M_inv @ jac.T
#         # if abs(np.linalg.det(Mx_inv)) >= 1e-2:
#         #     Mx = np.linalg.inv(Mx_inv)
#         # else:
#         #     Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
#         # Jbar = M_inv @ jac.T @ Mx
#         # C, g = get_coriolis_and_gravity(model, data)
#         # Cy = Jbar.T @ C @ data.qvel - Mx @ dJ_dt @ data.qvel
#         # ydd =  500 * twist + 20 * (- jac @ data.qvel)
#         # tau = jac.T @ (Mx @ ydd + Cy) + g + data.qfrc_passive
#         # data.ctrl = B @ pinv_B @ tau

        
#     except:
#         # print(f"failed convergence\n")
#         pass

#     return V, task_error, q, dq, yd



# def simulate_model():
#     model_path = Path("./mujoco_models/helix") / (str(model_name) + str(".xml"))
#     print(f"Loading model from {model_path}")
#     # Load the model and data
#     model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
#     model.jnt_stiffness[:] = stiffness
    
#     model.dof_damping[:] = 0.2
#     model.opt.gravity = (0, 0, -9.81)
#     data = mujoco.MjData(model)

#     data.qpos[2] = 0.0
#     model.jnt_range[range(2,len(data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(data.qpos),3)]
#     model.jnt_stiffness[range(2,len(data.qpos),3)] = 50
#     # model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

#     sim_ts = dict(
#         ts=[],
#         base_pos=[],
#         base_vel=[],
#         base_acc=[],
#         base_force=[],
#         base_torque=[],
#         q=[],
#         qvel=[],
#         ctrl=[],
#         actuator_force=[],
#         qfrc_fluid=[],
#         q_des=[],
#     )
#     time_last_ctrl = 0.0
#     q_des = np.ones(data.qpos.shape[0])*0.2
#     # print(np.shape(q_des))

#     V_log = []
#     task_error_log = []
#     q_vel = []
#     q_pos = []
#     time_log = []
#     t = 0.0
#     dt = model.opt.timestep

#     y_dd_real = []
#     y_dd_des = []

#     mocap_id = model.body("target").mocapid[0]
    


#     last_ctrl = time.time()


#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         sim_start = time.time()
#         while viewer.is_running():
#             step_start = time.time()
#             first_time = time.time()
#             # controller(model,data,acc_d)
#             mujoco.mj_step(model, data)

            
#             sim_ts["ts"].append(data.time)
#             # extract the sensor data
#             sim_ts["base_pos"].append(data.sensordata[:3].copy())
#             sim_ts["base_vel"].append(data.sensordata[3:6].copy())
#             sim_ts["base_acc"].append(data.sensordata[6:9].copy())
#             sim_ts["base_force"].append(data.sensordata[9:12].copy())
#             sim_ts["base_torque"].append(data.sensordata[12:15].copy())
#             sim_ts["q"].append(data.qpos.copy())
#             sim_ts["qvel"].append(data.qvel.copy())
#             sim_ts["ctrl"].append(data.ctrl.copy())
#             sim_ts["actuator_force"].append(data.actuator_force.copy())
#             sim_ts["qfrc_fluid"].append(data.qfrc_fluid.copy())
#             sim_ts["q_des"].append(q_des.copy())

#             # # Trajectory tracking
#             xd = pos_traj_circle_xy(t)          # pick one: circle_xy / line / lissajous / spline
#             data.mocap_pos[mocap_id] = xd
#             # print("xd is: ",xd)




#             # Pick up changes to the physics state, apply perturbations, update options from GUI.
#             viewer.sync()

#             V, task_error, q, dq, yd = controller(model, data)

#             # ---- Early-stop convergence check ----
#             # ---- Convergence debug ----
#             if len(q_vel) > 1:
#                 dq_diff_norm = np.linalg.norm(q_vel[-1] - q_vel[-2]) / dt
#                 print(f"t={data.time:.3f} s | ||dq - dq_prev||/dt = {dq_diff_norm:.3e}")

#                 if dq_diff_norm < 1e-5:   # your 10e-6 threshold
#                     print("✅ Converged, stopping simulation")
#                     break


#             # print("ydd is: ",np.shape(twist))
#             V_log.append(V)
#             task_error_log.append(task_error)
#             q_vel.append(dq.squeeze().copy())
#             q_pos.append(q.squeeze().copy())
#             y_dd_des.append
#             time_log.append(t)
#             t += dt

                
#             # Rudimentary time keeping, will drift relative to wall clock.
#             time_until_next_step = model.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 # print(time_until_next_step)
#                 time.sleep(time_until_next_step)
#             # print(time.time() - first_time)
#     print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")
#     return V_log, task_error_log, q_vel, q_pos, time_log, sim_ts


# if __name__ == "__main__":
#     # Simulate the model
#     V_log, task_error_log, q_vel, q_pos, time_log, sim_ts = simulate_model()
#     q_vel = np.array(q_vel)
#     q_pos = np.array(q_pos)

#     # print(task_error_log[-1])
#     # --- Save data to CSV ---
#     # Flatten q_pos and q_vel into labeled columns
#     q_pos_df = pd.DataFrame(q_pos, columns=[f"q{i}" for i in range(q_pos.shape[1])])
#     q_vel_df = pd.DataFrame(q_vel, columns=[f"qdot{i}" for i in range(q_vel.shape[1])])

#     # # Build main DataFrame
#     # df = pd.DataFrame({
#     #     "time": time_log,
#     #     "Lyapunov": V_log,
#     #     "task_error": task_error_log
#     # })
#     # # Save to CSV
#     # df.to_csv("simulation_results.csv", index=False)
#     # print("Saved all results to simulation_results.csv")


#     # #Lyapunov Function Over Time – shows how stability evolves.
#     # plt.figure()
#     # plt.plot(time_log, V_log, label="Lyapunov Function V")
#     # plt.xlabel("Time (s)")
#     # plt.ylabel("Lyapunov Function V")
#     # plt.title("Lyapunov Function Over Time")
#     # plt.grid(True)
#     # plt.legend()  # <--- Add legend

#     #End-Effector Position Error – checks task-space convergence.
#     plt.figure()
#     plt.plot(time_log, task_error_log, label="‖x_desired - x_actual‖")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Task-Space Position Error (m)")
#     plt.title("End-Effector Position Error Over Time")
#     plt.grid(True)
#     plt.legend()

#     # # Joint Angles (qq) 
#     # actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
#     # fig, axs = plt.subplots(len(actuated_indices), 1, figsize=(10, 8), sharex=True)

#     # for i, idx in enumerate(actuated_indices):
#     #     axs[i].plot(time_log, q_pos[:, idx], label=f"Joint {idx}")
#     #     axs[i].legend()
#     #     axs[i].grid(True)

#     # axs[-1].set_xlabel("Time (s)")
#     # fig.suptitle("Actuated Joint Angles Over Time (Rad)")
#     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#     # # Joint Velocities (q̇) 
#     # actuated_indices = np.where(np.any(B != 0, axis=0))[0]  # shape: (n_actuated,)
#     # fig, axs = plt.subplots(len(actuated_indices), 1, figsize=(10, 8), sharex=True)

#     # for i, idx in enumerate(actuated_indices):
#     #     axs[i].plot(time_log, q_vel[:, idx], label=f"Joint {idx}")
#     #     axs[i].legend()
#     #     axs[i].grid(True)

#     # axs[-1].set_xlabel("Time (s)")
#     # fig.suptitle("Actuated Joint Velocities Over Time (Rad/s)")
#     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

 


#     # Control inputs plot (recover 9 actuator commands from 36 generalized torques)
#     ctrl = np.array(sim_ts["ctrl"])                  # shape: (T, 36)
#     u_rec = (np.linalg.pinv(B) @ ctrl.T).T           # shape: (T, 9)  approximate actuator commands
#     num_actuators = u_rec.shape[1]
#     control_limit = 20.0

#     # fig, axs = plt.subplots(num_actuators, 1, figsize=(10, 8), sharex=True)
#     # if num_actuators == 1:
#     #     axs = [axs]  # make iterable

#     # for i in range(num_actuators):
#     #     axs[i].plot(time_log, u_rec[:, i], label=f"u{i+1}")
#     #     axs[i].axhline(control_limit,  linestyle='--', linewidth=3)
#     #     axs[i].axhline(-control_limit, linestyle='--', linewidth=3)
#     #     axs[i].legend()
#     #     axs[i].grid(True)

#     # axs[-1].set_xlabel("Time (s)")
#     # fig.suptitle("Control Effort (Nm)")
#     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])


#     plt.show()

#         # Build main DataFrame
#     df = pd.DataFrame({
#         "time": time_log,
#         "Lyapunov": V_log,
#         "task_error": task_error_log,
#     })

#     # Add nine control inputs as separate columns
#     for i in range(u_rec.shape[1]):
#         df[f"u{i+1}"] = u_rec[:, i]

#     # Save to CSV
#     df.to_csv("simulation_results.csv", index=False)
#     print("Saved all results to simulation_results.csv")


    





    





    


