import time

import mujoco
import mujoco.viewer
import numpy as np
import os
import pandas as pd

# Import controllers
from controllers import (ControllerResult, IDCLFQPController, ImpedanceController, 
                        ImpedanceQPController, MPCController, CLFQPController, OSCController)

# Import Robot class
from robot import Robot


# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"




def circular_trajectory(t, model_name):
    """
    Circular trajectory through the 4 given points.
    One full revolution in time T.
    """
    if model_name == 'tendon':
        L = 0.24
        h = L
    
    elif model_name == 'helix':
        L = 0.45
        h = 0.7

    elif model_name == 'spirob':
        L = 0.45
        h = L

    # Circle parameters
    # cx, cy, cz = 3*L/4 * np.cos(np.pi/4), 0, -3*L/4 * np.sin(np.pi/4) + h
    # r = L/4

    # Angle
    omega = 0.25 * np.pi 
    theta = omega * t

    a = L/3
    b = L/6
    phi = np.pi/4
    x1 = a * np.cos(theta)
    z1 = b * np.sin(theta) - (L-b)

    # Position
    x = x1 * np.cos(phi) - z1 * np.sin(phi)
    y = 0.0
    z = x1 * np.sin(phi) + z1 * np.cos(phi) + h

    # Velocity
    xd = -a * omega * np.sin(theta) * np.cos(phi) - b * omega * np.cos(theta) * np.sin(phi)
    yd = 0.0
    zd =  -a * omega * np.sin(theta) * np.sin(phi)+ b * omega * np.cos(theta) * np.cos(phi) 

    # Acceleration
    xdd = -a * omega**2 * np.cos(theta) * np.cos(phi) + b * omega**2 * np.sin(theta) * np.sin(phi)
    ydd = 0.0
    zdd = -a * omega**2 * np.cos(theta) * np.sin(phi) - b * omega**2 * np.sin(theta) * np.cos(phi)


    pos = np.array([x, y, z])
    vel = np.array([xd, yd, zd])
    acc = np.array([xdd, ydd, zdd])

    return omega,{"pos": pos, "vel": vel, "acc": acc}

def set_target(target_pos, model_name):
    if model_name == 'tendon':
        L = 0.24
        h = L
    elif model_name == 'helix':
        L = 0.45
        h = 0.7
    elif model_name == 'spirob':
        L = 0.48
        h = L

    # # Position
    theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

    a = L/3
    b = L/8
    phi = np.pi/4
    x1 = a * np.cos(theta)
    z1 = b * np.sin(theta) - (L-b)

    # Position
    x = x1 * np.cos(phi) - z1 * np.sin(phi)
    y = 0.0
    z = x1 * np.sin(phi) + z1 * np.cos(phi) + h

    pos1 = np.array([x[0], y, z[0]])
    pos2 = np.array([x[1], y, z[1]])
    pos3 = np.array([x[2], y, z[2]])
    pos4 = np.array([x[3], y, z[3]])

    targets = {
        'pos1': pos1,
        'pos2': pos2,
        'pos3': pos3,
        'pos4': pos4
    }

    return targets[target_pos]


def _create_log_arrays(num_steps, control_scheme, experiment, nu):
    """Pre-allocate logging arrays based on what we actually need"""
    logs = {
        'time': np.zeros(num_steps),
        'sim_time': np.zeros(num_steps), 
        'task_error': np.zeros(num_steps),
        'ctrl_time': np.zeros(num_steps),
        'u': np.zeros((num_steps, nu))
    }
    
    if control_scheme == 'id_clf_qp':
        logs['V'] = np.zeros(num_steps)
    
    if experiment == 'tracking':
        logs['x'] = np.zeros((num_steps, 3))
        logs['xd'] = np.zeros((num_steps, 3))
    
    return logs

def _log_simulation_data(logs, log_idx, data, control_scheme, experiment, result, t, ctrl_t, target):
    """Log simulation data into pre-allocated arrays."""
    logs['time'][log_idx] = t
    logs['ctrl_time'][log_idx] = ctrl_t
    logs['sim_time'][log_idx] = data.time
    logs['task_error'][log_idx] = result.task_error
    logs['u'][log_idx] = result.control_input.squeeze()

    if control_scheme == 'id_clf_qp' and result.lyapunov_value is not None:
        logs['V'][log_idx] = result.lyapunov_value
        
    if experiment == 'tracking':
        logs['x'][log_idx] = data.site("ee").xpos
        logs['xd'][log_idx] = target["pos"]


def simulate_model(headless=False, control_scheme=None, target_pos=None, controller=None, experiment=None, model_name=None, sim_duration=10.0):
    """Run physics simulation with specified controller and robot."""
    
    robot = Robot(model_name)
    
    # Create controller instance based on control_scheme
    controller_map = {
        'id_clf_qp': IDCLFQPController(),
        'impedance': ImpedanceController(),
        'impedance_QP': ImpedanceQPController(), 
        'mpc': MPCController(),
        'clf_qp': CLFQPController(),
        'osc': OSCController()
    }
    controller = controller_map[control_scheme]
    
    # Initialize robot-specific simulation state
    robot.initialize_simulation_state()
        
    # Initialize solution cache
    previous_solution = None

    # Pre-allocate logging arrays for better performance
    dt = robot.model.opt.timestep
    max_steps = int(sim_duration / dt) + 100  # Add buffer
    log_frequency = 5  # Log every 5 steps
    max_log_steps = max_steps // log_frequency + 1
    
    logs = _create_log_arrays(max_log_steps, control_scheme, experiment, robot.nu)
    
    t = 0.0
    step_count = 0
    log_idx = 0

    # Main simulation loop
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = robot.model.camera("ortho_side").id

    try:
        while True:
            # Check if viewer mode and viewer is closed
            if not headless and not viewer.is_running():
                break
                
            # Get target based on experiment type
            if experiment == 'tracking':
                omega, target = circular_trajectory(t, model_name)
            else:  # experiment == 'set'
                target = set_target(target_pos, model_name)
            
            # Update robot kinematics
            robot.update_kinematics()
            
            # Compute target data (position, velocities, errors)
            target_vel, target_acc, twist = robot.compute_target_data(experiment, target)
            
            # Call controller directly 
            # t_ctrl_start = time.time()
            result = controller(robot, target_vel, target_acc, twist, previous_solution)
            t_ctrl = result.t_ctrl if result.t_ctrl is not None else 0.0
            # Update previous_solution from result
            previous_solution = result.previous_solution
            
            # Step physics
            robot.step_simulation()
            
            # Log data periodically
            if step_count % log_frequency == 0 and log_idx < max_log_steps:
                _log_simulation_data(logs, log_idx, robot.data, control_scheme, experiment, result, t, t_ctrl, target)
                log_idx += 1

            # Terminate after fixed duration (10 seconds)
            if t >= sim_duration:
                break
            
            step_count += 1
            t = robot.data.time
            
            # Update viewer if in viewer mode
            if not headless:
                viewer.sync()
            
            
                
    finally:
        if viewer is not None:
            viewer.close()

    
    # Trim arrays to actual logged data
    actual_logs = {}
    for key, arr in logs.items():
        if arr.ndim == 1:
            actual_logs[key] = arr[:log_idx]
        else:
            actual_logs[key] = arr[:log_idx]
    print(f"Average Control Time {np.mean(actual_logs['ctrl_time']):.6f} seconds")
    print(f"Simulation finished after {actual_logs['sim_time'][-1]} seconds")

    return actual_logs

def save_results(results, experiment, control_scheme, model_name, target_pos=None):
    """Save simulation results to CSV file."""
    
    # Extract data from results dictionary
    task_error_log = results['task_error']
    time_log = results['time']
    sim_time = results['sim_time']
    u_log = results['u']
    V_log = results.get('V')
    x_log = results.get('x')
    xd_log = results.get('xd')
    wall_time = results.get('wall_time')
    sim_time_total = results.get('sim_time_total')
    ctrl_time = results.get('ctrl_time')

    # ================================
    # Convert to numpy safely
    # ================================
    u_log = np.asarray(u_log)
    time_log = np.asarray(time_log)
    sim_time_csv = np.asarray(sim_time)
    error_log = np.asarray(task_error_log)

    if V_log is not None:
        V_log = np.asarray(V_log)

    if experiment == "tracking":
        x_log = np.asarray(x_log).squeeze().tolist()
        xd_log = np.asarray(xd_log).squeeze().tolist()

    # ================================
    # Build dataframe
    # ================================
    df = pd.DataFrame()

    df["time"] = time_log
    df["sim_time"] = sim_time_csv

    if V_log is not None:
        df["lyapunov_V"] = V_log

    if experiment == "tracking":
        # store vectors as strings
        df["x_log"] = [list(v) for v in x_log]
        df["xd_log"] = [list(v) for v in xd_log]

    df["task_error"] = error_log

    # control inputs
    for i in range(u_log.shape[1]):
        df[f"u{i}"] = u_log[:, i]

    # ================================
    # Create folder
    # ================================
    out_dir = f"results/{model_name}/{control_scheme}"
    os.makedirs(out_dir, exist_ok=True)

    if experiment == "set":
        csv_path = f"{out_dir}/{experiment}_{control_scheme}_{target_pos}.csv"
    else:
        csv_path = f"{out_dir}/{experiment}_{control_scheme}.csv"

    # ================================
    # Compute performance metrics
    # ================================
    if wall_time is not None and sim_time_total is not None:
        rtf = sim_time_total / wall_time
    else:
        rtf = None

    # ================================
    # Write CSV with metadata header
    # ================================
    with open(csv_path, "w") as f:

        f.write("# ===== Simulation Performance =====\n")

        if wall_time is not None:
            f.write(f"# wall_clock_time,{wall_time:.6f}\n")

        if sim_time_total is not None:
            f.write(f"# physics_time,{sim_time_total:.6f}\n")

        if rtf is not None:
            f.write(f"# real_time_factor,{rtf:.6f}\n")

        if ctrl_time is not None:
            f.write(f"# control_time,{np.mean(ctrl_time):.6f}\n")

        f.write("# ==================================\n\n")

        df.to_csv(f, index=False)

    print(f"Saved CSV to {csv_path}")
    return csv_path

def auto_frame_camera(model, data, cam_name="side_perp", scale=2.2):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        return

    extent = model.stat.extent
    center = model.stat.center

    distance = scale * extent

    # Position: straight along -Y
    model.cam_pos[cam_id] = np.array([
        center[0],
        center[1] - distance,
        center[2]
    ])

    # Look at robot center
    model.cam_target[cam_id] = center

    # FORCE ORTHOGRAPHIC
    model.cam_orthographic[cam_id] = 1

