#!/usr/bin/env python3
"""Standalone MuJoCo runner for experiments with the ID-CLF-QP controller.

This is a small, editable entry point for trying targets and trajectories without
changing ``run.py``.  It uses ``Robot`` (and therefore the same XML files and
robot configuration) and the repository's ID-CLF-QP formulation, solved by
the same DAQP backend used in ``spirob_zmq``.

Examples:
    python spirob_clf.py
    python spirob_clf.py --experiment set --target -0.30 0.05 0.52
    python spirob_clf.py --experiment tracking --beta 0 --omega 0.5
    python spirob_clf.py --robot helix --experiment set --tune-camera
    python spirob_clf.py --robot helix --experiment sequence --headless
    python spirob_clf.py --robot tendon --experiment sequence --headless
    python spirob_clf.py --headless

For a custom experiment, edit ``experiment_target`` below.  It returns a
position for set-point runs, or a dict with ``pos``, ``vel`` and ``acc`` for
tracking runs.  Keep the derivatives consistent with the position so the
controller receives a coherent reference.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
try:
    import daqp
except ImportError:  # Keep --help usable before the optional runtime dependency is installed.
    daqp = None

from controllers import ControllerResult
from robot import Robot
from sequence_targets import SEQUENCE_HOLD_SECONDS, SEQUENCE_TARGETS


SPIROB_ZMQ_TARGET = np.array([0.0, 0.2, 0.175])
SPIROB_TRACKED_SITE = "marker_seg_14_2"
SPIROB_ZMQ_BASE_HEIGHT = 0.125
SPIROB_ZMQ_SETTLE_TIME = 5.0
TRACKED_MARKER_RADIUS_M = 0.008
TRACKED_MARKER_RGBA = np.array([0.0, 1.0, 0.0, 1.0])
CAMERA_VIEW_DIR = Path("results/camera_views")
EPSILON = 0.06  # CLF convergence parameter; gamma = 1 / epsilon [1/s].
ROBOT_EPSILONS = {
    "tendon": 0.02,
    "spirob_horz": 0.06,
    "helix": 0.1,
}
BETA = None  # None uses the robot-specific value configured in robot.py.
SCENE_PATHS = Robot.MODEL_PATH_OVERRIDES
QP_REG_QDD = 2.0
QP_REG_U = 1.5
QP_REG_DELTA = 1000.0 
QP_REG_NULL = 0.0  # Weight on null-space acceleration damping.
NULL_DAMPING_GAIN = 0   # qdd_ref = -gain * N @ dq.
MOTOR_V_BUS = 24.0
MOTOR_RESISTANCE = 2.2


def configure_spirob_current_source(robot):
    """Use the ZMQ current-source actuator interpretation and XML bounds."""
    # This scene compiles to current-source motors, while Robot's generic
    # spirob_horz setup assumes dcmotor voltage inputs.  Reuse the scene's
    # actual ctrlrange and avoid applying dcmotor force-range bounds (which are
    # unset on these current-source actuators and therefore collapse to zero).
    robot.lower_bounds = robot.model.actuator_ctrlrange[:, 0].copy()
    robot.upper_bounds = robot.model.actuator_ctrlrange[:, 1].copy()
    robot.control_limits = (float(robot.lower_bounds[0]),
                            float(robot.upper_bounds[0]))
    robot.velocity_dependent_bounds = False
    robot.k_v = 1.0
    robot.k_e = 0.0
    robot.include_constraint_forces = True


class DAQP_IDCLFQPController:
    """Full inverse-dynamics CLF-QP, solved with the native DAQP backend."""

    def __init__(self, epsilon=EPSILON, beta=BETA):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.epsilon = float(epsilon)
        self.beta = float(beta)

    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        if daqp is None:
            raise RuntimeError("DAQP is required; install the project dependencies with `pip install -r requirements.txt`.")
        started = time.perf_counter()
        nv, nq, nu = robot.model.nv, robot.model.nq, robot.nu
        u_prev = (np.full(nu, -0.1) if previous_solution is None
                  else np.asarray(previous_solution["u"], dtype=float))
        # ZMQ refreshes dynamics with the last applied command before building
        # B and h; match that state here, including contact reactions.
        if robot.model_name == "spirob_horz":
            robot.data.ctrl[:] = u_prev
        else:
            robot.apply_control_input(u_prev)
        mujoco.mj_forward(robot.model, robot.data)
        robot.update_input_matrix()
        M = robot.get_mass_matrix()
        site_name = (SPIROB_TRACKED_SITE if robot.model_name == "spirob_horz"
                     else "ee")
        J = robot.get_jacobian(site_name=site_name)
        if robot.model_name == "spirob_horz":
            site_id = robot.model.site(site_name).id
            point = robot.data.site_xpos[site_id].copy()
            jac6 = np.zeros((6, robot.model.nv))
            dJ6 = np.zeros((6, robot.model.nv))
            mujoco.mj_jacSite(robot.model, robot.data, jac6[:3], jac6[3:], site_id)
            mujoco.mj_jacDot(robot.model, robot.data, dJ6[:3], dJ6[3:], point,
                             robot.model.site_bodyid[site_id])
            J, dJ = jac6[:robot.task_dim], dJ6[:robot.task_dim]
        else:
            dJ = robot.get_jacobian_derivative(
                site_name=site_name, J_precomputed=J)
        dq = robot.get_joint_velocities()
        # The finite-difference Jdot helper restores qpos but leaves derived
        # force fields at its perturbed state; refresh before reading h/passives.
        mujoco.mj_forward(robot.model, robot.data)
        h = robot.get_bias_forces() + robot.get_passive_forces()

        if nq != nv:
            raise ValueError("ID-CLF-QP currently requires nq == nv")
        eta = np.concatenate((-twist, J @ dq - target_vel))
        eta_T_PeG = eta @ robot.PeG
        # Match the ZMQ task-space CLF scaling. The kinetic-energy term keeps
        # its 1/2 factor when beta is nonzero.
        task_clf_coeff = 2.0 * eta_T_PeG @ J
        task_clf_const = float(eta @ robot.FTPe_PeF @ eta
                               + 2.0 * eta_T_PeG @ (dJ @ dq - target_acc))

        # Add d/dt (beta/2 * dq' M dq).  With MuJoCo's equation of motion
        # and the manipulator energy identity, this is beta*dq'*(B*u +
        # qfrc_passive + qfrc_constraint - grad(U_gravity)).
        data = robot.data
        qvel = data.qvel.copy()
        mujoco.mj_forward(robot.model, data)
        passive = data.qfrc_passive.copy()
        constraint = data.qfrc_constraint.copy()
        data.qvel[:] = 0.0
        mujoco.mj_forward(robot.model, data)
        gravity_gradient = data.qfrc_bias.copy()
        data.qvel[:] = qvel
        mujoco.mj_forward(robot.model, data)
        B = robot.B.copy()
        beta = self.beta

        task_V = float(eta @ robot.Pe @ eta)
        joint_kinetic_energy = 0.5 * float(dq @ M @ dq)
        V = task_V + beta * joint_kinetic_energy
        clf_u_coeff = beta * (dq @ B)
        clf_const = task_clf_const + beta * float(
            dq @ (passive + constraint - gravity_gradient))

        pinv_B = robot.pinv_B
        P = pinv_B @ M
        p = pinv_B @ h
        lb, ub = robot.get_control_bounds()
        if robot.model_name == "spirob_horz":
            actuator_velocity = robot.data.actuator_velocity.copy()
            voltage_lo = (-MOTOR_V_BUS - actuator_velocity) / MOTOR_RESISTANCE
            voltage_hi = (MOTOR_V_BUS - actuator_velocity) / MOTOR_RESISTANCE
            lb = np.maximum(lb, voltage_lo)
            ub = np.maximum(np.minimum(ub, voltage_hi), lb)

        # Full decision vector z = [qdd, u, delta], with inverse dynamics as
        # an explicit equality.
        i_qdd = slice(0, nq)
        i_u = slice(nq, nq + nu)
        i_dl = nq + nu
        nvar = nq + nu + 1
        H = np.zeros((nvar, nvar))
        g = np.zeros(nvar)
        H[i_qdd, i_qdd] = 2.0 * QP_REG_QDD * np.eye(nv)
        H[i_u, i_u] = 2.0 * QP_REG_U * np.eye(nu)
        H[i_dl, i_dl] = 2.0 * QP_REG_DELTA
        # Track the configured PD desired task acceleration.
        mu_des = (target_acc + robot.Kp * twist
                  + robot.Kd * (target_vel - J @ dq))
        task_acc_offset = dJ @ dq - mu_des
        H[i_qdd, i_qdd] += 2.0 * robot.reg_task_acc * (J.T @ J)
        g[i_qdd] += 2.0 * robot.reg_task_acc * (J.T @ task_acc_offset)

        # Penalize null-space acceleration error around qdd_ref=-k*N*dq,
        # matching the damping term in controllers/id_clf_qp.py.
        null_projector = np.eye(nv) - np.linalg.pinv(J) @ J
        null_projector = 0.5 * (null_projector + null_projector.T)
        qdd_null_ref = -NULL_DAMPING_GAIN * (null_projector @ dq)
        H[i_qdd, i_qdd] += (2.0 * QP_REG_NULL
                            * (null_projector.T @ null_projector))
        g[i_qdd] += (-2.0 * QP_REG_NULL
                     * (null_projector.T @ qdd_null_ref))

        clf_row = 0
        dyn_rows = slice(1, 1 + nu)
        upper_rows = slice(1 + nu, 1 + 2 * nu)
        lower_rows = slice(1 + 2 * nu, 1 + 3 * nu)
        A = np.zeros((1 + 3 * nu, nvar))
        b = np.empty(A.shape[0])
        A[clf_row, i_qdd] = task_clf_coeff
        A[clf_row, i_u] = clf_u_coeff
        A[clf_row, i_dl] = -1.0
        b[clf_row] = -V / self.epsilon - clf_const
        A[dyn_rows, i_qdd] = P
        A[dyn_rows, i_u] = -np.eye(nu)
        b[dyn_rows] = -p
        A[upper_rows, i_u] = np.eye(nu)
        b[upper_rows] = ub
        A[lower_rows, i_u] = -np.eye(nu)
        b[lower_rows] = -lb
        # DAQP receives lower/upper bounds per constraint row. Mark the inverse
        # dynamics rows as equalities; all other rows are upper-bound inequalities.
        lower = np.full(A.shape[0], -1e30, dtype=np.float64)
        lower[dyn_rows] = b[dyn_rows]
        sense = np.zeros(A.shape[0], dtype=np.intc)
        sense[dyn_rows] = 5  # DAQP equality constraint
        solve_options = {}
        if previous_solution is not None:
            try:
                solve_options["primal_start"] = np.concatenate((
                    np.asarray(previous_solution["qdd"], dtype=np.float64),
                    np.asarray(previous_solution["u"], dtype=np.float64),
                    np.asarray(previous_solution["dl"], dtype=np.float64),
                ))
            except (KeyError, TypeError, ValueError):
                pass
        z, _, exitflag, _ = daqp.solve(
            np.ascontiguousarray(H, dtype=np.float64),
            np.ascontiguousarray(g, dtype=np.float64),
            np.ascontiguousarray(A, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            lower, sense, **solve_options)
        elapsed = time.perf_counter() - started
        if exitflag < 1 or z is None:
            print(f"DAQP did not find an optimum (exit flag {exitflag}); applying zero input")
            return ControllerResult(
                task_error=float(np.linalg.norm(twist[:3])),
                control_input=np.zeros(nu), lyapunov_value=V,
                previous_solution=previous_solution, t_ctrl=elapsed)

        qdd = z[i_qdd]
        u = np.clip(z[i_u], lb, ub)
        robot.apply_control_input(u)
        return ControllerResult(
            task_error=float(np.linalg.norm(twist[:3])),
            control_input=u.copy(), lyapunov_value=V,
            previous_solution={"u": u.copy(), "qdd": qdd.copy(),
                               "dl": np.array([z[i_dl]]),
                               "task_V": task_V,
                               "joint_kinetic_energy": joint_kinetic_energy},
            t_ctrl=elapsed)


def experiment_target(t, args):
    """Reference generator: this is the main place to try a new experiment."""
    if args.experiment == "sequence":
        points = SEQUENCE_TARGETS[args.robot]
        point_idx = min(int(max(t, 0.0) / SEQUENCE_HOLD_SECONDS), len(points) - 1)
        return points[point_idx].copy()
    if args.experiment == "tracking":
        if args.robot == "spirob_horz":
            # Match control_node._reference() in spirob_zmq: start at
            # [0, 0.2, 0.15] and trace its Y-Z circle.
            phase = args.omega * t
            radius = 0.025
            return {
                "pos": np.array([0.0, 0.225 - radius * np.cos(phase),
                                 0.15 - radius * np.sin(phase)]),
                "vel": np.array([0.0, radius * args.omega * np.sin(phase),
                                 -radius * args.omega * np.cos(phase)]),
                "acc": np.array([0.0, radius * args.omega**2 * np.cos(phase),
                                 radius * args.omega**2 * np.sin(phase)]),
            }
        from utils import circular_trajectory
        return circular_trajectory(t, args.robot, args.omega)
    if args.target is not None:
        return np.asarray(args.target, dtype=float)
    if args.robot == "spirob_horz":
        return SPIROB_ZMQ_TARGET.copy()
    from utils import set_target
    return set_target(args.target_pos, args.robot)


def initialize_robot(robot, match_spirob_zmq):
    """Initialize the Spirob from the same keyframe/base/settle settings as ZMQ."""
    if not match_spirob_zmq:
        robot.initialize_simulation_state()
        return

    # spirob_zmq.common.rest_state() applies these profile values before
    # settling the model from the XML's `init` keyframe.
    base_body = "segment_1__configuration_default"
    robot.model.body_pos[robot.model.body(base_body).id][2] = SPIROB_ZMQ_BASE_HEIGHT
    key_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_KEY, "init")
    if key_id < 0:
        raise ValueError("The Spirob model is missing its 'init' keyframe")
    mujoco.mj_resetData(robot.model, robot.data)
    robot.data.qpos[:] = robot.model.key_qpos[key_id]
    robot.data.qvel[:] = 0.0
    for _ in range(round(SPIROB_ZMQ_SETTLE_TIME / robot.model.opt.timestep)):
        mujoco.mj_step(robot.model, robot.data)
    robot.data.time = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def camera_view_path(robot_name):
    return CAMERA_VIEW_DIR / f"{robot_name}.json"


def load_camera_view(viewer, robot_name):
    """Apply a previously saved free-camera view, returning its path on success."""
    path = camera_view_path(robot_name)
    if not path.exists():
        return None
    with path.open() as source:
        view = json.load(source)
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = float(view["azimuth"])
        viewer.cam.elevation = float(view["elevation"])
        viewer.cam.distance = float(view["distance"])
        viewer.cam.lookat[:] = np.asarray(view["lookat"], dtype=float)
    return path


def save_camera_view(viewer, robot_name):
    """Save the current free-camera pose so later runs start at this view."""
    path = camera_view_path(robot_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with viewer.lock():
        view = {
            "azimuth": float(viewer.cam.azimuth),
            "elevation": float(viewer.cam.elevation),
            "distance": float(viewer.cam.distance),
            "lookat": [float(value) for value in viewer.cam.lookat],
        }
    with path.open("w") as output:
        json.dump(view, output, indent=2)
        output.write("\n")
    return path


def initialize_free_camera(viewer, robot_name):
    """Set a reasonable free-camera starting view when tuning a robot."""
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if robot_name == "spirob_horz":
            viewer.cam.azimuth = 0.0
            viewer.cam.elevation = -20.0
            azimuth = np.deg2rad(viewer.cam.azimuth)
            viewer.cam.lookat[:] += 0.08 * np.array(
                [-np.sin(azimuth), np.cos(azimuth), 0.0])
        elif robot_name == "tendon":
            viewer.cam.lookat[:] = [0.10, 0.0, 0.05]
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = 0.0
        else:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.36]
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -20.0


def velocity_dof_groups(model):
    """Return qvel indices grouped by angular versus linear joint coordinates."""
    angular, linear = [], []
    for joint_id, joint_type in enumerate(model.jnt_type):
        dof = int(model.jnt_dofadr[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            angular.append(dof)
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            linear.append(dof)
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            angular.extend(range(dof, dof + 3))
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            linear.extend(range(dof, dof + 3))
            angular.extend(range(dof + 3, dof + 6))
    return np.asarray(angular, dtype=int), np.asarray(linear, dtype=int)


def run(args):
    robot = Robot(args.robot, "id_clf_qp", timestep=args.timestep,
                  model_path=SCENE_PATHS.get(args.robot))
    # Keep the LQR-derived task matrix and the CLF decay bound on the same
    # epsilon value selected for this experiment.
    robot.e = args.epsilon
    robot._setup_clf_matrices()
    if args.robot == "spirob_horz":
        configure_spirob_current_source(robot)
    initialize_robot(robot, args.robot == "spirob_horz")
    target_body_id = mujoco.mj_name2id(
        robot.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = (int(robot.model.body_mocapid[target_body_id])
                       if target_body_id >= 0 else -1)
    if args.experiment in ("set", "sequence") and target_mocap_id >= 0:
        robot.data.mocap_pos[target_mocap_id] = experiment_target(0.0, args)
        mujoco.mj_forward(robot.model, robot.data)
    # The kinetic-energy CLF assumes a stationary target. Keep tracking runs
    # at beta=0 by default; set-point and sequence runs use robot.py's value.
    beta = (0.0 if args.experiment == "tracking" else robot.beta)
    if args.beta is not None:
        beta = args.beta
    controller = DAQP_IDCLFQPController(epsilon=args.epsilon, beta=beta)
    angular_dof, linear_dof = velocity_dof_groups(robot.model)
    previous_solution = None
    dt = robot.model.opt.timestep
    control_period = 1.0 / args.control_rate
    next_control_time = 0.0
    if control_period < dt:
        print(f"Requested {args.control_rate:g} Hz control rate exceeds the "
              f"physics rate ({1.0 / dt:.1f} Hz); controlling once per physics step.")

    viewer = None
    camera_save_requested = {"value": False}
    if not args.headless:
        def on_viewer_key(key):
            if key in (ord("s"), ord("S")):
                camera_save_requested["value"] = True

        viewer = mujoco.viewer.launch_passive(
            robot.model, robot.data, key_callback=on_viewer_key)
        tracked_site_name = (SPIROB_TRACKED_SITE
                             if args.robot == "spirob_horz" else "ee")
        tracked_id = robot.model.site(tracked_site_name).id
        with viewer.lock():
            viewer.opt.sitegroup[0] = 1
            robot.model.site(tracked_id).group = 0
            robot.model.site(tracked_id).rgba[:] = TRACKED_MARKER_RGBA
            robot.model.site_size[tracked_id, 0] = TRACKED_MARKER_RADIUS_M
        if args.experiment == "set" and target_mocap_id >= 0:
            print("Drag the red target: double-click it to select, then hold Ctrl and "
                  "right-drag to translate. Hold Shift as well to move horizontally.")
        saved_camera_path = load_camera_view(viewer, args.robot)
        if args.tune_camera:
            if saved_camera_path is None:
                initialize_free_camera(viewer, args.robot)
            print("Camera tuning enabled: orbit, pan, and zoom with the viewer controls; "
                  "close the viewer to save this pose.")
        elif saved_camera_path is not None:
            print(f"Loaded saved camera view from {saved_camera_path}")
        else:
            if args.robot == "spirob_horz":
                initialize_free_camera(viewer, args.robot)
            elif args.robot == "helix":
                camera_id = mujoco.mj_name2id(
                    robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "ortho_side")
                if camera_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = camera_id
                else:
                    initialize_free_camera(viewer, args.robot)
            else:
                # The Tendon scene's fixed-camera frustum can miss the full
                # arm and target, so start with a movable side-on camera.
                initialize_free_camera(viewer, args.robot)
        print("Press S in the viewer at any time to save the current camera view.")

    errors = []
    control_times = []
    trace = []
    next_report = 0.0
    last_sequence_idx = 0
    sim_step = 0
    started = time.perf_counter()
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if viewer is not None and camera_save_requested["value"]:
                camera_save_requested["value"] = False
                camera_path = save_camera_view(viewer, args.robot)
                print(f"Saved camera view to {camera_path}")

            t = robot.data.time
            if (args.experiment == "sequence"
                    and t >= len(SEQUENCE_TARGETS[args.robot]) * SEQUENCE_HOLD_SECONDS):
                break
            if t + 1e-12 >= next_control_time:
                # Schedule against simulation time, allowing non-integer ratios
                # between controller period and physics timestep. If physics
                # runs slower than requested, missed updates are skipped.
                while next_control_time <= t + 1e-12:
                    next_control_time += control_period
                if (viewer is not None and args.experiment == "set"
                        and target_mocap_id >= 0):
                    # Read the viewer's live mocap target on each control tick.
                    with viewer.lock():
                        target = robot.data.mocap_pos[target_mocap_id].copy()
                else:
                    target = experiment_target(t, args)
                robot.update_kinematics()
                target_vel, target_acc, twist = robot.compute_target_data(
                    args.experiment, target)
                tracked_name = (SPIROB_TRACKED_SITE
                                if args.robot == "spirob_horz" else "ee")
                tracked_pos = robot.data.site(tracked_name).xpos.copy()
                # compute_target_data uses ee internally; replace its position
                # error with the site whose Jacobian the controller uses.
                target_pos = target["pos"] if args.experiment == "tracking" else target
                twist[:3] = np.asarray(target_pos) - tracked_pos

                sequence_idx = (min(int(t / SEQUENCE_HOLD_SECONDS),
                                    len(SEQUENCE_TARGETS[args.robot]) - 1) + 1
                                if args.experiment == "sequence" else "")
                if (args.experiment == "sequence"
                        and sequence_idx != last_sequence_idx):
                    print(f"Set point {sequence_idx}/4 at t={t:.2f}s: "
                          f"{np.array2string(target_pos, precision=3)} "
                          f"(hold {SEQUENCE_HOLD_SECONDS:g}s)")
                    last_sequence_idx = sequence_idx

                result = controller(robot, target_vel, target_acc, twist,
                                    previous_solution)
                previous_solution = result.previous_solution

                errors.append(float(result.task_error))
                control_times.append(float(result.t_ctrl or 0.0))
                qp_solution = result.previous_solution or {}
                qvel = robot.get_joint_velocities()
                angular_velocity = qvel[angular_dof]
                linear_velocity = qvel[linear_dof]
                qvel_angular_mean = (float(np.mean(angular_velocity))
                                     if angular_velocity.size else float("nan"))
                qvel_angular_variance = (float(np.var(angular_velocity))
                                         if angular_velocity.size else float("nan"))
                qvel_angular_std = (float(np.std(angular_velocity))
                                    if angular_velocity.size else float("nan"))
                qvel_angular_rms = (float(np.sqrt(np.mean(angular_velocity**2)))
                                    if angular_velocity.size else float("nan"))
                qvel_linear_mean = (float(np.mean(linear_velocity))
                                    if linear_velocity.size else float("nan"))
                qvel_linear_variance = (float(np.var(linear_velocity))
                                        if linear_velocity.size else float("nan"))
                qvel_linear_std = (float(np.std(linear_velocity))
                                   if linear_velocity.size else float("nan"))
                qvel_linear_rms = (float(np.sqrt(np.mean(linear_velocity**2)))
                                   if linear_velocity.size else float("nan"))
                trace.append([t, *tracked_pos, *np.asarray(target_pos),
                              float(result.task_error), float(result.lyapunov_value or 0.0),
                              float(qp_solution.get("task_V", 0.0)),
                              float(qp_solution.get("joint_kinetic_energy", 0.0)),
                              sequence_idx, qvel_angular_mean,
                              qvel_angular_variance, qvel_angular_std,
                              qvel_angular_rms, qvel_linear_mean,
                              qvel_linear_variance, qvel_linear_std,
                              qvel_linear_rms, *qvel.tolist()])
                if args.robot == "spirob_horz" and t >= next_report:
                    print(f"t={t:6.2f}s  {tracked_name}={np.array2string(tracked_pos, precision=3)}  "
                          f"target={np.array2string(target_pos, precision=3)}  "
                          f"error={result.task_error:.3f} m")
                    next_report = t + 0.5

            robot.step_simulation()
            sim_step += 1

            if viewer is not None:
                viewer.sync()
            elif args.realtime:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if viewer is not None:
            if args.tune_camera:
                camera_path = save_camera_view(viewer, args.robot)
                print(f"Saved camera view to {camera_path}")
            viewer.close()

    elapsed = time.perf_counter() - started
    final_error = errors[-1] if errors else float("nan")
    avg_control = float(np.mean(control_times)) if control_times else float("nan")
    print(f"Simulated {robot.data.time:.3f} s in {elapsed:.2f} s wall time")
    print(f"Final task error: {final_error:.6g} m")
    print(f"Average controller time: {avg_control * 1e3:.3f} ms")
    if trace:
        if args.trace_csv:
            trace_path = Path(args.trace_csv)
        elif args.experiment == "sequence":
            trace_path = Path("results") / f"{args.robot}_clf_sequence.csv"
        else:
            trace_path = Path("spirob_clf_tracking.csv")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", newline="") as output:
            writer = csv.writer(output)
            tracked_name = (SPIROB_TRACKED_SITE
                            if args.robot == "spirob_horz" else "ee")
            writer.writerow(["time_s", f"{tracked_name}_x_m",
                             f"{tracked_name}_y_m", f"{tracked_name}_z_m",
                             "target_x_m", "target_y_m", "target_z_m",
                             "position_error_m", "lyapunov_V", "task_V",
                             "joint_kinetic_energy_J", "setpoint_index",
                             "qvel_angular_mean_rad_s",
                             "qvel_angular_variance_rad2_s2",
                             "qvel_angular_std_rad_s", "qvel_angular_rms_rad_s",
                             "qvel_linear_mean_m_s", "qvel_linear_variance_m2_s2",
                             "qvel_linear_std_m_s", "qvel_linear_rms_m_s",
                             *[f"qvel_{i}_{'rad_s' if i in angular_dof else 'm_s' if i in linear_dof else 'unclassified'}"
                               for i in range(robot.model.nv)]])
            writer.writerows(trace)
        print(f"Tracked-site trace saved to {trace_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("spirob_horz", "helix", "tendon"),
                        default="spirob_horz")
    parser.add_argument("--experiment", choices=("set", "tracking", "sequence"),
                        default="set")
    parser.add_argument("--target-pos", choices=("pos1", "pos2", "pos3", "pos4"),
                        default="pos1", help="Named set-point target for other robots")
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Custom set-point in meters; overrides the Spirob default too")
    parser.add_argument("--omega", type=float, default=0.5,
                        help="Tracking angular rate in rad/s")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="CLF convergence parameter epsilon (gamma = 1/epsilon)")
    parser.add_argument("--beta", type=float, default=BETA,
                        help="Weight beta on joint kinetic energy in the combined CLF")
    parser.add_argument("--control-rate", type=float, default=200.0,
                        help="Controller update rate in Hz (default matches spirob_zmq)")
    parser.add_argument("--timestep", type=float, default=None,
                        help="Override the XML integration timestep")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the MuJoCo viewer")
    parser.add_argument("--tune-camera", action="store_true",
                        help="Use a free camera and save its pose when the viewer closes")
    parser.add_argument("--realtime", action="store_true",
                        help="Pace headless simulation to approximately real time")
    parser.add_argument("--trace-csv", default=None,
                        help="CSV path for the end-effector and target trace")
    args = parser.parse_args()
    if args.epsilon is None:
        args.epsilon = ROBOT_EPSILONS[args.robot]
    if args.omega <= 0:
        parser.error("--omega must be positive")
    if not np.isfinite(args.control_rate) or args.control_rate <= 0:
        parser.error("--control-rate must be finite and positive")
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be finite and positive")
    if args.beta is not None and (not np.isfinite(args.beta) or args.beta < 0):
        parser.error("--beta must be finite and nonnegative")
    if args.experiment == "tracking" and args.beta is not None and args.beta > 0:
        parser.error("the joint-velocity CLF assumes a stationary target; use --experiment set or --beta 0")
    if args.timestep is not None and args.timestep <= 0:
        parser.error("--timestep must be positive")
    if args.tune_camera and args.headless:
        parser.error("--tune-camera requires the viewer; remove --headless")
    return args


if __name__ == "__main__":
    run(parse_args())
