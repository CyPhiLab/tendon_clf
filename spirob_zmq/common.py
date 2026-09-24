"""Helpers shared by several nodes."""

import math
from pathlib import Path

import mujoco
import numpy as np

# Topic names, identical to the ROS package.
MOTOR_COMMAND = '/spirob/motor_command'
MOTOR_STATE = '/spirob/motor_state'
ROBOT_STATE = '/spirob/robot_state'
SITE_MEASUREMENT = '/spirob/site_measurement'
# Ground truth from the simulated plant, for evaluating the estimator (no ROS equivalent).
TRUE_STATE = '/spirob/true_state'

def load_model(node):
    """Load the MuJoCo model for ``node.robot`` and apply the shared settings.

    Every node (plant, estimator, controller, viewer) goes through here so they
    all see the same model: stiffness/damping overridden if the robot profile
    says so, and the base raised. Any of these
    can be overridden per node (e.g. ``-p virtual_measurement_node.joint_stiffness=0.35``
    to test robustness to model error).
    """
    model_path = node.declare_parameter('model_path')
    if not Path(model_path).exists():
        node.get_logger().error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)

    stiffness = node.declare_parameter('joint_stiffness')
    if stiffness is not None:
        model.jnt_stiffness[:] = stiffness
    damping = node.declare_parameter('joint_damping')
    if damping is not None:
        model.dof_damping[:] = damping
    model.opt.gravity[:] = node.declare_parameter('gravity', [0.0, 0.0, -9.81])
    # Actuator force (for the dcmotor, rotor torque) limit, overriding the
    # model's forcerange
    torque_limit = node.declare_parameter('motor_torque_limit')
    if torque_limit is not None:
        model.actuator_forcerange[:] = [-torque_limit, torque_limit]
    base_height = node.declare_parameter('base_height')
    if base_height is not None:
        model.body_pos[model.body('segment_1__configuration_default').id][2] = base_height
    return model


def rest_state(node, model, data):
    """Put ``data`` in the robot's initial state: qpos = 0, then (if the profile
    has a ``settle_time``) let gravity settle the arm with zero ctrl. The plant
    and the estimator both start here, so the EKF's initial estimate is right."""
    settle_time = node.declare_parameter('settle_time', 0.0)
    mujoco.mj_resetData(model, data)
    for _ in range(int(round(settle_time / model.opt.timestep))):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)


def substeps(model, period):
    """Number of model timesteps per node tick. Nodes step the model at its own
    timestep instead of overwriting it with 1/rate_hz."""
    n = max(1, int(round(period / model.opt.timestep)))
    if not math.isclose(n * model.opt.timestep, period, rel_tol=1e-6):
        raise ValueError(f'tick period {period} is not a multiple of the model timestep '
                         f'{model.opt.timestep}')
    return n


class ActuatorLaw:
    """Quasi-static actuator force as MuJoCo computes it, probed from the model:

        force = clip(k_v * ctrl - k_e * actuator_velocity, forcerange)

    k_e = 0 for a plain ``motor``. For a ``dcmotor`` with inductance the model
    carries the motor current as an activation state; its electrical time
    constant (L/R, ~0.6 ms) is far below the control period, so the law is
    taken at the steady-state current,

        act = a_u * ctrl + a_v * actuator_velocity,

    which ``steady_act`` returns. Probing rather than hardcoding means retuned
    motor parameters upstream are picked up automatically.
    """

    def __init__(self, model):
        self.model = model
        nu = model.nu
        self.has_act = model.na > 0
        if self.has_act and not (model.na == nu and np.all(model.actuator_actnum == 1)):
            raise ValueError('ActuatorLaw expects one activation per actuator')
        data = mujoco.MjData(model)
        # k_v: unit ctrl at rest
        f_u, _, act_u = self._steady(data, -1.0, 0.0)
        self.k_v = -f_u
        self.a_u = -act_u
        # k_e: joint velocity with zero ctrl
        f_v, v, act_v = self._steady(data, 0.0, 0.01)
        safe_v = np.where(v == 0, 1.0, v)
        self.k_e = np.where(np.abs(v) > 1e-12, -f_v / safe_v, 0.0)
        self.a_v = np.where(np.abs(v) > 1e-12, act_v / safe_v, 0.0)
        limited = model.actuator_forcelimited.astype(bool)
        self.f_min = np.where(limited, model.actuator_forcerange[:, 0], -np.inf)
        self.f_max = np.where(limited, model.actuator_forcerange[:, 1], np.inf)
        ctrl_limited = model.actuator_ctrllimited.astype(bool)
        self.u_min = np.where(ctrl_limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self.u_max = np.where(ctrl_limited, model.actuator_ctrlrange[:, 1], np.inf)
        # Check the law is affine by predicting a third point.
        f, v, act = self._steady(data, -2.0, -0.005)
        self.affine = bool(np.allclose(self.force(-2.0, v), f, atol=1e-9, rtol=1e-6)
                           and (not self.has_act
                                or np.allclose(self.steady_act(-2.0, v), act, atol=1e-9, rtol=1e-6)))

    def _steady(self, data, ctrl, qvel):
        """actuator_force, actuator_velocity and act at the steady-state
        activation, for this ctrl and joint velocity (at qpos = 0)."""
        m = self.model
        data.ctrl[:] = ctrl
        data.qvel[:] = qvel
        data.act[:] = 0.0
        mujoco.mj_forward(m, data)
        if self.has_act:
            # Activation dynamics are linear in act (a dcmotor's current):
            # solve act_dot(act) = 0 from two evaluations
            adot0 = data.act_dot.copy()
            data.act[:] = 1.0
            mujoco.mj_forward(m, data)
            slope = data.act_dot - adot0
            data.act[:] = np.where(slope != 0, -adot0 / np.where(slope == 0, 1, slope), 0.0)
            mujoco.mj_forward(m, data)
        return data.actuator_force.copy(), data.actuator_velocity.copy(), data.act.copy()

    def force(self, ctrl, actuator_velocity):
        return np.clip(self.k_v * ctrl - self.k_e * actuator_velocity, self.f_min, self.f_max)

    def steady_act(self, ctrl, actuator_velocity):
        """Steady-state activation (motor current) for this ctrl and velocity."""
        return self.a_u * ctrl + self.a_v * actuator_velocity

    def ctrl_bounds(self, u_min, u_max, actuator_velocity):
        """ctrl bounds valid at this actuator velocity: the ctrl range
        intersected with the force limit, which for a dcmotor is velocity
        dependent (port-progress 6b45049). ``u_min``/``u_max`` of None use the
        model's ctrlrange. Collapsed to the achievable end rather than
        returning lo > hi."""
        lo = self.u_min.copy() if u_min is None else np.full(len(self.k_v), float(u_min))
        hi = self.u_max.copy() if u_max is None else np.full(len(self.k_v), float(u_max))
        kev = self.k_e * actuator_velocity
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = np.maximum(lo, (self.f_min + kev) / self.k_v)
            hi = np.minimum(hi, (self.f_max + kev) / self.k_v)
        hi = np.maximum(hi, lo)
        return lo, hi


def actuator_moment(model, data):
    """Dense (nu, nv) actuator moment (gear included)."""
    moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(moment, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return moment


def command_to_u(msg, motor_ids):
    """Reorder a MotorCommand's ``u`` to match ``motor_ids`` (missing ids -> 0)."""
    if list(msg['motor_ids']) == list(motor_ids):
        return np.asarray(msg['u'], dtype=float).copy()
    id_to_u = dict(zip(msg['motor_ids'], msg['u']))
    return np.array([id_to_u.get(motor_id, 0.0) for motor_id in motor_ids], dtype=float)
