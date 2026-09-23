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

# Marker sites the EKF measures, added by load_model if the model lacks them
# (the spirob_mujoco model does): the origin of the body that follows segment N.
MARKER_BODIES = {
    'ee_seg5': 'segment_6__configuration_default',
    'ee_seg10': 'segment_11_2__configuration_default',
    'ee_seg15': 'segment_16__configuration_default',
    'ee_seg20': 'segment_21__configuration_default',
}


def load_model(node):
    """Load the MuJoCo model for ``node.robot`` and apply the shared settings.

    Every node (plant, estimator, controller, viewer) goes through here so they
    all see the same model: missing marker sites added, stiffness/damping
    overridden if the robot profile says so, and the base raised. Any of these
    can be overridden per node (e.g. ``-p virtual_measurement_node.joint_stiffness=0.35``
    to test robustness to model error).
    """
    model_path = node.declare_parameter('model_path')
    if not Path(model_path).exists():
        node.get_logger().error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = mujoco.MjSpec.from_file(model_path)
    existing = {site.name for site in spec.sites}
    for site_name, body_name in MARKER_BODIES.items():
        if site_name not in existing and spec.body(body_name) is not None:
            spec.body(body_name).add_site(name=site_name, pos=[0, 0, 0], size=[0.003, 0, 0],
                                          rgba=[0, 1, 0, 1], group=3)
    model = spec.compile()

    stiffness = node.declare_parameter('joint_stiffness')
    if stiffness is not None:
        model.jnt_stiffness[:] = stiffness
    damping = node.declare_parameter('joint_damping')
    if damping is not None:
        model.dof_damping[:] = damping
    model.opt.gravity[:] = node.declare_parameter('gravity', [0.0, 0.0, -9.81])
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
    """Actuator force as MuJoCo computes it, probed from the compiled model:

        force = clip(k_v * ctrl - k_e * actuator_velocity, forcerange)

    k_e = 0 for plain ``motor`` actuators; a ``dcmotor`` has back-EMF. Probing
    (as Robot._probe_actuator_constants does on the port branch) means a
    retuned ``nominal="..."`` upstream is picked up automatically.
    """

    def __init__(self, model):
        data = mujoco.MjData(model)
        nu = model.nu
        # k_v: unit ctrl at rest
        data.ctrl[:] = -1.0
        mujoco.mj_forward(model, data)
        self.k_v = -data.actuator_force.copy()
        # k_e: joint velocity with zero ctrl
        data.ctrl[:] = 0.0
        data.qvel[:] = 0.01
        mujoco.mj_forward(model, data)
        v = data.actuator_velocity.copy()
        self.k_e = np.where(np.abs(v) > 1e-12, -data.actuator_force / np.where(v == 0, 1, v), 0.0)
        limited = model.actuator_forcelimited.astype(bool) if hasattr(model, 'actuator_forcelimited') \
            else np.ones(nu, bool)
        self.f_min = np.where(limited, model.actuator_forcerange[:, 0], -np.inf)
        self.f_max = np.where(limited, model.actuator_forcerange[:, 1], np.inf)
        # Check the law is affine by predicting a third point.
        data.ctrl[:] = -2.0
        data.qvel[:] = -0.005
        mujoco.mj_forward(model, data)
        predicted = self.force(data.ctrl, data.actuator_velocity)
        self.affine = bool(np.allclose(predicted, data.actuator_force, atol=1e-9, rtol=1e-6))

    def force(self, ctrl, actuator_velocity):
        return np.clip(self.k_v * ctrl - self.k_e * actuator_velocity, self.f_min, self.f_max)

    def ctrl_bounds(self, u_min, u_max, actuator_velocity):
        """ctrl bounds valid at this actuator velocity: the voltage range
        intersected with the force limit, which for a dcmotor is velocity
        dependent (port-progress 6b45049). Collapsed to the achievable end
        rather than returning lo > hi."""
        lo = np.full(len(self.k_v), float(u_min))
        hi = np.full(len(self.k_v), float(u_max))
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
