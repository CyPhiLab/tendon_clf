import math
from pathlib import Path

import mujoco
import numpy as np

MOTOR_COMMAND = '/spirob/motor_command'
MOTOR_STATE = '/spirob/motor_state'
ROBOT_STATE = '/spirob/robot_state'
SITE_MEASUREMENT = '/spirob/site_measurement'
TRUE_STATE = '/spirob/true_state'

def load_model(node):
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
    torque_limit = node.declare_parameter('motor_torque_limit')
    if torque_limit is not None:
        model.actuator_forcerange[:] = [-torque_limit, torque_limit]
    base_height = node.declare_parameter('base_height')
    if base_height is not None:
        model.body_pos[model.body('segment_1__configuration_default').id][2] = base_height
    return model


def rest_state(node, model, data):
    settle_time = node.declare_parameter('settle_time', 0.0)
    mujoco.mj_resetData(model, data)
    for _ in range(int(round(settle_time / model.opt.timestep))):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)


def substeps(model, period):
    n = max(1, int(round(period / model.opt.timestep)))
    if not math.isclose(n * model.opt.timestep, period, rel_tol=1e-6):
        raise ValueError(f'tick period {period} is not a multiple of the model timestep '
                         f'{model.opt.timestep}')
    return n


class ActuatorLaw:
    """force = clip(k_v * ctrl - k_e * actuator_velocity, forcerange), at steady-state act."""

    def __init__(self, model):
        self.model = model
        nu = model.nu
        self.has_act = model.na > 0
        if self.has_act and not (model.na == nu and np.all(model.actuator_actnum == 1)):
            raise ValueError('ActuatorLaw expects one activation per actuator')
        data = mujoco.MjData(model)
        f_u, _, act_u = self._steady(data, -1.0, 0.0)
        self.k_v = -f_u
        self.a_u = -act_u
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
        f, v, act = self._steady(data, -2.0, -0.005)
        self.affine = bool(np.allclose(self.force(-2.0, v), f, atol=1e-9, rtol=1e-6)
                           and (not self.has_act
                                or np.allclose(self.steady_act(-2.0, v), act, atol=1e-9, rtol=1e-6)))

    def _steady(self, data, ctrl, qvel):
        m = self.model
        data.ctrl[:] = ctrl
        data.qvel[:] = qvel
        data.act[:] = 0.0
        mujoco.mj_forward(m, data)
        if self.has_act:
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
        return self.a_u * ctrl + self.a_v * actuator_velocity

    def ctrl_bounds(self, u_min, u_max, actuator_velocity):
        lo = self.u_min.copy() if u_min is None else np.full(len(self.k_v), float(u_min))
        hi = self.u_max.copy() if u_max is None else np.full(len(self.k_v), float(u_max))
        kev = self.k_e * actuator_velocity
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = np.maximum(lo, (self.f_min + kev) / self.k_v)
            hi = np.minimum(hi, (self.f_max + kev) / self.k_v)
        hi = np.maximum(hi, lo)
        return lo, hi


def actuator_moment(model, data):
    moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(moment, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    return moment


def command_to_u(msg, motor_ids):
    if list(msg['motor_ids']) == list(motor_ids):
        return np.asarray(msg['u'], dtype=float).copy()
    id_to_u = dict(zip(msg['motor_ids'], msg['u']))
    return np.array([id_to_u.get(motor_id, 0.0) for motor_id in motor_ids], dtype=float)
