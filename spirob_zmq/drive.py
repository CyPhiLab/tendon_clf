"""Drive models: how motor current becomes tendon tension.

The robot XML (external/spirob_mujoco/models/spirob.xml) models the drives as on the hardware
(docs/drive_model.md):

- Each AK drive is a slide joint `motor<i>` in tendon units [m] (motor angle = qpos / r_spool; pulling is
  negative, as on the AK encoders), carrying the rotor inertia (0.9833 kg on the tendon) and damping.
- A MuJoCo `dcmotor` drives it in current mode: input `ff` through the on-board controller, which applies
  v = R/K tau + K l_dot (R 2.2 ohm) clamped at the 24 V bus. With gear 31.75 = Kt * 10 / r_spool and
  K = 1, ctrl is the motor current in A and the actuator velocity is the back-EMF in V: the same law as
  common.VoltageLimit, inside the model.
- LuGre friction on each drive (stick-slip). LuGre acts in actuator units: tendon-side forces are divided
  by the gear, stiffness and damping by the gear squared, and the Stribeck velocity is multiplied by it.
- The motors sit on `drive_carriage`, a spring for the robot's axial compression, which shortens all three
  tendon paths alike against the sum of the tensions.
- The tendons run from the motors through the base plate and are pull-only (dead-band springs).
- The hinges have Coulomb friction with a stiff friction constraint, and the fitted stiffness.

`drive_model='motor'` uses that model as it is (nq = nv = 52: the robot's 48 hinges, then the carriage and
the three motors; 3 activations, the LuGre bristle states). Profile parameters can override the drive
friction (`drive_static_friction`, `drive_coulomb_friction`, `drive_stribeck_velocity`,
`drive_bristle_stiffness`, `drive_damping`), in tendon units.

`drive_model='current'` (default) derives the previous model from it: no drive DOFs, each actuator an
ideal current source on the robot's tendon (`<motor gear="31.75">`, the 24 V bus limit applied in code by
common.VoltageLimit), Coulomb tendon friction 2.5 N, rotor inertia and damping on the tendon, MuJoCo's
default hinge friction and the CAD hinge stiffness. The controller, the pose fit and the viewer use it;
the simulated plants, the EKF and replay_hw use the profile's choice.
"""

import mujoco
import numpy as np

DRIVE_MODELS = ('current', 'motor')

GEAR = 31.75             # N/A: gearbox 10 / spool radius 0.04 m * Kt 0.127 N m/A
R_SPOOL = 0.04
CARRIAGE = 'drive_carriage'

# The current-source model as it was before the robot XML modelled the drives
CURRENT_SOURCE = {
    'tendon_armature': 0.9833,        # rotor inertia on the tendon [kg]
    'tendon_damping': 20.0,           # N s/m
    'tendon_frictionloss': 2.5,       # N
    'tendon_range': [-0.1, 0.63],
    'ctrlrange': [-1.9, -0.1],
    'joint_frictionloss': 0.001,      # N m, the XML's class default
    'joint_solref_friction': [0.02, 1.0],
    'joint_solimp_friction': [0.9, 0.95, 0.001, 0.5, 2.0],
}


def is_motor_model(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CARRIAGE) >= 0


def _numeric(model, name):
    n = model.numeric(name)
    return model.numeric_data[n.adr[0]:n.adr[0] + n.size[0]].copy()


def rest_tension(model):
    """Tendon tensions [N] the tendon springs carry with the robot straight and the drives at 0."""
    return _numeric(model, 'drive_rest_tension')


def motor_qpos_ids(model):
    return np.array([model.jnt_qposadr[model.joint(f'motor{i}').id] for i in range(model.nu)])


def robot_nq(model):
    """Number of robot coordinates; the carriage and the motors come after them."""
    return model.jnt_qposadr[model.joint(CARRIAGE).id] if is_motor_model(model) else model.nq


def tendon_ids(model):
    """The tendon each actuator pulls (current model: its transmission; motor model: from its motor)."""
    if not is_motor_model(model):
        return np.array([model.actuator_trnid[i, 0] for i in range(model.nu)])
    out = []
    for i in range(model.nu):
        site = model.site(f'motor{i}_site').id
        out.append(next(t for t in range(model.ntendon)
                        if model.wrap_type[model.tendon_adr[t]] == mujoco.mjtWrap.mjWRAP_SITE
                        and model.wrap_objid[model.tendon_adr[t]] == site))
    return np.array(out)


def load(model_path, node, drive_model):
    """The robot model for drive_model ('motor': the XML as it is; 'current': derived, see module doc)."""
    if drive_model not in DRIVE_MODELS:
        raise ValueError(f'drive_model must be one of {DRIVE_MODELS}, got {drive_model!r}')
    model = mujoco.MjModel.from_xml_path(model_path)
    if not is_motor_model(model):
        if drive_model == 'motor':
            raise ValueError(f'{model_path} has no drive DOFs ({CARRIAGE}); update external/spirob_mujoco')
        return model
    if drive_model == 'current':
        return current_source_model(model_path)
    _apply_drive_overrides(model, node)
    return model


def _apply_drive_overrides(model, node):
    nu = model.nu
    static = node.declare_parameter('drive_static_friction')
    if static is not None:
        model.actuator_biasprm[:, 4] = np.broadcast_to(np.asarray(static, float), (nu,)) / GEAR
    coulomb = node.declare_parameter('drive_coulomb_friction')
    if coulomb is not None:
        model.actuator_biasprm[:, 3] = np.broadcast_to(np.asarray(coulomb, float), (nu,)) / GEAR
    v_s = node.declare_parameter('drive_stribeck_velocity')
    if v_s is not None:
        model.actuator_biasprm[:, 5] = float(v_s) * GEAR
    sigma0 = node.declare_parameter('drive_bristle_stiffness')
    if sigma0 is not None:
        # keep the bristle damping critical with the rotor inertia
        armature = model.dof_armature[model.jnt_dofadr[model.actuator_trnid[:, 0]]]
        model.actuator_dynprm[:, 5] = float(sigma0) / GEAR ** 2
        model.actuator_dynprm[:, 6] = 2.0 * np.sqrt(float(sigma0) * armature) / GEAR ** 2
    damping = node.declare_parameter('drive_damping')
    if damping is not None:
        model.dof_damping[motor_qpos_ids(model)] = damping  # slide joints: qpos and dof addresses agree


def current_source_model(model_path):
    """The previous current-source robot model, derived from the motor-drive XML."""
    return current_source_spec(model_path).compile()


def current_source_spec(model_path):
    """MjSpec of current_source_model (for building variants on it)."""
    model = mujoco.MjModel.from_xml_path(model_path)
    spec = mujoco.MjSpec.from_file(model_path)
    c = CURRENT_SOURCE
    n_robot = robot_nq(model)
    scale = float(_numeric(model, 'drive_joint_stiffness_scale')[0])
    tendons = [model.tendon(t).name for t in tendon_ids(model)]
    paths = [[model.site(model.wrap_objid[w]).name
              for w in range(model.tendon_adr[t] + 1, model.tendon_adr[t] + model.tendon_num[t])]
             for t in tendon_ids(model)]

    # Tendons and actuators first: deleting the carriage deletes whatever still refers to its sites
    old = {t.name: t for t in spec.tendons}
    for name, path in zip(tendons, paths):
        rgba, width = list(old[name].rgba), old[name].width
        spec.delete(old[name])
        t = spec.add_tendon(name=name, rgba=rgba, width=width, armature=c['tendon_armature'],
                            frictionloss=c['tendon_frictionloss'], range=c['tendon_range'],
                            limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        t.damping = [c['tendon_damping'], 0.0, 0.0]
        for s in path:
            t.wrap_site(s)
    for act, name in zip(spec.actuators, tendons):
        act.set_to_motor()
        act.trntype = mujoco.mjtTrn.mjTRN_TENDON
        act.target = name
        act.gear[0] = GEAR
        act.ctrlrange = c['ctrlrange']
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    spec.delete(spec.body(CARRIAGE))
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        js = spec.joint(model.joint(j).name)
        js.stiffness = [float(model.jnt_stiffness[j] / scale), 0.0, 0.0]
        js.frictionloss = c['joint_frictionloss']
        js.solref_friction = c['joint_solref_friction']
        js.solimp_friction = c['joint_solimp_friction']
    for name in ('drive_rest_tension', 'drive_joint_stiffness_scale'):
        spec.delete(spec.numeric(name))
    for key in spec.keys:
        key.qpos = list(key.qpos)[:n_robot] + list(key.qpos)[model.jnt_qposadr[model.joint(CARRIAGE).id] + 1 + model.nu:]
    return spec


def full_qpos(model, q_robot, data=None):
    """qpos from robot coordinates: carriage at 0 and the motors placed for the rest tensions."""
    q_robot = np.asarray(q_robot, dtype=float)
    if not is_motor_model(model) or q_robot.shape == (model.nq,):
        return q_robot.copy()
    data = mujoco.MjData(model) if data is None else data
    qpos = np.zeros(model.nq)
    qpos[:robot_nq(model)] = q_robot
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)
    t = tendon_ids(model)
    stretch = data.ten_length[t] - model.tendon_lengthspring[t, 1]
    # Tendon length falls 1:1 as a motor pays out (qpos up)
    qpos[motor_qpos_ids(model)] = stretch - rest_tension(model) / model.tendon_stiffness[t]
    return qpos


def motor_position(model, data, r_spool=R_SPOOL):
    """Drive-shaft angles [rad] up to a constant: the motor DOFs, or the tendon lengths (current model)."""
    if is_motor_model(model):
        return data.qpos[motor_qpos_ids(model)] / r_spool
    return data.ten_length[tendon_ids(model)] / r_spool


def tendon_tension(model, data):
    """Tension in each driven tendon [N]."""
    t = tendon_ids(model)
    if is_motor_model(model):
        stretch = np.maximum(0.0, data.ten_length[t] - model.tendon_lengthspring[t, 1])
        return model.tendon_stiffness[t] * stretch
    return -data.actuator_force * model.actuator_gear[:, 0]
