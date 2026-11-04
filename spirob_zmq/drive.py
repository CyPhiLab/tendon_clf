"""Drive models: how motor current becomes tendon tension.

`drive_model='current'` (default): the robot XML as it is. Each AK drive is an ideal current source
on the robot's spatial tendon (`<motor gear="31.75">`) with Coulomb tendon friction, and the 24 V bus
limit is applied in code (common.VoltageLimit).

`drive_model='motor'`: the drives and the robot's give as they are on the hardware (docs/drive_model.md):

- Each drive is a slide joint `motor<i>` in tendon units [m] (motor angle = qpos / r_spool; pulling is
  negative, as on the AK encoders). It carries the rotor inertia (armature 0.9833 kg = 1.5733e-5 kg m^2
  * (10 / 0.04 m)^2) and viscous damping.
- A MuJoCo `dcmotor` drives it in current mode: input `ff` through the on-board controller, which applies
  v = R/K tau + K l_dot (R 2.2 ohm) clamped at the 24 V bus. With gear 31.75 = Kt * 10 / r_spool and
  K = 1, ctrl is the motor current in A and the actuator velocity is the back-EMF in V: the same law as
  common.VoltageLimit, now inside the model.
- LuGre friction on each drive: static above Coulomb gives the stick-slip, the bristle stiffness the
  stuck drive's own small give. LuGre acts in actuator units: the tendon-side values below are divided by
  the gear (forces) or the gear squared (stiffness, damping), and the Stribeck velocity is multiplied by it.
- The motors sit on one carriage, a slide joint `drive_carriage` with a spring: it stands for the robot's
  axial compression (segments squeezing about 0.3-0.6 mm per joint, ~11 mm along the robot under load),
  which shortens all three tendon paths alike, against the sum of the tensions. That is what lets one
  tendon wind in and bend the robot while the other two drives stay stuck, without loading them.
- Each tendon runs from a site on its motor through the base plate and the robot: a spatial tendon that is
  pull-only (a stiff dead-band spring: zero force when slack).
- The robot's hinges get Coulomb friction with a stiff friction constraint, so the robot holds a pose
  between slips as the hardware does (MuJoCo's default friction impedance lets it creep).

Everything is plain MJCF (`python -m spirob_zmq.drive --export model.xml` writes it), so the model runs
without this module, e.g. in an RL pipeline. Only placing the motors for a new start pose (`full_qpos`)
needs code; it reads the rest tensions from the `drive_rest_tension` custom numeric.

The motor model adds 4 DOFs (carriage and motors, nq = nv = 52, after the robot's 48) and 3 activations
(the LuGre bristle states). Nodes that only need the robot (controller, pose fit, viewer) keep
`drive_model='current'`; ROBOT_STATE carries the robot's 48 coordinates either way.
"""

import argparse
import os
import re

import mujoco
import numpy as np

DRIVE_MODELS = ('current', 'motor')

GEAR = 31.75             # N/A: gearbox 10 / spool radius 0.04 m * Kt 0.127 N m/A
R_SPOOL = 0.04
ROTOR_ARMATURE = 0.9833  # kg on the tendon
CARRIAGE = 'drive_carriage'
MOTOR_OFFSET = 1.0       # m from the plate hole to the motor site: room for 25 rad of pay-out

# Fitted to the 20261103 open-loop runs (docs/drive_model.md). Tendon units: N, m, s.
# Per-drive values are listed in actuator order (motor_ids 10, 1, 2).
MOTOR_DRIVE = {
    'drive_static_friction': [9.4, 9.2, 9.6],   # N, LuGre static (breakaway)
    'drive_coulomb_friction': 2.0,              # N, LuGre Coulomb (sliding)
    'drive_stribeck_velocity': 0.05,            # m/s
    'drive_bristle_stiffness': 4e4,             # N/m, LuGre sigma0 (stuck drive: +-0.005 rad at +-0.24 A)
    'drive_bristle_damping': None,              # N s/m, LuGre sigma1; None: critical with the rotor inertia
    'drive_damping': 20.0,                      # N s/m, viscous
    'tendon_stiffness': 500.0,                 # N/m, pull-only tendon (stretch, knots, plate)
    'compression_stiffness': 485.0,             # N/m, robot axial give against the sum of the tensions
    'compression_damping': 20.0,                # N s/m
    'compression_inertia': 0.05,                # kg
    'drive_rest_tension': [1.9, 4.5, 3.8],      # N per tendon at the init keyframe and a fitted start pose
    'drive_joint_frictionloss': 0.165,            # N m on the largest hinge
    'drive_joint_friction_power': 0.55,          # hinge i: frictionloss * (k_i / k_1)^power
    'drive_joint_friction_solref': [0.002, 1.0],
    'drive_joint_friction_solimp': [0.9999, 0.99999],
    'drive_joint_stiffness_scale': 0.61,
}


def _param(node, key, n=None):
    value = node.declare_parameter(key)
    value = MOTOR_DRIVE[key] if value is None else value
    if n is not None:
        value = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    return value


def _tendon_sites(model, t):
    return [model.site(model.wrap_objid[w]).name for w in range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t])]


# Not the robot hinges' class defaults (range, friction): the drive joints are free slides
_FREE = dict(limited=mujoco.mjtLimited.mjLIMITED_FALSE, range=[0.0, 0.0], frictionloss=0.0)


def _point_mass(body):
    # Tiny non-colliding geom: the inertia of these bodies is the joints' armature
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0], mass=1e-4, contype=0, conaffinity=0,
                  group=5, rgba=[0.5, 0.5, 0.5, 0.3])


def build_motor_model(model_path, node):
    """The robot from model_path with each actuator's tendon driven by a motor DOF (see module doc)."""
    return build_motor_spec(model_path, node)[0]


def build_motor_spec(model_path, node):
    """(model, spec) of the motor drive model; the spec writes the same model as MJCF."""
    base = mujoco.MjModel.from_xml_path(model_path)
    if base.nu == 0 or any(base.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON for i in range(base.nu)):
        raise ValueError('drive_model=motor expects every actuator on a tendon')
    if base.nq != base.nv:
        raise ValueError('drive_model=motor expects hinge/slide joints only')
    nu = base.nu
    spec = mujoco.MjSpec.from_file(model_path)
    acts = list(spec.actuators)
    tendons = [base.tendon(base.actuator_trnid[i, 0]).name for i in range(nu)]
    paths = [_tendon_sites(base, base.tendon(name).id) for name in tendons]

    static = _param(node, 'drive_static_friction', nu)
    coulomb = _param(node, 'drive_coulomb_friction', nu)
    sigma0 = float(_param(node, 'drive_bristle_stiffness'))
    sigma1 = _param(node, 'drive_bristle_damping')
    sigma1 = 2.0 * np.sqrt(sigma0 * ROTOR_ARMATURE) if sigma1 is None else float(sigma1)
    v_stribeck = float(_param(node, 'drive_stribeck_velocity'))
    k_tendon = float(_param(node, 'tendon_stiffness'))
    rest = _param(node, 'drive_rest_tension', nu)
    k_comp = float(_param(node, 'compression_stiffness'))

    # The tendons leave the base body (the plate) parallel; the carriage slides along them
    first = [base.site(p[0]) for p in paths]
    base_body = base.body(first[0].bodyid[0]).name
    if any(base.body(f.bodyid[0]).name != base_body for f in first):
        raise ValueError('drive_model=motor expects every tendon to start on the same body')
    direction = base.site_pos[first[0].id] - base.site_pos[base.site(paths[0][1]).id]
    direction /= np.linalg.norm(direction)
    carriage = spec.body(base_body).add_body(name=CARRIAGE)
    _point_mass(carriage)
    # qpos > 0 moves the motors towards the plate: all tendons longer (the robot compressed)
    carriage.add_joint(name=CARRIAGE, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=-direction,
                       armature=float(_param(node, 'compression_inertia')),
                       damping=float(_param(node, 'compression_damping')),
                       stiffness=[k_comp, 0.0, 0.0], springref=-float(np.sum(rest)) / k_comp, **_FREE)

    old = {t.name: t for t in spec.tendons}
    for i, (act, name, path) in enumerate(zip(acts, tendons, paths)):
        # The motor sits MOTOR_OFFSET behind the tendon's first site (the base-plate hole), on the line the
        # tendon leaves the robot along, and slides along it: qpos > 0 pays tendon out
        motor = carriage.add_body(name=f'motor{i}', pos=base.site_pos[base.site(path[0]).id] + MOTOR_OFFSET * direction)
        _point_mass(motor)
        motor.add_joint(name=f'motor{i}', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=-direction,
                        armature=ROTOR_ARMATURE, damping=float(_param(node, 'drive_damping')), **_FREE)
        motor.add_site(name=f'motor{i}_site')

        t_old = old[name]
        rgba, width = list(t_old.rgba), t_old.width
        spec.delete(t_old)
        t = spec.add_tendon(name=name, rgba=rgba, width=width)
        t.wrap_site(f'motor{i}_site')
        for s in path:
            t.wrap_site(s)
        t.stiffness = [k_tendon, 0.0, 0.0]

        act.set_to_dcmotor(motorconst=[1.0, 1.0], resistance=float(node.declare_parameter('motor_resistance', 2.2)),
                           controller=[0, 0, 0, 0, 0, float(node.declare_parameter('v_bus', 24.0))],
                           lugre=[sigma0 / GEAR ** 2, sigma1 / GEAR ** 2, coulomb[i] / GEAR, static[i] / GEAR,
                                  v_stribeck * GEAR],
                           ctrlspec=int(mujoco.mjtCtrlInput.mjINPUT_FF))
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = f'motor{i}'
        act.gear[0] = GEAR
        # A motor can pay tendon out; the tendon then goes slack
        act.ctrlrange = [float(base.actuator_ctrlrange[i, 0]), max(float(base.actuator_ctrlrange[i, 1]), 0.55)]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # Robot hinges: friction that holds a pose, stiffness scale
    k = base.jnt_stiffness.copy()
    friction = _param(node, 'drive_joint_frictionloss') * (k / k[0]) ** _param(node, 'drive_joint_friction_power')
    scale = float(_param(node, 'drive_joint_stiffness_scale'))
    solref = list(_param(node, 'drive_joint_friction_solref'))
    solimp = list(_param(node, 'drive_joint_friction_solimp'))
    for j in range(base.njnt):
        js = spec.joint(base.joint(j).name)
        js.frictionloss = float(friction[j])
        js.stiffness = [float(k[j] * scale), 0.0, 0.0]
        js.solref_friction = solref
        imp = list(js.solimp_friction)
        js.solimp_friction = solimp + imp[2:]

    spec.add_numeric(name='drive_rest_tension', data=[float(x) for x in rest])
    for key in spec.keys:
        key.qpos = list(key.qpos) + [0.0] * (1 + nu)

    model = spec.compile()
    # Tendon spring rest lengths (upper edge of the dead band): the rest tensions at the init keyframe
    # with the carriage and the motors at 0. Written back into the spec too, for export.
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[model.key(node.declare_parameter('init_key', 'init')).id]
    mujoco.mj_fwdPosition(model, data)
    for i, name in enumerate(tendons):
        tid = model.tendon(name).id
        length = float(data.ten_length[tid] - rest[i] / k_tendon)
        model.tendon_lengthspring[tid] = [0.0, length]
        spec.tendon(name).springlength = [0.0, length]
    return model, spec


def is_motor_model(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CARRIAGE) >= 0


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


def rest_tension(model):
    n = model.numeric('drive_rest_tension')
    return model.numeric_data[n.adr[0]:n.adr[0] + n.size[0]].copy()


def full_qpos(model, q_robot, data=None):
    """qpos from robot coordinates: carriage at 0 and the motors placed for the rest tensions."""
    q_robot = np.asarray(q_robot, dtype=float)
    if not is_motor_model(model):
        return q_robot.copy()
    if q_robot.shape == (model.nq,):
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


def main(argv=None):
    from spirob_zmq.common import ProfileParams
    parser = argparse.ArgumentParser(description='Write the robot with drive_model=motor as one MJCF file.')
    parser.add_argument('--export', required=True, help='output XML path')
    parser.add_argument('--robot', default=None)
    args = parser.parse_args(argv)
    params = ProfileParams(args.robot)
    model_path = params.declare_parameter('model_path')
    model, spec = build_motor_spec(model_path, params)
    xml = spec.to_xml()
    # Mesh paths stay valid from wherever the file is written
    meshdir = os.path.join(os.path.dirname(os.path.abspath(model_path)), spec.meshdir)
    rel = os.path.relpath(meshdir, os.path.dirname(os.path.abspath(args.export)))
    xml = xml.replace(f'meshdir="{spec.meshdir}"', f'meshdir="{rel}"', 1)
    # to_xml keeps 6 digits; the tendon rest lengths need more (1 um is 2 mN of tension)
    for t in tendon_ids(model):
        name = model.tendon(t).name
        xml = re.sub(rf'(<spatial name="{name}"[^>]*springlength=")[^"]*"',
                     lambda m_, t=t: f'{m_.group(1)}0 {float(model.tendon_lengthspring[t, 1])!r}"', xml)
    with open(args.export, 'w') as f:
        f.write(xml)
    print(f'wrote {args.export}: nq {model.nq}, nu {model.nu}, na {model.na}')


if __name__ == '__main__':
    main()
