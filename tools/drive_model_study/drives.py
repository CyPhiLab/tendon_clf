"""Drive-model variants A-E for the SpiRob model, built with MjSpec; one stepper per variant.

Units: tendon forces in N, tendon lengths in m. G = 31.75 N/A (ratio 10 / r_spool 0.04 * Kt 0.127).
Actuator units (dcmotor): length l = G * tendon length (= Kt * motor angle), force = tendon force / G (= current in A
for an ideal current source). So LuGre stiffness [N/m on the tendon] -> /G^2, forces [N] -> /G, velocity [m/s] -> *G.
"""
import os
import sys
import numpy as np
import mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from spirob_zmq.common import ProfileParams
from spirob_zmq.drive import current_source_spec

G = 31.75
R_SPOOL = 0.04
ARMATURE = 0.9833          # rotor inertia 1.5733e-5 * (10 / 0.04)^2, kg on the tendon
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mujoco_models', 'spirob', 'spirob_horz_control.xml')

DEFAULTS = dict(
    drive='A',
    k_scale=1.0,           # joint stiffness scale (robot joints)
    jf=None,               # robot joint frictionloss [N m] (None: keep XML 0.001)
    jf_solimp=None,        # (d0, dwidth) for robot joint friction
    jf_p=0.0,              # joint friction scales as (stiffness_i / stiffness_1)^jf_p (0: uniform)
    hole=0.0,              # plate-hole friction / tension (Coulomb, at the motor side)
    # tendon-level friction (A, B)
    tf=2.5, tf_solimp=None,
    # B: tendon stick-slip
    fs=9.5, fk=2.0, veps=2e-3,
    # motor DOF (C, E)
    m_damp=20.0,           # motor viscous damping on the tendon [N s/m]
    rope_k=400.0,          # robot-side give, as a pull-only spring [N/m]
    rope_k2=0.0,           # quadratic term [N/m^2]
    rope_b=0.0,            # rope damping [N s/m]
    give='rope',           # 'rope': give as a spring in each tendon; 'common': robot axial compression,
                           # one carriage DOF carrying the motors (shortens all three tendons alike)
    comp_k=800.0,          # compression stiffness against the sum of tensions [N/m]
    comp_b=20.0,           # compression damping [N s/m]
    comp_m=0.05,           # compression inertia [kg]
                           # 'backbone': a stiff slide joint along the backbone at every hinged segment,
                           # total compliance 1/comp_k, spread as segment length^-bb_p (0: uniform)
    bb_p=0.6,              # per-joint compression 0.3 mm proximal -> 0.6 mm at segments 13-16
    bb_m=0.02,             # axial armature per slide [kg] (keeps the light distal slides stable at 1 ms)
    bb_zeta=1.0,           # damping ratio of each slide
    bb_fric=0.0,           # Coulomb friction per slide [N]
    T0=3.175,              # rope tension at the fitted start pose [N]
    # LuGre (D, E) in tendon units
    sigma0=2e4, sigma1=None, Fc=2.0, Fs=9.5, vs=0.002,
    dt=0.001,
)


def _robot_tendon_sites(model):
    out = []
    for t in range(model.ntendon):
        adr, num = model.tendon_adr[t], model.tendon_num[t]
        out.append([model.site(model.wrap_objid[w]).name for w in range(adr, adr + num)])
    return out


def build(cfg, base_height=0.125):
    """Return (model, info). info: robot nq/nv, motor qpos/dof ids, tendon ids, etc."""
    c = dict(DEFAULTS, **cfg)
    for key in [k for k in c if '.' in k]:
        name, i = key.split('.')
        arr = list(np.broadcast_to(np.asarray(c[name], float), (3,)).copy())
        arr[int(i)] = c.pop(key)
        c[name] = arr
    drive = c['drive']
    # the current-source robot (the robot XML now carries drive E itself; the variants are built on this)
    spec = current_source_spec(MODEL_PATH)
    base = current_source_spec(MODEL_PATH).compile()
    sites = _robot_tendon_sites(base)
    tnames = [base.tendon(t).name for t in range(base.ntendon)]
    acts = list(spec.actuators)
    info = {'drive': drive, 'nq_robot': base.nq, 'nv_robot': base.nv}

    if drive in ('C', 'E'):
        seg1 = spec.body('segment_1__configuration_default')
        parent, off = seg1, np.zeros(3)
        if c['give'] == 'common':
            parent = seg1.add_body(name='carriage', pos=[0.3, 0, 0], mass=1e-4, inertia=[1e-8, 1e-8, 1e-8])
            parent.add_joint(name='carriage', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                             armature=c['comp_m'], damping=c['comp_b'], stiffness=[c['comp_k'], 0, 0])
            off = np.array([-0.3, 0, 0])
        for i, (tn, slist) in enumerate(zip(tnames, sites)):
            p_site = base.site_pos[base.site(slist[0]).id].copy() + off
            body = parent.add_body(name=f'motor{i}', pos=p_site + np.array([0.3, 0, 0]), mass=1e-4,
                                 inertia=[1e-8, 1e-8, 1e-8])
            body.add_joint(name=f'motor{i}', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                           armature=ARMATURE, damping=c['m_damp'], range=[-1, 1], limited=mujoco.mjtLimited.mjLIMITED_FALSE)
            body.add_site(name=f'motor{i}_site', pos=[0, 0, 0])
        if c['give'] == 'backbone':
            hinged = [b for b in range(1, base.nbody) if base.body_jntnum[b]]
            size = np.array([np.linalg.norm(base.body_pos[b]) for b in hinged])
            compliance = size ** -c['bb_p']
            compliance *= (1.0 / c['comp_k']) / compliance.sum()
            info['bb_k'] = 1.0 / compliance
            for b, kj in zip(hinged, info['bb_k']):
                R = np.zeros(9); mujoco.mju_quat2Mat(R, base.body_quat[b]); R = R.reshape(3, 3)
                axis = R.T @ base.body_pos[b]; axis /= np.linalg.norm(axis)
                mass = base.body_subtreemass[b] + c['bb_m']
                spec.body(base.body(b).name).add_joint(
                    name=f'comp_{b}', type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, armature=c['bb_m'],
                    stiffness=[float(kj), 0, 0], damping=float(2 * c['bb_zeta'] * np.sqrt(kj * mass)),
                    frictionloss=c['bb_fric'], limited=mujoco.mjtLimited.mjLIMITED_FALSE)
        old = {t.name: t for t in spec.tendons}
        for i, (tn, slist) in enumerate(zip(tnames, sites)):
            t_old = old[tn]
            spec.delete(t_old)
            t = spec.add_tendon(name=tn)
            t.wrap_site(f'motor{i}_site')
            for s in slist:
                t.wrap_site(s)
            t.width = 0.0005
            t.rgba = [0.95, 0.3, 1, 1]
            t.stiffness = [c['rope_k'], c['rope_k2'], 0.0]
            t.damping = [c['rope_b'], 0.0, 0.0]
            t.armature = 0.0
            t.frictionloss = 0.0
        for i, a in enumerate(acts):
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.target = f'motor{i}'
            a.ctrlrange = [-1.9, 0.55]
    if drive in ('D', 'E'):
        sigma1 = c['sigma1'] if c['sigma1'] is not None else 2 * np.sqrt(c['sigma0'] * ARMATURE)
        Fs3 = np.broadcast_to(np.asarray(c['Fs'], float), (3,))
        Fc3 = np.broadcast_to(np.asarray(c['Fc'], float), (3,))
        for i, a in enumerate(acts):
            # current mode: ff torque input through the on-board controller, V = R/K tau + K l_dot, |V| <= 24 V
            a.set_to_dcmotor(motorconst=[1.0, 1.0], resistance=2.2,
                             controller=[0, 0, 0, 0, 0, 24.0],
                             lugre=[c['sigma0'] / G**2, sigma1 / G**2, Fc3[i] / G, Fs3[i] / G, c['vs'] * G],
                             ctrlspec=int(mujoco.mjtCtrlInput.mjINPUT_FF))
            a.gear[0] = G
            if drive == 'D':
                a.ctrlrange = [-1.9, -0.1]
            else:
                a.ctrlrange = [-1.9, 0.55]
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    model = spec.compile()
    model.body_pos[model.body('segment_1__configuration_default').id][2] = base_height
    model.opt.timestep = c['dt']
    # robot hinges by name (backbone slides interleave with them)
    hinge = np.array([model.joint(base.joint(j).name).id for j in range(base.njnt)])
    info['r_qpos'] = model.jnt_qposadr[hinge]
    rdof = model.jnt_dofadr[hinge]
    model.jnt_stiffness[hinge] *= c['k_scale']
    info['nq_robot'] = base.nq
    if c['jf'] is not None:
        k = base.jnt_stiffness
        model.dof_frictionloss[rdof] = c['jf'] * (k / k[0]) ** c['jf_p']
    if c['jf_solimp'] is not None:
        model.dof_solimp[rdof, :2] = c['jf_solimp']
    if c.get('jf_solref') is not None:
        model.dof_solref[rdof] = c['jf_solref']
    if drive in ('C', 'E') and c['give'] == 'backbone':
        # slides at 0 hold the sum of the start tensions (all three tendons run to the tip)
        T0 = np.sum(np.broadcast_to(np.asarray(c['T0'], float), (3,)))
        sj = np.array([model.joint(f'comp_{b}').id for b in range(1, base.nbody) if base.body_jntnum[b]])
        info['bb_qpos'] = model.jnt_qposadr[sj]
        model.qpos_spring[info['bb_qpos']] = T0 / info['bb_k']
    tid = np.array([model.tendon(n).id for n in tnames])
    info['tendons'] = tid
    if drive in ('A', 'B', 'D'):
        model.tendon_frictionloss[tid] = c['tf'] if drive != 'D' else 0.0
        if drive == 'D':
            model.tendon_frictionloss[tid] = 0.0
        if c['tf_solimp'] is not None:
            model.tendon_solimp_fri[tid, :2] = c['tf_solimp']
    if drive in ('C', 'E'):
        jid = [model.joint(f'motor{i}').id for i in range(3)]
        info['m_qpos'] = model.jnt_qposadr[jid]
        info['m_dof'] = model.jnt_dofadr[jid]
        if c['give'] == 'common':
            cj = model.joint('carriage').id
            info['c_qpos'] = model.jnt_qposadr[cj]
            # carriage at 0 holds the sum of the start tensions
            model.qpos_spring[info['c_qpos']] = -np.sum(np.broadcast_to(np.asarray(c['T0'], float), (3,))) / c['comp_k']
        if drive == 'C':
            model.dof_solimp[info['m_dof'], 0] = 0.999
            model.dof_solimp[info['m_dof'], 1] = 0.9999
    info['cfg'] = c
    return model, info


def init_state(model, info, data, q_robot, floor_current=None):
    """Robot at q_robot; motors placed so each rope carries T0 (C, E)."""
    c = info['cfg']
    mujoco.mj_resetData(model, data)
    data.qpos[info['r_qpos']] = q_robot
    if c['drive'] in ('C', 'E'):
        tid = info['tendons']
        data.qpos[info['m_qpos']] = 0.0
        mujoco.mj_fwdPosition(model, data)
        L = data.ten_length[tid].copy()
        # spring rest (upper deadband edge) fixed by the model at x=0 so that L(x) - L0 = T0/k at the start pose
        T0 = np.broadcast_to(np.asarray(c['T0'], float), (3,))
        L0 = L - T0 / c['rope_k']
        model.tendon_lengthspring[tid, 0] = 0.0
        model.tendon_lengthspring[tid, 1] = L0
    mujoco.mj_forward(model, data)


def motor_angle(model, info, data, ref):
    if info['cfg']['drive'] in ('C', 'E'):
        return (data.qpos[info['m_qpos']] - ref) / R_SPOOL
    return (data.ten_length[info['tendons']] - ref) / R_SPOOL


def motor_ref(model, info, data):
    if info['cfg']['drive'] in ('C', 'E'):
        return data.qpos[info['m_qpos']].copy()
    return data.ten_length[info['tendons']].copy()


def rope_tension(model, info, data):
    tid = info['tendons']
    e = np.maximum(0.0, data.ten_length[tid] - model.tendon_lengthspring[tid, 1])
    return model.tendon_stiffness[tid] * e + model.tendon_stiffnesspoly[tid, 0] * e ** 2


class Stepper:
    def __init__(self, model, info, v_bus=24.0, R=2.2):
        self.m, self.info, self.c = model, info, info['cfg']
        self.v_bus, self.R = v_bus, R
        self.tid = info['tendons']
        self.gear = model.actuator_gear[:, 0].copy()
        d = self.c['drive']
        if d == 'B':
            model.tendon_solimp_fri[self.tid, 0] = 0.999
            model.tendon_solimp_fri[self.tid, 1] = 0.9999
        self.lo = model.actuator_ctrlrange[:, 0].copy()
        self.hi = model.actuator_ctrlrange[:, 1].copy()

    def vclip(self, cur, data):
        v = data.actuator_velocity
        return np.clip(cur, (-self.v_bus - v) / self.R, (self.v_bus - v) / self.R)

    def step(self, data, cur):
        m, c, d = self.m, self.c, self.c['drive']
        if d == 'A':
            data.ctrl[:] = self.vclip(cur, data)
        elif d == 'B':
            u = self.vclip(np.minimum(cur, 0.0), data)
            sliding = np.abs(data.ten_velocity[self.tid]) > c['veps']
            m.tendon_frictionloss[self.tid] = np.where(sliding, c['fk'], c['fs']) + c['hole'] * np.abs(self.gear * u)
            data.ctrl[:] = u
        elif d == 'C':
            u = self.vclip(np.clip(cur, self.lo, self.hi), data)
            T = rope_tension(m, self.info, data)
            v = data.qvel[self.info['m_dof']]
            sliding = np.abs(v) > c['veps']
            m.dof_frictionloss[self.info['m_dof']] = np.where(sliding, np.asarray(c['fk'], float), np.asarray(c['fs'], float)) + c['hole'] * T
            data.ctrl[:] = u
        elif d == 'D':
            data.ctrl[:] = np.clip(cur, self.lo, self.hi)
        elif d == 'E':
            if c['hole']:
                T = rope_tension(m, self.info, data)
                m.actuator_biasprm[:, 3] = (np.asarray(c['Fc']) + c['hole'] * T) / G
                m.actuator_biasprm[:, 4] = (np.asarray(c['Fs']) + c['hole'] * T) / G
            data.ctrl[:] = np.clip(cur, self.lo, self.hi)
        mujoco.mj_step(m, data)
