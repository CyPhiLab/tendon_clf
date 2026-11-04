"""Replay a recorded run through the model, open loop, and compare it with the robot.

Drives the robot model with the motor currents from a recording (MotorState 'commanded_current',
or 'current' with --measured) through the same voltage-limited current source the EKF and the
simulated plant use. Starts from the pose fitted to the markers at startup (INITIAL_STATE) when the
recording has one, else from the model's rest pose. Compares motor-shaft angle (tendon length
change / r_spool) and, when markers were recorded, every marker position (3D error), overall and
per phase of a current_profile_node run (launch --mode real_sysid).

-p KEY=VALUE overrides a robot-profile parameter of the model (joint_frictionloss,
tendon_frictionloss, joint_damping, joint_stiffness, ...); --sweep KEY=V1,V2,... compares values.

    python -m spirob_zmq.replay_hw ekf_results/real/<run>.jsonl --plot replay.png
    python -m spirob_zmq.replay_hw <run>.jsonl --sweep tendon_frictionloss=2.5,3.5,4.5 -p joint_frictionloss=0.02
"""

import argparse
import json
from multiprocessing import Pool

import mujoco
import numpy as np

from spirob_zmq.common import (INITIAL_STATE, MOTOR_COMMAND, MOTOR_STATE, SITE_MEASUREMENT, ProfileParams,
                               VoltageLimit, load_model, rest_state, step_current)
from spirob_zmq.robots import DEFAULT_ROBOT, ROBOTS


def load(path, motor_ids, measured):
    states = {i: [] for i in motor_ids}
    first_cmd, initial, meas, phases, site_names, events = None, None, [], [], None, []
    # Marker stamps from a real-time simulation are in simulated time, the motor stamps in wall time;
    # then fall back to the recorder's receive time for the marker topics
    clock_offset = None
    for line in open(path):
        r = json.loads(line)
        topic, m = r['topic'], r['msg']
        if topic in (INITIAL_STATE, SITE_MEASUREMENT):
            if clock_offset is None:
                clock_offset = abs(r['t'] - m['stamp']) > 3600.0
            if clock_offset:
                m = dict(m, stamp=r['t'])
        if topic == MOTOR_STATE and m['motor_id'] in states:
            states[m['motor_id']].append((m['stamp'], m['position'],
                                          m['current'] if measured else m.get('commanded_current', m['current'])))
        elif topic == MOTOR_COMMAND:
            if first_cmd is None:
                first_cmd = m['stamp']
            if 'phase' in m and (not phases or phases[-1][1] != m['phase']):
                phases.append((m['stamp'], m['phase']))
            if 'event' in m:
                events.append(m['event'])
        elif topic == INITIAL_STATE and initial is None:
            initial = m
        elif topic == SITE_MEASUREMENT:
            site_names = m['site_names']
            meas.append((m['stamp'], np.array([np.nan if v is None else v for v in m['data']], float).reshape(-1, 3)))
    if first_cmd is None:
        raise SystemExit(f'{path}: no {MOTOR_COMMAND} messages')
    return {k: np.array(v) for k, v in states.items()}, first_cmd, initial, meas, phases, site_names, events


def replay(path, robot=DEFAULT_ROBOT, measured=False, lead=0.2, overrides=None, from_rest=False):
    params = ProfileParams(robot, **(overrides or {}))
    motor_ids = list(params.declare_parameter('motor_ids'))
    r_spool = params.declare_parameter('r_spool')
    states, first_cmd, initial, meas, phases, site_names, events = load(path, motor_ids, measured)
    t1 = min(a[-1, 0] for a in states.values())
    start = initial['stamp'] if initial is not None and not from_rest else first_cmd - lead
    if not 0.0 < t1 - start < 3600.0:
        raise SystemExit(f'{path}: replay window of {t1 - start:.0f} s; the topics disagree on their clock')

    model = load_model(params)
    data = mujoco.MjData(model)
    if initial is not None and not from_rest:
        t0 = initial['stamp']
        data.qpos[:] = initial['q']
        mujoco.mj_forward(model, data)
    else:
        t0 = first_cmd - lead
        rest_state(params, model, data)
    vlimit = VoltageLimit(params)
    tendons = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    length0 = data.ten_length[tendons].copy()
    site_ids = [model.site(n).id for n in site_names] if site_names else []
    meas = [(s, z) for s, z in meas if t0 <= s <= t1]

    ts = np.arange(t0, t1, model.opt.timestep)
    sim = np.zeros((len(ts), len(motor_ids)))
    rec = np.zeros_like(sim)
    pos0 = [np.interp(t0, states[k][:, 0], states[k][:, 1]) for k in motor_ids]
    marker_err = np.full((len(meas), len(site_ids)), np.nan)
    j = 0
    for n, t in enumerate(ts):
        current = np.array([np.interp(t, states[k][:, 0], states[k][:, 2]) for k in motor_ids])
        step_current(model, data, current, 1, vlimit)
        sim[n] = (data.ten_length[tendons] - length0) / r_spool
        rec[n] = [np.interp(t, states[k][:, 0], states[k][:, 1]) - p0 for k, p0 in zip(motor_ids, pos0)]
        while j < len(meas) and meas[j][0] <= t + model.opt.timestep:
            marker_err[j] = np.linalg.norm(data.site_xpos[site_ids] - meas[j][1], axis=1)
            j += 1
    return {'t': ts - t0, 'sim': sim, 'rec': rec, 'motor_ids': motor_ids, 'site_names': site_names or [],
            'marker_t': np.array([s for s, _ in meas]) - t0, 'marker_err': marker_err,
            'phases': [(s - t0, name) for s, name in phases], 'events': events, 'start': 'fitted pose' if initial is not None
            and not from_rest else 'rest pose'}


def motor_window(r):
    moving = np.abs(r['rec']).max(axis=1) > 0.02
    return slice(np.argmax(moving), len(r['t']) - np.argmax(moving[::-1])) if moving.any() else slice(0, len(r['t']))


def summary(r):
    """Motor angle rms per motor (rad) and marker rms (m) over the run."""
    w = motor_window(r)
    motor = np.sqrt(np.mean((r['sim'][w] - r['rec'][w]) ** 2, axis=0))
    markers = np.sqrt(np.nanmean(r['marker_err'] ** 2)) if r['marker_err'].size else np.nan
    return motor, markers


def phase_table(r):
    """(phase, marker rms (m), worst marker name) for each profile phase."""
    out = []
    bounds = [s for s, _ in r['phases']] + [np.inf]
    for (start, name), end in zip(r['phases'], bounds[1:]):
        sel = (r['marker_t'] >= start) & (r['marker_t'] < end)
        if not sel.any():
            continue
        e = r['marker_err'][sel]
        if np.isnan(e).all():
            continue
        with np.errstate(invalid='ignore'):
            per = np.nanmean(e ** 2, axis=0)
        worst = int(np.argmax(np.where(np.isnan(per), -1.0, per)))
        out.append((name, np.sqrt(np.nanmean(e ** 2)), r['site_names'][worst]))
    return out


def _parse(items):
    out = {}
    for item in items:
        key, sep, raw = item.partition('=')
        if not sep:
            raise SystemExit(f'-p expects KEY=VALUE, got {item!r}')
        out[key] = json.loads(raw)
    return out


def _run(args):
    return replay(*args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log')
    parser.add_argument('--robot', default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    parser.add_argument('--measured', action='store_true', help='drive with measured instead of commanded current')
    parser.add_argument('--from-rest', action='store_true', help='start from the rest pose, not the fitted pose')
    parser.add_argument('-p', '--param', action='append', default=[], metavar='KEY=VALUE',
                        help='model parameter override (robot profile key)')
    parser.add_argument('--sweep', metavar='KEY=V1,V2,...', help='replay once per value and compare')
    parser.add_argument('--jobs', type=int, default=4, help='parallel replays for --sweep')
    parser.add_argument('--plot', help='write a PNG of simulated vs recorded motor angles and marker error')
    args = parser.parse_args(argv)
    overrides = _parse(args.param)

    if args.sweep:
        key, _, values = args.sweep.partition('=')
        values = [json.loads(v) for v in values.split(',')]
        jobs = [(args.log, args.robot, args.measured, 0.2, dict(overrides, **{key: v}), args.from_rest)
                for v in values]
        with Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(_run, jobs)
        print(f'{key:>22s} | marker rms [mm] | motor angle rms [rad]')
        for v, r in zip(values, results):
            motor, markers = summary(r)
            print(f'{json.dumps(v):>22s} | {1e3 * markers:15.1f} | ' + '  '.join(f'{e:.3f}' for e in motor))
        return

    r = replay(args.log, args.robot, args.measured, overrides=overrides, from_rest=args.from_rest)
    if r['events']:
        print('drive tests (current_profile_node):')
        for e in r['events']:
            print(f'  {e["text"]}')
        cur = {}
        for e in r['events']:
            if e.get('kind') in ('in', 'out'):
                cur.setdefault(e['motor_id'], {})[e['kind']] = e['current']
        for m, c in cur.items():
            if len(c) == 2:
                # Paying out at rest is resisted by the drive only; winding in also has to pull the robot
                # and the other tendons (on hardware a tendon cannot push, so pay-out is unloaded)
                print(f'  motor {m}: drive friction ~{c["out"]:.3f} A (pay-out at rest), wind-in needs '
                      f'{-c["in"]:.3f} A: the robot and the other tendons add ~{-c["in"] - c["out"]:+.3f} A')
    t, sim, rec, motor_ids = r['t'], r['sim'], r['rec'], r['motor_ids']
    w = motor_window(r)
    print(f'replayed {t[-1]:.2f} s from the {r["start"]}; motor angle change [rad], recorded / simulated')
    for tt in np.linspace(t[w][0], t[w][-1], 8 if np.any(rec) else 0):
        i = min(np.searchsorted(t, tt), len(t) - 1)
        print(f'  t={tt:5.2f}s  ' + '  '.join(f'motor {m}: {rec[i, j]:+6.2f} / {sim[i, j]:+6.2f}'
                                            for j, m in enumerate(motor_ids)))
    motor, markers = summary(r)
    if np.any(rec):
        print('rms error over the motion: ' + ', '.join(f'motor {m} {e:.3f} rad' for m, e in zip(motor_ids, motor)))
    else:
        print('no motor angles in the recording (hardware_node dry run); compare markers only')
    if r['marker_err'].size:
        per = np.sqrt(np.nanmean(r['marker_err'] ** 2, axis=0))
        print(f'marker error rms {1e3 * markers:.1f} mm; per marker [mm]: '
              + ', '.join(f'{n.replace("marker_", "")} {1e3 * e:.0f}' for n, e in zip(r['site_names'], per)))
        for name, e, worst in phase_table(r):
            print(f'  {name:22s} {1e3 * e:6.1f} mm   worst {worst}')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = len(motor_ids) + (1 if r['marker_err'].size else 0)
        fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(9, 2.2 * rows))
        axes = np.atleast_1d(axes)
        for j, (ax, m) in enumerate(zip(axes, motor_ids)):
            ax.plot(t, rec[:, j], 'k-', lw=1.2, label='recorded')
            ax.plot(t, sim[:, j], '--', lw=1.2, label='model')
            ax.set_ylabel(f'motor {m}\nangle [rad]')
            ax.grid(alpha=0.3)
        axes[0].legend(loc='best', fontsize=8)
        if r['marker_err'].size:
            ax = axes[-1]
            ax.plot(r['marker_t'], 1e3 * np.sqrt(np.nanmean(r['marker_err'] ** 2, axis=1)), 'k-', lw=1.0,
                    label='all markers')
            ax.plot(r['marker_t'], 1e3 * np.nanmax(r['marker_err'], axis=1), '-', lw=0.8, alpha=0.6, label='worst')
            ax.set_ylabel('model - Vicon\n[mm]')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
        for ax in axes:
            for start, name in r['phases']:
                if name not in ('wait', 'done'):
                    ax.axvline(start, color='0.7', lw=0.6)
        axes[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f'wrote {args.plot}')


if __name__ == '__main__':
    main()
