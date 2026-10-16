"""Replay a recorded hardware run through the model and compare motor angles.

Drives the robot model with the motor currents from a recording (MotorState 'commanded_current',
or 'current' with --measured) through the same voltage-limited current source the EKF and the
simulated plant use, starting from the model's rest pose. Prints simulated vs recorded motor-shaft
angle (tendon length change / r_spool) over the commanded window.

    python -m spirob_zmq.replay_hw ekf_results/real/<run>.jsonl
"""

import argparse
import json

import mujoco
import numpy as np

from spirob_zmq.common import (MOTOR_COMMAND, MOTOR_STATE, ProfileParams, VoltageLimit, load_model, rest_state,
                               step_current)
from spirob_zmq.robots import DEFAULT_ROBOT, ROBOTS


def load(path, motor_ids, measured):
    states = {i: [] for i in motor_ids}
    first_cmd = None
    for line in open(path):
        r = json.loads(line)
        if r['topic'] == MOTOR_STATE and r['msg']['motor_id'] in states:
            m = r['msg']
            states[m['motor_id']].append((m['stamp'], m['position'],
                                          m['current'] if measured else m.get('commanded_current', m['current'])))
        elif r['topic'] == MOTOR_COMMAND and first_cmd is None:
            first_cmd = r['msg']['stamp']
    if first_cmd is None:
        raise SystemExit(f'{path}: no {MOTOR_COMMAND} messages')
    return {k: np.array(v) for k, v in states.items()}, first_cmd


def replay(path, robot=DEFAULT_ROBOT, measured=False, lead=0.2):
    params = ProfileParams(robot)
    motor_ids = list(params.declare_parameter('motor_ids'))
    r_spool = params.declare_parameter('r_spool')
    states, first_cmd = load(path, motor_ids, measured)
    t0 = first_cmd - lead
    t1 = min(a[-1, 0] for a in states.values())

    model = load_model(params)
    data = mujoco.MjData(model)
    rest_state(params, model, data)
    vlimit = VoltageLimit(params)
    tendons = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    length0 = data.ten_length[tendons].copy()

    ts = np.arange(t0, t1, model.opt.timestep)
    sim = np.zeros((len(ts), len(motor_ids)))
    rec = np.zeros_like(sim)
    pos0 = [np.interp(t0, states[k][:, 0], states[k][:, 1]) for k in motor_ids]
    for n, t in enumerate(ts):
        current = np.array([np.interp(t, states[k][:, 0], states[k][:, 2]) for k in motor_ids])
        step_current(model, data, current, 1, vlimit)
        sim[n] = (data.ten_length[tendons] - length0) / r_spool
        rec[n] = [np.interp(t, states[k][:, 0], states[k][:, 1]) - p0 for k, p0 in zip(motor_ids, pos0)]
    return ts - t0, sim, rec, motor_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log')
    parser.add_argument('--robot', default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    parser.add_argument('--measured', action='store_true', help='drive with measured instead of commanded current')
    parser.add_argument('--plot', help='write a PNG of simulated vs recorded motor angles')
    args = parser.parse_args(argv)

    t, sim, rec, motor_ids = replay(args.log, args.robot, args.measured)
    moving = np.abs(rec).max(axis=1) > 0.02
    window = slice(np.argmax(moving), len(t) - np.argmax(moving[::-1])) if moving.any() else slice(0, len(t))
    print(f'replayed {t[-1]:.2f} s; motor angle change [rad], recorded / simulated')
    for tt in np.linspace(t[window][0], t[window][-1], 8):
        i = np.searchsorted(t, tt)
        print(f'  t={tt:5.2f}s  ' + '  '.join(f'motor {m}: {rec[i, j]:+6.2f} / {sim[i, j]:+6.2f}'
                                            for j, m in enumerate(motor_ids)))
    rms = np.sqrt(np.mean((sim[window] - rec[window]) ** 2, axis=0))
    print('rms error over the motion: ' + ', '.join(f'motor {m} {e:.3f} rad' for m, e in zip(motor_ids, rms)))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(motor_ids), 1, sharex=True, figsize=(8, 2.2 * len(motor_ids)))
        for j, (ax, m) in enumerate(zip(np.atleast_1d(axes), motor_ids)):
            ax.plot(t, rec[:, j], 'k-', lw=1.2, label='recorded')
            ax.plot(t, sim[:, j], '--', lw=1.2, label='model')
            ax.set_ylabel(f'motor {m}\nangle [rad]')
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].legend(loc='best', fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f'wrote {args.plot}')


if __name__ == '__main__':
    main()
