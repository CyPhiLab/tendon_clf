"""Evaluate named drive options on fit + validation runs: metrics JSON, plots, sim cost.

python evaluate.py options.json outdir
options.json: {"A": cfg, "B": cfg, ...}
"""
import sys, os, json, time
import numpy as np
import mujoco
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import harness as H
import drives


def cost(cfg, seconds=5.0):
    """Wall time per simulated second of the bare stepping loop (drive law included), one core."""
    r = H.load_run('20261103_174014')
    model, info = drives.build(cfg)
    data = mujoco.MjData(model)
    drives.init_state(model, info, data, r['q0'])
    st = drives.Stepper(model, info)
    n = int(seconds / model.opt.timestep)
    cur = r['cur'][int(5 / 0.001):int(5 / 0.001) + n]
    t0 = time.perf_counter()
    for k in range(n):
        st.step(data, cur[k])
    return (time.perf_counter() - t0) / seconds


def job(args):
    name, cfg, run = args
    o = H.score(cfg, run, keep=True)
    pairs, spur = H.match(o['slips_rec'], o['slips_sim'])
    return name, run, dict(
        marker_rms=o['marker_rms'], phase=o['phase'], motor_rms=o['motor_rms'].tolist(), ok=o['ok'],
        pairs=[(a, b) for a, b in pairs], spur=spur, spring_rec=o['spring_rec'].tolist(),
        spring_sim=o['spring_sim'].tolist(), t=o['t'][::20].tolist(), sim=o['sim'][::20].tolist(),
        rec=o['rec'][::20].tolist(), mt=H.load_run(run)['mt'][:len(o['err'])].tolist(),
        err=np.sqrt(np.nanmean(o['err'] ** 2, axis=1)).tolist())


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if k != 'm'}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


if __name__ == '__main__':
    opts = json.load(open(sys.argv[1])); out = sys.argv[2]
    runs = sys.argv[3].split(',') if len(sys.argv) > 3 else H.FIT + H.VAL
    os.makedirs(out, exist_ok=True)
    jobs = [(n, c, r) for n, c in opts.items() for r in runs]
    with Pool(4) as p:
        res = p.map(job, jobs)
    allres = {}
    for name, run, d in res:
        allres.setdefault(name, {})[run] = d
    for name, cfg in opts.items():
        allres[name]['_cost'] = cost(cfg)
    json.dump(_clean(allres), open(os.path.join(out, 'results.json'), 'w'))
    print('wrote', os.path.join(out, 'results.json'))
