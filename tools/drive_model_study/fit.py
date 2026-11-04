"""Random / local search over drive parameters. Logs each evaluated config to fitlog_<tag>.jsonl.

python fit.py <tag> random '<base cfg json>' '<space json: {key: [lo, hi, "log"|"lin"]}>' N
python fit.py <tag> local  '<start cfg json>' '<space json>' rounds
"""
import sys, json, os
import numpy as np
from multiprocessing import Pool
import harness as H

def evaluate(cfgs, runs=H.FIT, pool=None):
    jobs = [(c, r, None) for r in runs[::-1] for c in cfgs]
    res = pool.map(H._job, jobs)
    out = []
    for i, c in enumerate(cfgs):
        outs = [res[j * len(cfgs) + i] for j in range(len(runs))][::-1]
        if any('error' in o for o in outs):
            out.append((c, 1e9, {'error': [o.get('error') for o in outs]}, outs))
            continue
        J, parts = H.objective(outs)
        out.append((c, J, parts, outs))
    return out

def log(tag, rows):
    with open(f'fitlog_{tag}.jsonl', 'a') as f:
        for c, J, parts, outs in rows:
            f.write(json.dumps(dict(cfg=c, J=J, parts=parts,
                                    runs={o['run']: dict(marker=o.get('marker_rms'), phase=o.get('phase'),
                                                         motor=list(map(float, o.get('motor_rms', []))))
                                          for o in outs if 'error' not in o})) + '\n')

def sample(space, rng):
    c = {}
    for k, (lo, hi, kind) in space.items():
        u = rng.random()
        c[k] = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))) if kind == 'log' else float(lo + u * (hi - lo))
    return c

if __name__ == '__main__':
    tag, mode, base, space, n = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng(sum(map(ord, tag)) + 7)
    with Pool(4) as pool:
        if mode == 'random':
            cfgs = [dict(base, **sample(space, rng)) for _ in range(n)]
            for i in range(0, len(cfgs), 4):
                rows = evaluate(cfgs[i:i + 4], pool=pool)
                log(tag, rows)
                for c, J, p, _ in rows:
                    print(f'J={J:8.1f} {json.dumps({k: round(c[k], 4) for k in space})} {json.dumps({k: round(v, 2) for k, v in p.items()}) if "error" not in p else p}', flush=True)
        else:
            cur = dict(base)
            (c0, J0, p0, _), = evaluate([cur], pool=pool); log(tag, [(c0, J0, p0, _)])
            print(f'start J={J0:.1f} {p0}', flush=True)
            step = {k: 0.5 if kind == 'log' else 0.25 * (hi - lo) for k, (lo, hi, kind) in space.items()}
            for rnd in range(n):
                improved = False
                for k, (lo, hi, kind) in space.items():
                    cands = []
                    for sgn in (-1, 1):
                        c = dict(cur)
                        c[k] = float(np.clip(cur[k] * np.exp(sgn * step[k]) if kind == 'log' else cur[k] + sgn * step[k], lo, hi))
                        if c[k] != cur[k]:
                            cands.append(c)
                    rows = evaluate(cands, pool=pool); log(tag, rows)
                    best = min(rows, key=lambda x: x[1]) if rows else None
                    if best and best[1] < J0:
                        cur, J0, p0 = best[0], best[1], best[2]; improved = True
                        print(f'  {k} -> {cur[k]:.4g}  J={J0:.1f} {json.dumps({kk: round(v, 2) for kk, v in p0.items()})}', flush=True)
                    else:
                        step[k] *= 0.6
                print(f'round {rnd}: J={J0:.1f} cfg={json.dumps(cur)}', flush=True)
                if not improved and max(step.values()) < 0.05:
                    break
