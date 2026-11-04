"""Open-loop replay of recorded currents through a drive variant, scored on markers, slips, spring-back."""
import os, sys, json, time, pickle
import numpy as np
import mujoco
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spirob_zmq.replay_hw as rh
import drives

DATA = os.environ.get('SYSID_DATA', 'data').rstrip('/') + '/'   # unzipped drive_model_bundle/data
CACHE = os.environ.get('SYSID_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache'))
FIT = ['20261103_174014', '20261103_191933']
VAL = ['20261103_target_0.0_0.2_0.15_trial1', '20261103_target_0.0_0.2_0.15_trial2',
       '20261103_target_0.0_0.2_0.15_trial3', '20261103_target_-0.05_0.2_0.15_trial1',
       '20261103_target_-0.05_0.2_0.15_trial2']
IDS = [10, 1, 2]


def load_run(run):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, run + '.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    states, first_cmd, initial, meas, phases, names, events = rh.load(DATA + run + '.jsonl', IDS, False)
    t0 = initial['stamp']
    t1 = min(a[-1, 0] for a in states.values())
    h = 0.001
    ts = np.arange(t0, t1, h)
    cur = np.stack([np.interp(ts, states[k][:, 0], states[k][:, 2]) for k in IDS], 1)
    pos0 = [np.interp(t0, states[k][:, 0], states[k][:, 1]) for k in IDS]
    rec = np.stack([np.interp(ts, states[k][:, 0], states[k][:, 1]) - p0 for k, p0 in zip(IDS, pos0)], 1)
    meas = [(s, z) for s, z in meas if t0 <= s <= t1]
    r = dict(run=run, t=ts - t0, cur=cur, rec=rec, q0=np.array(initial['q']), names=names,
             mt=np.array([s for s, _ in meas]) - t0, mz=np.array([z for _, z in meas]),
             phases=[(s - t0, n) for s, n in phases], events=events,
             raw={k: np.column_stack([states[k][:, 0] - t0, states[k][:, 1] - pos0[i], states[k][:, 2]]) for i, k in enumerate(IDS)})
    pickle.dump(r, open(p, 'wb'))
    return r


def simulate(cfg, run, t_max=None, every=10):
    """Replay; returns sim motor angles (1 ms), marker errors at marker stamps, wall time per sim second."""
    r = load_run(run) if isinstance(run, str) else run
    model, info = drives.build(cfg)
    data = mujoco.MjData(model)
    drives.init_state(model, info, data, r['q0'])
    st = drives.Stepper(model, info)
    ref = drives.motor_ref(model, info, data)
    site_ids = [model.site(n).id for n in r['names']]
    n = len(r['t']) if t_max is None else min(len(r['t']), int(t_max / 0.001))
    sim = np.zeros((n, 3))
    err = np.full((len(r['mt']), len(site_ids)), np.nan)
    xpos = np.full((len(r['mt']), len(site_ids), 3), np.nan)
    j = 0
    sub = int(round(model.opt.timestep / 0.001)) if model.opt.timestep > 0.001 else 1
    wall0 = time.perf_counter()
    k = 0
    ok = True
    while k < n:
        st.step(data, r['cur'][k])
        kk = min(k + sub, n)
        sim[k:kk] = drives.motor_angle(model, info, data, ref)
        t = r['t'][kk - 1]
        while j < len(r['mt']) and r['mt'][j] <= t + 0.001:
            xpos[j] = data.site_xpos[site_ids]
            err[j] = np.linalg.norm(data.site_xpos[site_ids] - r['mz'][j], axis=1)
            j += 1
        k = kk
        if not np.isfinite(data.qacc).all() or data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            ok = False
            break
    wall = time.perf_counter() - wall0
    return dict(sim=sim, err=err[:j], xpos=xpos[:j], n=k, ok=ok, wall_per_s=wall / max(1e-9, k * 0.001))


# ---------------------------------------------------------------- metrics
def slips(t, theta, cur, vth=0.5, win=0.02, min_size=0.03, gap=0.06):
    """Fast slips: |d theta / dt| over a 20 ms window above vth; returns list of dicts."""
    h = t[1] - t[0]
    w = max(1, int(round(win / h)))
    out = []
    for m in range(theta.shape[1]):
        th = theta[:, m]
        v = np.zeros_like(th)
        v[:-w] = (th[w:] - th[:-w]) / (w * h)
        mov = np.abs(v) > vth
        idx = np.flatnonzero(mov)
        if not len(idx):
            continue
        # group indices
        groups = []
        s = idx[0]; p = idx[0]
        for i in idx[1:]:
            if (i - p) * h > gap:
                groups.append((s, p)); s = i
            p = i
        groups.append((s, p))
        for s, e in groups:
            e2 = min(len(th) - 1, e + w)
            size = th[e2] - th[s]
            if abs(size) < min_size:
                continue
            out.append(dict(motor=IDS[m], m=m, t=t[s], cur=cur[s, m], size=size, dur=(e2 - s) * h))
    return sorted(out, key=lambda x: x['t'])


def unloading_springback(t, theta, cur, min_len=0.3):
    """Largest motor-angle change toward pay-out while the (negative) current is being reduced."""
    h = t[1] - t[0]
    w = int(round(0.05 / h))
    out = np.zeros(theta.shape[1])
    for m in range(theta.shape[1]):
        c = np.convolve(cur[:, m], np.ones(w) / w, mode='same')
        dc = np.gradient(c, h)
        unl = (dc > 0.02) & (c < -0.12)
        idx = np.flatnonzero(np.diff(np.r_[0, unl.astype(int), 0]))
        for s, e in zip(idx[::2], idx[1::2]):
            if (e - s) * h < min_len:
                continue
            e = min(e, len(t) - 1)
            out[m] = max(out[m], theta[e, m] - theta[s, m])
    return out


def phase_rms(r, err):
    rows = []
    bounds = [s for s, _ in r['phases']] + [np.inf]
    mt = r['mt'][:len(err)]
    for (s, name), e in zip(r['phases'], bounds[1:]):
        sel = (mt >= s) & (mt < e)
        if not sel.any() or np.isnan(err[sel]).all():
            continue
        rows.append((name, float(np.sqrt(np.nanmean(err[sel] ** 2)))))
    return rows


def score(cfg, run, t_max=None, keep=False):
    r = load_run(run)
    s = simulate(cfg, r, t_max)
    n = s['n']
    t, rec, cur = r['t'][:n], r['rec'][:n], r['cur'][:n]
    sim = s['sim'][:n]
    out = dict(run=run, cfg=cfg, ok=s['ok'], wall_per_s=s['wall_per_s'])
    out['marker_rms'] = float(np.sqrt(np.nanmean(s['err'] ** 2))) if s['err'].size else np.nan
    out['marker_mean'] = float(np.nanmean(np.sqrt(np.nanmean(s['err'] ** 2, axis=1)))) if s['err'].size else np.nan
    out['phase'] = phase_rms(r, s['err'])
    out['motor_rms'] = np.sqrt(np.mean((sim - rec) ** 2, axis=0))
    rs, ss = slips(t, rec, cur), slips(t, sim, cur)
    out['slips_rec'], out['slips_sim'] = rs, ss
    out['spring_rec'], out['spring_sim'] = unloading_springback(t, rec, cur), unloading_springback(t, sim, cur)
    if keep:
        out['sim'], out['err'], out['t'], out['rec'] = sim, s['err'], t, rec
    return out


def match(rs, ss, tol=2.0):
    """Pair recorded with simulated slips (same motor and direction, nearest start within tol)."""
    used = set(); pairs = []
    for a in rs:
        best = None
        for i, b in enumerate(ss):
            if i in used or b['m'] != a['m'] or np.sign(b['size']) != np.sign(a['size']):
                continue
            dt = abs(b['t'] - a['t'])
            if dt <= tol and (best is None or dt < best[0]):
                best = (dt, i)
        if best:
            used.add(best[1]); pairs.append((a, ss[best[1]]))
        else:
            pairs.append((a, None))
    spurious = [b for i, b in enumerate(ss) if i not in used]
    return pairs, spurious


def report(o, verbose=True):
    pairs, spur = match(o['slips_rec'], o['slips_sim'])
    lines = [f"{o['run'][-24:]}: marker rms {1e3*o['marker_rms']:.1f} mm | motor rms {np.round(o['motor_rms'], 3)} | "
             f"wall {o['wall_per_s']*1e3:.0f} ms/s{'' if o['ok'] else ' UNSTABLE'}"]
    if verbose:
        lines.append('  phases: ' + ', '.join(f'{n} {1e3*e:.0f}' for n, e in o['phase']))
        for a, b in pairs:
            lines.append(f"  slip m{a['motor']:>2d} t={a['t']:6.2f} rec {a['cur']:+.3f}A {a['size']:+.3f}rad | "
                         + (f"sim t={b['t']:6.2f} {b['cur']:+.3f}A {b['size']:+.3f}rad" if b else 'sim: none'))
        for b in spur:
            lines.append(f"  spurious sim slip m{b['motor']} t={b['t']:.2f} {b['cur']:+.3f}A {b['size']:+.3f}rad dur {b['dur']:.2f}s")
        lines.append(f"  spring-back on unloading [rad] rec {np.round(o['spring_rec'], 3)} sim {np.round(o['spring_sim'], 3)}")
    return '\n'.join(lines)


def _job(args):
    os.nice(5)
    cfg, run, t_max = args
    try:
        return score(cfg, run, t_max)
    except Exception as e:
        return dict(run=run, cfg=cfg, error=repr(e))


if __name__ == '__main__':
    from multiprocessing import Pool
    cfgs = json.loads(sys.argv[1])
    runs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else FIT
    t_max = float(sys.argv[3]) if len(sys.argv) > 3 else None
    if isinstance(cfgs, dict):
        cfgs = [cfgs]
    jobs = [(c, r, t_max) for c in cfgs for r in runs]
    with Pool(min(4, len(jobs))) as p:
        res = p.map(_job, jobs)
    for o in res:
        print(json.dumps(o['cfg']))
        print(o['error'] if 'error' in o else report(o))


def objective(outs):
    """Composite: mean marker rms [mm] + onset/size/spurious-slip penalties [mm-equivalent]."""
    mk = np.mean([1e3 * o['marker_rms'] for o in outs])
    on, sz, sp, n = 0.0, 0.0, 0.0, 0
    for o in outs:
        pairs, spur = match(o['slips_rec'], o['slips_sim'])
        for a, b in pairs:
            n += 1
            if b is None:
                on += 3; sz += 3
            else:
                on += min(abs(b['cur'] - a['cur']) / 0.03, 3)
                sz += min(abs(b['size'] - a['size']) / 0.1, 3)
        sp += sum(min(abs(b['size']) / 0.2, 3) for b in spur)
    if not all(o['ok'] for o in outs):
        mk += 1000
    return mk + 10 * (on + sz + sp) / max(n, 1), dict(marker=mk, onset=on / max(n, 1), size=sz / max(n, 1), spur=sp / max(n, 1))
