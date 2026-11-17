"""Reduce the hardware set-point runs (one JSONL per run) to one compact .npz per run.

Usage: python tools/hardware_figs/extract.py <set_point_tracking dir> [cache dir]

Each run folder is named after the target tip position "x_y_z" (m, MuJoCo frame). Time is
seconds since the first /spirob/tracking_state message (same zero as the *_tracking.csv files).
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TOPICS = ('/spirob/tracking_state', '/spirob/site_measurement', '/spirob/robot_state',
          '/spirob/velocity_feedback', '/spirob/ekf_diagnostics', '/spirob/motor_command')


def extract(src, out):
    track = {'vicon': [], 'estimated': []}
    vic_t, vic, est_t, est_site, est_task, est_q = [], [], [], [], [], []
    fb_t, fb_i, fb_ref, fb_meas, fb_sat = [], [], [], [], []
    ekf_t, ekf_rms = [], []
    cmd_t, cmd_compute = [], []
    target, site_names = None, None
    n_partial_fb = 0
    with open(src) as f:
        for line in f:
            # cheap topic filter before parsing: motor_state is ~75% of the lines
            if '"/spirob/motor_state"' in line[:80]:
                continue
            rec = json.loads(line)
            topic, m, t = rec['topic'], rec['msg'], rec['t']
            if topic == '/spirob/tracking_state':
                if m.get('is_valid', True):
                    track[m['source']].append([m['stamp'], *m['position_m']])
                    target = m['target_m']
            elif topic == '/spirob/site_measurement':
                vic_t.append(m['stamp'])
                vic.append(m['data'])
                site_names = m['site_names']
            elif topic == '/spirob/robot_state':
                if m.get('is_valid', True) and 'site_pos' in m:
                    est_t.append(m['stamp'])
                    est_site.append(m['site_pos'])
                    est_task.append(m['task_pos'])
                    est_q.append(m['q'])
            elif topic == '/spirob/velocity_feedback':
                if 'sent_current_a' not in m:
                    n_partial_fb += 1
                    continue
                fb_t.append(t)
                fb_i.append(m['sent_current_a'])
                fb_ref.append(m['speed_reference_rad_s'])
                fb_meas.append(m['measured_speed_rad_s'])
                fb_sat.append(m['saturated'])
            elif topic == '/spirob/ekf_diagnostics':
                ekf_t.append(m['stamp'])
                ekf_rms.append(m['innovation_rms_m'])
            elif topic == '/spirob/motor_command':
                p = m.get('prediction', {})
                cmd_t.append(t)
                cmd_compute.append(p.get('prediction_compute_s', np.nan) + p.get('outer_compute_s', np.nan))
    vic_tr = np.asarray(track['vicon'], float)
    t0 = vic_tr[0, 0]
    est_tr = np.asarray(track['estimated'], float)
    a = lambda x: np.asarray(x, float)
    np.savez_compressed(
        out,
        target=a(target), site_names=np.asarray(site_names),
        tip_vicon_t=vic_tr[:, 0] - t0, tip_vicon=vic_tr[:, 1:],
        tip_est_t=est_tr[:, 0] - t0, tip_est=est_tr[:, 1:],
        markers_t=a(vic_t) - t0, markers=a(vic).reshape(len(vic_t), -1, 3),
        est_sites_t=a(est_t) - t0, est_sites=a(est_site).reshape(len(est_t), -1, 3)[:, :len(site_names)],
        est_task=a(est_task), est_q=a(est_q),
        fb_t=a(fb_t) - t0, current=a(fb_i), speed_ref=a(fb_ref), speed_meas=a(fb_meas),
        saturated=np.asarray(fb_sat, bool),
        ekf_t=a(ekf_t) - t0, innov_rms=a(ekf_rms),
        cmd_t=a(cmd_t) - t0, compute_s=a(cmd_compute),
    )
    if n_partial_fb:
        print(f'{src.name}: skipped {n_partial_fb} velocity_feedback messages without sent_current_a')
    return out


def main():
    root = Path(sys.argv[1])
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('cache')
    jobs = []
    for src in sorted(root.glob('*/*.jsonl')):
        out = cache / src.parent.name / f'{src.stem}.npz'
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((src, out))
    print(f'{len(jobs)} runs to extract')
    with ProcessPoolExecutor() as ex:
        for out in ex.map(extract, *zip(*jobs)) if jobs else []:
            print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
