"""Render the robot at its start pose and at the steady-state pose reached for each set point.

Poses are reconstructed from the Vicon markers (time-averaged over the window) with the repo's pose fit
(spirob_zmq.pose_init.MarkerModel.fit), for the median-error trial of each set point. The fit regularizes
toward the straight configuration, so the segments past the last marker ring (15-24, unobserved) stay
near straight. The EKF state is not used: it lets those unobserved segments curl. Vicon-measured tip
positions are drawn as small dark spheres and targets as translucent spheres.

Usage: python tools/hardware_figs/render_poses.py [--azimuth 150 --elevation -20 --no-floor] [--sheet]
(--sheet: camera-angle contact sheet instead of the per-set-point renders)
"""

import argparse
from pathlib import Path

import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import ROBOT_NQ, load
from runs import T_ON, T_SS, load_runs, setpoints

OUT = Path(__file__).with_name('figures')
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
GHOST = (0.62, 0.62, 0.60, 0.30)


def rgba(hex_, a=1.0):
    h = hex_.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], dtype=np.float32)


def representative(rr):
    ss = np.array([r.err[r.window(*T_SS)].mean() for r in rr])
    return rr[np.argsort(ss)[len(ss) // 2]]


def mean_q(r, lo, hi):
    sel = (r.d['est_sites_t'] >= lo) & (r.d['est_sites_t'] <= hi)
    return r.d['est_q'][sel].mean(0)


_FITTER = {}


def fit_q(r, lo, hi):
    """Joint angles fitted to the Vicon markers averaged over [lo, hi]; also per-marker residual [m]."""
    from spirob_zmq.pose_init import MarkerModel
    names = [str(n) for n in r.d['site_names']]
    if 'mm' not in _FITTER:
        model = load()
        _FITTER['mm'] = MarkerModel(model, names, q_prior=model.qpos0, lam=1e-3)
    mm = _FITTER['mm']
    sel = (r.d['markers_t'] >= lo) & (r.d['markers_t'] <= hi)
    z = r.d['markers'][sel].mean(0)
    q0 = mm.q_prior.copy()
    q0[:ROBOT_NQ] = mean_q(r, lo, hi)
    q, err = mm.fit(names, z, q0=q0, max_iter=200)
    return q[:ROBOT_NQ], err


def mean_tip(r, lo, hi):
    return r.tip[r.window(lo, hi)].mean(0)


class PoseRenderer:
    def __init__(self, width, height):
        self.model = load()
        m = self.model
        m.vis.global_.offwidth = max(width, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(height, m.vis.global_.offheight)
        m.vis.quality.shadowsize = 8192
        m.vis.quality.offsamples = 8
        # hide the scene's own target ball; targets are drawn per set point
        for name in ('target_vis',):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                m.geom_rgba[gid, 3] = 0
        self.renderer = mujoco.Renderer(m, height, width)
        self.opt = mujoco.MjvOption()
        for flag in ('mjVIS_TENDON', 'mjVIS_ACTUATOR'):
            self.opt.flags[getattr(mujoco.mjtVisFlag, flag)] = 0
        self.opt.sitegroup[:] = 0
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

    def data_for(self, q):
        d = mujoco.MjData(self.model)
        d.qpos[:ROBOT_NQ] = q
        mujoco.mj_forward(self.model, d)
        return d

    def _tint(self, scn, start, color):
        for g in scn.geoms[start:scn.ngeom]:
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == self.floor:
                continue
            g.rgba[:] = color

    def _sphere(self, scn, pos, radius, color):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, float),
                            np.eye(3).ravel(), np.asarray(color, np.float32))
        scn.ngeom += 1

    def render(self, ghost_q, poses, camera, show_floor=True):
        """poses: list of (q, color_rgba, target_xyz, vicon_tip_xyz)."""
        r = self.renderer
        base = self.data_for(ghost_q)
        r.update_scene(base, camera=camera, scene_option=self.opt)
        scn = r.scene
        self._tint(scn, 0, np.array(GHOST, np.float32))
        if not show_floor:
            for g in scn.geoms[:scn.ngeom]:
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == self.floor:
                    g.rgba[3] = 0
        cat = mujoco.mjtCatBit.mjCAT_DYNAMIC
        for q, color, target, tip in poses:
            d = self.data_for(q)
            start = scn.ngeom
            mujoco.mjv_addGeoms(self.model, d, self.opt, mujoco.MjvPerturb(), cat, scn)
            self._tint(scn, start, color)
            self._sphere(scn, target, 0.012, np.r_[color[:3], 0.4])
            self._sphere(scn, tip, 0.005, np.array([0.05, 0.05, 0.05, 1.0]))
        return r.render()


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def scene_poses(runs, sps):
    reps = [representative([x for x in runs if x.setpoint == sp]) for sp in sps]
    ghost, err = fit_q(reps[-1], 0, T_ON - 0.5)
    print(f'start pose ({reps[-1].stamp}): marker residual rms {1e3 * np.sqrt(np.mean(err ** 2)):.1f} mm')
    poses = []
    for k, (sp, r) in enumerate(zip(sps, reps)):
        q, err = fit_q(r, *T_SS)
        poses.append((q, rgba(COLORS[k]), r.target, mean_tip(r, *T_SS)))
        print(f'P{k + 1}: {sp}/{r.stamp}  marker residual rms {1e3 * np.sqrt(np.mean(err ** 2)):.1f} mm, '
              f'max {1e3 * err.max():.1f} mm')
    return ghost, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', action='store_true')
    ap.add_argument('--azimuth', type=float, default=150)
    ap.add_argument('--elevation', type=float, default=-20)
    ap.add_argument('--no-floor', action='store_true')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = load_runs()
    sps = setpoints(runs)
    ghost, poses = scene_poses(runs, sps)
    lookat = np.array([0.0, 0.17, 0.11])
    if args.sheet:
        pr = PoseRenderer(800, 600)
        views = [(az, el) for el in (-10, -25, -45) for az in (180, 150, 120, 90)]
        fig, axs = plt.subplots(3, 4, figsize=(16, 9))
        for ax, (az, el) in zip(axs.ravel(), views):
            ax.imshow(pr.render(ghost, poses, camera(lookat, 0.55, az, el)))
            ax.set_title(f'az {az}, el {el}')
            ax.axis('off')
        fig.tight_layout()
        out = Path(__file__).with_name('cache') / 'camera_sheet.png'
        fig.savefig(out, dpi=60)
        print('saved', out)
        return

    # one render per set point, same camera; cropped to the union of their content so scale is shared
    pr = PoseRenderer(1400, 1200)
    cam = camera(lookat, 0.50, args.azimuth, args.elevation)
    imgs = [pr.render(ghost, [p], cam, show_floor=not args.no_floor) for p in poses]
    content = np.any(np.stack(imgs).min(-1) < 245, axis=0)
    rows, cols = np.where(content)
    pad = 12
    r0, r1 = max(rows.min() - pad, 0), rows.max() + pad
    c0, c1 = max(cols.min() - pad, 0), cols.max() + pad
    OUT.mkdir(exist_ok=True)
    for k, img in enumerate(imgs):
        out = OUT / f'render_P{k + 1}.png'
        plt.imsave(out, img[r0:r1, c0:c1])
        print('saved', out)


if __name__ == '__main__':
    main()
