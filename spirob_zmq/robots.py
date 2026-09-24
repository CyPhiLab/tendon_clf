"""Per-robot defaults, selected with the ``robot`` parameter (``-p robot=spirob_horz``).

A node's parameter is resolved as: explicit ``-p`` override > this profile >
the default written in the node. Values here are what differs between the
robots; everything else stays in the nodes.

``spirob_horz`` is the horizontal SpiRob from the CyPhiLab/spirob_mujoco
submodule (``external/spirob_mujoco``); its settings follow the measurements in
SPIROB_HORZ_NOTES.md.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROBOT = 'spirob_horz'

ROBOTS = {
    'spirob_horz': {
        'model_path': str(REPO_ROOT / 'mujoco_models' / 'spirob' / 'spirob_horz_control.xml'),
        # None keeps the model's own stiffness/damping (uniform 0.3 / 0.1 upstream).
        'joint_stiffness': None,
        'joint_damping': None,
        # Raise the base so the arm hangs clear of the floor, and start from the
        # gravity-settled pose: qpos = 0 is where B is worst conditioned (cond ~950).
        'base_height': 0.55,
        'settle_time': 2.0,
        # Set-point pos1 of the branch's target circle (y-z plane at x = -0.30).
        # With the 5 A peak limit all four circle targets are reached to ~1 mm
        # in 0.3-1.2 s (controller alone); at the rated 1.9 A they are not.
        'target_pos': [-0.30, 0.08, 0.52],
        # Pull-only: the model's ctrlrange allows +2 V, which pushes on a
        # cable in simulation; the real tendons can only pull.
        'u_max': 0.0,
        # The six mocap markers on the arm (spirob.xml, group 4)
        'site_names': ['site_seg_2', 'site_seg_5', 'site_seg_8', 'site_seg_11',
                       'site_seg_15', 'site_seg_25'],
        # Controller, as tuned on the branch.
        'task_dim': 3,
        'K': 200.0,
        'e': 0.01,
        'task_weight': 1.0,
        'reg_qdd': 0.2,
        'reg_u': 0.5,
        'reg_null': 0.1,
        'reg_dl': 1000.0,
        'pinv_rcond': 1e-2,
        'include_constraint_forces': True,
        # Peak motor current is 5 A (peak output torque 7 N m); the model's
        # saturation="0.2413 1.9 0" is the rated 1.9 A. Rotor torque limit =
        # motorconst 0.127 N m/A x 5 A = 0.635 N m (6.35 N m after the 10:1
        # gearbox, ~159 N of tendon tension). Candidate change for upstream.
        'motor_torque_limit': 0.127 * 5.0,
        'max_current': 5.0,
        # The dcmotor models the AK rotor (motorconst 0.127 N m/A)
        # with gear 250 = gearbox 10 / spool radius, so r_spool = 10 / 250.
        'r_spool': 10.0 / 250.0,
    },
}
