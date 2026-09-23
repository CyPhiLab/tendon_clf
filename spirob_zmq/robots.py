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
        'target_pos': [-0.30, 0.08, 0.52],
        # dcmotor ctrl is a voltage.
        'u_min': -12.0,
        'u_max': 0.0,
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
        # Spool radius: the dcmotor's gear (43.478) is 1/r_spool, so the
        # actuator force is the motor output torque.
        'r_spool': 1.0 / 43.478,
    },
}
