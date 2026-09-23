"""Per-robot defaults, selected with the ``robot`` parameter (``-p robot=spirob``).

A node's parameter is resolved as: explicit ``-p`` override > this profile >
the default written in the node. Values here are what differs between the
robots; everything else stays in the nodes.

``spirob_horz`` is the horizontal SpiRob from the CyPhiLab/spirob_mujoco
submodule (``external/spirob_mujoco``); its settings follow the measurements in
SPIROB_HORZ_NOTES.md on the claude/port-progress-shc9sv branch. ``spirob`` is
the original vertical model, with the values the ROS nodes used.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROBOT = 'spirob_horz'

ROBOTS = {
    'spirob_horz': {
        'model_path': str(REPO_ROOT / 'mujoco_models' / 'spirob' / 'spirob_horz_control.xml'),
        # Keep the model's own stiffness/damping (uniform 0.3 / 0.1 upstream).
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
    'spirob': {
        'model_path': str(REPO_ROOT / 'mujoco_models' / 'spirob' / 'spirob_control.xml'),
        # The XML has zero stiffness; the ROS nodes overwrote it with these.
        'joint_stiffness': 0.3,
        'joint_damping': 0.1,
        'base_height': None,
        'settle_time': 0.0,
        'target_pos': [0.2, 0.0, 0.2],
        'u_min': -1.0,
        'u_max': 0.0,
        # Reproduces the ROS control_node's objective and constraints.
        'task_dim': 6,
        'K': 500.0,
        'e': 0.05,
        'task_weight': 0.5,
        'reg_qdd': 0.5,
        'reg_u': 0.5,
        'reg_null': 0.0,
        'reg_dl': 1000.0,
        'pinv_rcond': None,
        'include_constraint_forces': False,
        'r_spool': 0.05,
    },
}
