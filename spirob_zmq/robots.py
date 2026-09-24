from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROBOT = 'spirob_horz'

ROBOTS = {
    'spirob_horz': {
        'model_path': str(REPO_ROOT / 'mujoco_models' / 'spirob' / 'spirob_horz_control.xml'),
        'joint_stiffness': None,
        'joint_damping': None,
        'base_height': 0.55,
        'settle_time': 2.0,
        'target_pos': [-0.30, 0.08, 0.52],
        'u_max': 0.0,
        'site_names': ['site_seg_2', 'site_seg_5', 'site_seg_8', 'site_seg_11',
                       'site_seg_15', 'site_seg_25'],
        'task_dim': 3,
        'K': 200.0,
        'e': 0.02,
        'task_weight': 1.0,
        'reg_qdd': 0.2,
        'reg_u': 0.5,
        'reg_null': 0.1,
        'reg_dl': 1000.0,
        'pinv_rcond': 1e-2,
        'include_constraint_forces': True,
        # 5 A peak current
        'motor_torque_limit': 0.127 * 5.0,
        'max_current': 5.0,
        'r_spool': 10.0 / 250.0,
    },
}
