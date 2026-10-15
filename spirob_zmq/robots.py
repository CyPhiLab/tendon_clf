from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROBOT = 'spirob_horz'

ROBOTS = {
    'spirob_horz': {
        'model_path': str(REPO_ROOT / 'mujoco_models' / 'spirob' / 'spirob_horz_control.xml'),
        'joint_stiffness': None,
        'joint_damping': None,
        'init_key': 'init',
        'motor_ids': [10, 1, 2],
        'u_step_max': 0.02,
        'vicon_ip': '172.20.54.13',
        'vicon_rate_hz': 100.0,
        'base_height': 0.125,
        'settle_time': 5.0,
        'target_pos': [0.0,0.2,0.1],
        'site_name': 'marker_2_seg_8',  # site tracked by control_node (default 'ee')
        'u_max': 0.0,
        'site_names': ['marker_1_seg_2', 'marker_2_seg_2', 'marker_3_seg_2',
                       'marker_1_seg_5', 'marker_2_seg_5', 'marker_3_seg_5',
                       'marker_1_seg_8', 'marker_2_seg_8', 'marker_3_seg_8'],
                    #    'marker_1_seg_11', 'marker_2_seg_11', 'marker_3_seg_11'],
                    #    'marker_1_seg_16', 'marker_2_seg_16', 'marker_3_seg_16'],
        'task_dim': 3,
        'K': 200.0,
        'e': 0.02,
        'task_weight': 1.0,
        'reg_qdd': 2.0,
        'reg_u': 5.0,
        'reg_null': 0.1,
        'reg_dl': 1000.0,
        'pinv_rcond': 1e-2,
        'include_constraint_forces': True,
        # 5 A peak current
        'motor_torque_limit': 0.127 * 5.0,
        'max_current': 1.9,
        'r_spool': 10.0 / 250.0,
    },
}
