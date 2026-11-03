from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROBOT = 'spirob_horz'

ROBOTS = {
    'spirob_horz': {
        'model_path': str(REPO_ROOT / 'mujoco_models' / 'spirob' / 'spirob_horz_control.xml'),
        'joint_stiffness': None,
        'init_key': 'init',
        'motor_ids': [10, 1, 2],
        'u_step_max': 0.02,
        'vicon_ip': '172.20.51.10',
        'vicon_rate_hz': 100.0,
        'base_height': 0.125,
        'settle_time': 5.0,
        'viewer_azimuth': 0.0,
        'viewer_elevation': -20.0,
        'viewer_pan_right': 0.08,

        # 'target_pos': [0.0,0.2,0.1],
        # 'target_pos': [0.0,0.15,0.15],
        # 'target_pos': [0.0,0.2,0.15],
        # 'target_pos': [0.0,0.2,0.2],

        # 'target_pos': [0.0,0.2,0.3],
        # 'target_pos': [0.0,0.1,0.3],
        # 'target_pos': [0.0,0.15,0.25],
        # 'target_pos': [0.0,0.25,0.1],

        'target_pos': [-0.05,0.2,0.15],
        # 'target_pos': [0.0,0.2,0.15],
        # 'target_pos': [0.0,0.2,0.2],
        # 'target_pos': [0.0,0.2,0.05],

        'site_name': 'marker_2_seg_16',  # site tracked by control_node (default 'ee')

        'site_names': [
            'marker_1_seg_2', 'marker_2_seg_2', 'marker_3_seg_2',
            'marker_1_seg_4', 'marker_2_seg_4', 'marker_3_seg_4',
            'marker_1_seg_6', 'marker_2_seg_6', 'marker_3_seg_6',
            'marker_1_seg_13', 'marker_2_seg_13', 'marker_3_seg_13',
            'marker_1_seg_16', 'marker_2_seg_16', 'marker_3_seg_16',
        ],

        'unpublished_sites': ['marker_1_seg_1', 'marker_2_seg_1', 'marker_3_seg_1'],
        'task_dim': 3,
        'K': 2000.0,
        'e': 0.06,  # 0.02 oscillates and diverges with the current-source model unless u_step_max >= 0.5
        'task_weight': 1.0,
        'reg_qdd': 2.0,
        'reg_u': 5.0,  # 5.0 when u was in volts; u in amps is 0.4545x smaller, so scale by 1/0.4545^2
        'reg_null': 0.1,
        'reg_dl': 1000.0,
        'pinv_rcond': 1e-2,
        'include_constraint_forces': True,
        # EKF. 5-6.5 mm covers the marker error no pose of the model removes (about 5 mm); lower and
        # the filter chases it. The velocity noise lets updates stop the model's drift while the robot
        # is held still by friction the model lacks (replays of the 20261103 runs).
        'measurement_noise': 5e-3,
        'process_dq_std': 1.0,
        'max_current': 1.9,
        'r_spool': 10.0 / 250.0,
        # AK drive bus voltage and winding resistance, for the voltage limit in current mode
        'v_bus': 24.0,
        'motor_resistance': 2.2,
    },
}
