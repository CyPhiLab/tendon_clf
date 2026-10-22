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
        # u is the motor current [A]. With the current-source model the slew limit must let the
        # controller reverse within a few ticks: 0.05-0.5 A/tick settle in about 1 s with e=0.05 in sim
        'u_step_max': 0.05,
        'vicon_ip': '172.20.59.34',
        'vicon_rate_hz': 100.0,
        'base_height': 0.125,
        'settle_time': 5.0,
        'target_pos': [0.0,0.2,0.2],
        'site_name': 'marker_1_seg_16',  # site tracked by control_node (default 'ee')
        # 'site_name': 'ee',  # site tracked by control_node (default 'ee')

        'site_names': ['marker_1_seg_2', 'marker_2_seg_2', 'marker_3_seg_2',
                       'marker_1_seg_5', 'marker_2_seg_5', 'marker_3_seg_5',
                       'marker_1_seg_8', 'marker_2_seg_8', 'marker_3_seg_8',
                    #    'marker_1_seg_10', 'marker_2_seg_10', 'marker_3_seg_10',
                       'marker_1_seg_11', 'marker_2_seg_11', 'marker_3_seg_11',
                    #    'marker_1_seg_14', 'marker_2_seg_14', 'marker_3_seg_14',
                    #    'marker_1_seg_18', 'marker_2_seg_18', 'marker_3_seg_18',
                       'marker_1_seg_16', 'marker_2_seg_16', 'marker_3_seg_16'],
        'task_dim': 3,
        'K': 200.0,
        'e': 0.05,  # 0.02 oscillates and diverges with the current-source model unless u_step_max >= 0.5
        'task_weight': 1.0,
        'reg_qdd': 20.0,
        'reg_u': 200,  # 5.0 when u was in volts; u in amps is 0.4545x smaller, so scale by 1/0.4545^2
        'reg_null': 0.2,
        'reg_dl': 1000.0,
        'pinv_rcond': 1e-2,
        'include_constraint_forces': True,
        'max_current': 1.9,
        'r_spool': 10.0 / 250.0,
        # AK drive bus voltage and winding resistance, for the voltage limit in current mode
        'v_bus': 24.0,
        'motor_resistance': 2.2,
    },
}
