from launch import LaunchDescription
from launch_ros.actions import Node


MODEL_PATH = '/home/zach/huy/tendon_clf/mujoco_models/spirob/spirob_control.xml'


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spirob_ros',
            executable='virtual_measurement_node',
            name='virtual_measurement_node',
            output='screen',
            parameters=[{
                'model_path': MODEL_PATH,
            }],
        ),

        Node(
            package='spirob_ros',
            executable='state_estimation_node',
            name='state_estimation_node',
            output='screen',
            parameters=[{
                'model_path': MODEL_PATH,
            }],
        ),

        Node(
            package='spirob_ros',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[{
                'model_path': MODEL_PATH,
            }],
        ),

        Node(
            package='spirob_ros',
            executable='simulation_node',
            name='simulation_node',
            output='screen',
            parameters=[{
                'model_path': MODEL_PATH,
            }],
        ),

        Node(
            package='spirob_ros',
            executable='hardware_node',
            name='hardware_node',
            output='screen',
            parameters=[{
                'model_path': MODEL_PATH,
            }],
        ),

    ])


# cd ~/huy/tendon_clf/ros
# colcon build --symlink-install
# source install/setup.bash
# ros2 launch spirob_ros spirob.launch.py