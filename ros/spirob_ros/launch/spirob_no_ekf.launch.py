from launch import LaunchDescription
from launch_ros.actions import Node


MODEL_PATH = '/home/zach/huy/tendon_clf/mujoco_models/spirob/spirob_control.xml'


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spirob_ros',
            executable='get_state_node',
            name='get_state_node',
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

    ])


