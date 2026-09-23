from setuptools import find_packages, setup

package_name = 'spirob_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/spirob.launch.py']),
        ('share/' + package_name + '/launch', ['launch/spirob_no_ekf.launch.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zach',
    maintainer_email='hbp16@case.edu',
    description='ROS 2 nodes for the SpiRob tendon-driven robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hardware_node = spirob_ros.hardware_node:main',
            'state_estimation_node = spirob_ros.state_estimation_node:main',
            'control_node = spirob_ros.control_node:main',
            'virtual_measurement_node = spirob_ros.virtual_measurement_node:main',
            'simulation_node = spirob_ros.simulation_node:main',
            'get_state_node = spirob_ros.get_state_node:main',
        ],
    },
)
