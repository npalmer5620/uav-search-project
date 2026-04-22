"""Launch ros_gz_bridge to bridge Gazebo clock and camera topics into ROS 2.

Use this when PX4 SITL + Gazebo are already running (e.g. via scripts/launch_sim.bash
or manually with `make px4_sitl gz_x500_mono_cam`).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('uav_bringup')
    bridge_config = os.path.join(pkg_dir, 'config', 'bridge.yaml')

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config}],
        output='screen',
    )

    return LaunchDescription([bridge])
