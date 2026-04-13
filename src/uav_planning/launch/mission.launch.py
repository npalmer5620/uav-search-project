"""Launch the mission controller (spiral search + detection response)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("max_radius", default_value="20.0",
                              description="Spiral max radius in meters."),
        DeclareLaunchArgument("spacing", default_value="5.0",
                              description="Distance between spiral loops in meters."),
        DeclareLaunchArgument("angular_speed", default_value="0.3",
                              description="Spiral angular speed in rad/s."),
        DeclareLaunchArgument("altitude", default_value="-10.0",
                              description="Search altitude in NED (negative = up)."),
        DeclareLaunchArgument("detection_confidence_threshold", default_value="0.6",
                              description="Min confidence to trigger investigation."),
        DeclareLaunchArgument("investigate_duration", default_value="10.0",
                              description="Seconds to investigate each detection."),
        DeclareLaunchArgument("investigate_radius", default_value="3.0",
                              description="Orbit radius around target (0 = hover)."),
        DeclareLaunchArgument("tracking_match_distance", default_value="2.5",
                              description="Max 3D distance to associate detections to an existing track."),
        DeclareLaunchArgument("tracking_duplicate_distance", default_value="1.5",
                              description="Same-class detections within this distance are merged per frame."),
        DeclareLaunchArgument("tracking_timeout", default_value="1.0",
                              description="Seconds before an unseen track expires."),
        DeclareLaunchArgument("tracking_history_size", default_value="5",
                              description="Rolling history length used for position/confidence filtering."),
        DeclareLaunchArgument("tracking_min_hits", default_value="3",
                              description="Minimum updates before a track can trigger investigation."),
        DeclareLaunchArgument("tracking_min_age", default_value="0.5",
                              description="Minimum track age in seconds before promotion."),
        DeclareLaunchArgument("tracking_min_confidence", default_value="0.65",
                              description="Minimum filtered confidence before promotion."),
    ]

    mission_node = Node(
        package="uav_planning",
        executable="mission_controller",
        name="mission_controller",
        output="screen",
        parameters=[{
            "spiral.max_radius": LaunchConfiguration("max_radius"),
            "spiral.spacing": LaunchConfiguration("spacing"),
            "spiral.angular_speed": LaunchConfiguration("angular_speed"),
            "spiral.altitude": LaunchConfiguration("altitude"),
            "detection_confidence_threshold": LaunchConfiguration("detection_confidence_threshold"),
            "investigate_duration": LaunchConfiguration("investigate_duration"),
            "investigate_radius": LaunchConfiguration("investigate_radius"),
            "tracking.match_distance": LaunchConfiguration("tracking_match_distance"),
            "tracking.duplicate_distance": LaunchConfiguration("tracking_duplicate_distance"),
            "tracking.track_timeout": LaunchConfiguration("tracking_timeout"),
            "tracking.history_size": LaunchConfiguration("tracking_history_size"),
            "tracking.min_hits": LaunchConfiguration("tracking_min_hits"),
            "tracking.min_age": LaunchConfiguration("tracking_min_age"),
            "tracking.min_confidence": LaunchConfiguration("tracking_min_confidence"),
        }],
    )

    return LaunchDescription([*args, mission_node])
