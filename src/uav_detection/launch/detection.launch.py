"""Launch the YOLO detection node."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="yolo11n.pt",
        description="Path to YOLO weights (.pt file). Defaults to yolo11n (auto-download).",
    )
    confidence_arg = DeclareLaunchArgument(
        "confidence",
        default_value="0.4",
        description="Detection confidence threshold (0.0-1.0).",
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
        description="Inference device: 'cpu' or 'cuda'.",
    )
    publish_viz_arg = DeclareLaunchArgument(
        "publish_viz",
        default_value="true",
        description="Publish annotated image to /camera/image_annotated.",
    )

    detection_node = Node(
        package="uav_detection",
        executable="detection_node",
        name="detection_node",
        output="screen",
        parameters=[{
            "model_path":  LaunchConfiguration("model_path"),
            "confidence":  LaunchConfiguration("confidence"),
            "device":      LaunchConfiguration("device"),
            "image_topic": "/camera/image_raw",
            "publish_viz": LaunchConfiguration("publish_viz"),
        }],
    )

    return LaunchDescription([
        model_path_arg,
        confidence_arg,
        device_arg,
        publish_viz_arg,
        detection_node,
    ])
