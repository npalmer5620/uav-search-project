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
    imgsz_arg = DeclareLaunchArgument(
        "imgsz",
        default_value="640",
        description="Inference image size passed to the detector backend.",
    )
    frame_skip_arg = DeclareLaunchArgument(
        "frame_skip",
        default_value="0",
        description="Skip N frames between inferences (0 = process every frame).",
    )
    publish_viz_arg = DeclareLaunchArgument(
        "publish_viz",
        default_value="true",
        description="Publish annotated image to /camera/image_annotated.",
    )
    publish_3d_arg = DeclareLaunchArgument(
        "publish_3d",
        default_value="true",
        description="Fuse detections with depth and publish /detections_3d.",
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
            "imgsz":       LaunchConfiguration("imgsz"),
            "frame_skip":  LaunchConfiguration("frame_skip"),
            "image_topic": "/camera/image_raw",
            "camera_info_topic": "/camera/camera_info",
            "depth_topic": "/camera/depth/image_raw",
            "publish_viz": LaunchConfiguration("publish_viz"),
            "publish_3d": LaunchConfiguration("publish_3d"),
            "world_frame": "map",
        }],
    )

    return LaunchDescription([
        model_path_arg,
        confidence_arg,
        device_arg,
        imgsz_arg,
        frame_skip_arg,
        publish_viz_arg,
        publish_3d_arg,
        detection_node,
    ])
