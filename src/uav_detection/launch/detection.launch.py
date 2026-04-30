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
        description="Publish localized target estimates to /detections_3d.",
    )
    depth_localization_enabled_arg = DeclareLaunchArgument(
        "depth_localization_enabled",
        default_value="false",
        description="Use depth-image localization when ground-plane projection is unavailable.",
    )
    ground_projection_fallback_arg = DeclareLaunchArgument(
        "ground_projection_fallback",
        default_value="true",
        description="Raycast bbox footpoints onto the ground plane for RGB-only localization.",
    )
    prefer_ground_projection_arg = DeclareLaunchArgument(
        "prefer_ground_projection",
        default_value="true",
        description="Use RGB bbox ground-plane projection before depth when publishing /detections_3d.",
    )
    ground_plane_z_arg = DeclareLaunchArgument(
        "ground_plane_z",
        default_value="0.0",
        description="World-frame ENU Z value of the ground plane used for RGB-only localization.",
    )
    ground_projection_max_range_arg = DeclareLaunchArgument(
        "ground_projection_max_range_m",
        default_value="35.0",
        description="Maximum acceptable raycast range for RGB-only ground-plane localization.",
    )
    bbox_ground_y_fraction_arg = DeclareLaunchArgument(
        "bbox_ground_y_fraction",
        default_value="1.0",
        description="Vertical bbox fraction used as ground footpoint; 1.0 means bottom center.",
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
            "depth_localization_enabled": LaunchConfiguration("depth_localization_enabled"),
            "ground_projection_fallback": LaunchConfiguration("ground_projection_fallback"),
            "prefer_ground_projection": LaunchConfiguration("prefer_ground_projection"),
            "ground_plane_z": LaunchConfiguration("ground_plane_z"),
            "ground_projection_max_range_m": LaunchConfiguration("ground_projection_max_range_m"),
            "bbox_ground_y_fraction": LaunchConfiguration("bbox_ground_y_fraction"),
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
        depth_localization_enabled_arg,
        ground_projection_fallback_arg,
        prefer_ground_projection_arg,
        ground_plane_z_arg,
        ground_projection_max_range_arg,
        bbox_ground_y_fraction_arg,
        detection_node,
    ])
