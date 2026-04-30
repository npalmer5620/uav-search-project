#!/usr/bin/env python3
"""YOLO detector with depth or ground-plane 3D target localization."""

import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics not installed — run: pip install ultralytics --break-system-packages"
    ) from exc


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

TARGET_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


def quat_xyzw_to_rotmat(quat: Quaternion) -> np.ndarray:
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)

    x /= n
    y /= n
    z /= n
    w /= n

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def intersect_ray_with_horizontal_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_z: float,
    *,
    min_distance_m: float,
    max_distance_m: float,
) -> tuple[np.ndarray, float] | None:
    """Intersect a world-frame ray with a horizontal plane."""
    direction_z = float(direction[2])
    if abs(direction_z) < 1e-6:
        return None

    t = (float(plane_z) - float(origin[2])) / direction_z
    if not math.isfinite(t) or t <= 0.0:
        return None

    point = origin + direction * t
    distance = float(np.linalg.norm(point - origin))
    if distance < min_distance_m or distance > max_distance_m:
        return None

    return point, distance


def project_bbox_footpoint_to_horizontal_plane(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_origin_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    plane_z: float,
    min_distance_m: float,
    max_distance_m: float,
    bbox_ground_y_fraction: float = 1.0,
) -> tuple[np.ndarray, float] | None:
    """Raycast a bbox footpoint from optical camera frame to a horizontal plane."""
    if fx == 0.0 or fy == 0.0:
        return None

    clamped_fraction = min(1.0, max(0.0, float(bbox_ground_y_fraction)))
    u = min(max((x1 + x2) * 0.5, 0.0), float(max(0, image_width - 1)))
    bbox_v = y1 + (y2 - y1) * clamped_fraction
    v = min(max(bbox_v, 0.0), float(max(0, image_height - 1)))

    ray_camera = np.array([
        (u - cx) / fx,
        (v - cy) / fy,
        1.0,
    ], dtype=float)
    ray_camera_norm = np.linalg.norm(ray_camera)
    if ray_camera_norm <= 0.0:
        return None
    ray_camera /= ray_camera_norm

    ray_world = camera_rotation_world @ ray_camera
    ray_world_norm = np.linalg.norm(ray_world)
    if ray_world_norm <= 0.0:
        return None
    ray_world /= ray_world_norm

    return intersect_ray_with_horizontal_plane(
        camera_origin_world,
        ray_world,
        plane_z,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
    )


class DetectionNode(Node):
    """Run 2D YOLO and estimate 3D target positions."""

    def __init__(self) -> None:
        super().__init__("detection_node")

        self.declare_parameter("model_path", "yolo11n.pt")
        self.declare_parameter("confidence", 0.4)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("frame_skip", 0)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("pose_topic", "/uav/pose")
        self.declare_parameter("publish_viz", True)
        self.declare_parameter("publish_3d", True)
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("depth_localization_enabled", False)
        self.declare_parameter("ground_projection_fallback", True)
        self.declare_parameter("prefer_ground_projection", True)
        self.declare_parameter("ground_plane_z", 0.0)
        self.declare_parameter("ground_projection_min_range_m", 0.5)
        self.declare_parameter("ground_projection_max_range_m", 35.0)
        self.declare_parameter("bbox_ground_y_fraction", 1.0)

        model_path = self.get_parameter("model_path").value
        self.confidence_threshold = float(self.get_parameter("confidence").value)
        self.device = self.get_parameter("device").value
        device = self.device
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.frame_skip = max(0, int(self.get_parameter("frame_skip").value))
        image_topic = self.get_parameter("image_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        self.publish_viz = bool(self.get_parameter("publish_viz").value)
        self.publish_3d = bool(self.get_parameter("publish_3d").value)
        self.world_frame = self.get_parameter("world_frame").value
        self.depth_localization_enabled = bool(
            self.get_parameter("depth_localization_enabled").value
        )
        self.ground_projection_fallback = bool(
            self.get_parameter("ground_projection_fallback").value
        )
        self.prefer_ground_projection = bool(
            self.get_parameter("prefer_ground_projection").value
        )
        self.ground_plane_z = float(self.get_parameter("ground_plane_z").value)
        self.ground_projection_min_range_m = max(
            0.0,
            float(self.get_parameter("ground_projection_min_range_m").value),
        )
        self.ground_projection_max_range_m = max(
            self.ground_projection_min_range_m,
            float(self.get_parameter("ground_projection_max_range_m").value),
        )
        self.bbox_ground_y_fraction = min(
            1.0,
            max(0.0, float(self.get_parameter("bbox_ground_y_fraction").value)),
        )

        self.get_logger().info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path, task="detect")
        # .to() is only valid for PyTorch (.pt) models; ONNX/TensorRT models
        # receive the device at predict time instead.
        if model_path.endswith(".pt"):
            self.model.to(device)
        self.frame_index = 0
        self.get_logger().info(
            f"YOLO model loaded: path={model_path}, backend={type(self.model.model).__name__}, imgsz={self.imgsz}, frame_skip={self.frame_skip}"
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.camera_info: Optional[CameraInfo] = None
        self.depth_image: Optional[np.ndarray] = None
        self.depth_encoding: Optional[str] = None
        self.depth_frame_id: Optional[str] = None
        self.latest_uav_pose: Optional[PoseStamped] = None
        self.camera_translation_base = np.array([0.12, 0.03, 0.242], dtype=float)
        self.camera_rotation_base = quat_xyzw_to_rotmat(
            Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5)
        )

        self.create_subscription(Image, image_topic, self.image_callback, qos)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos)
        self.create_subscription(Image, depth_topic, self.depth_callback, qos)
        self.create_subscription(PoseStamped, pose_topic, self.pose_callback, qos)

        self.det_pub = self.create_publisher(Detection2DArray, "/detections", 10)
        self.det3d_pub = self.create_publisher(Detection3DArray, "/detections_3d", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detections_3d_markers", 10)

        if self.publish_viz:
            self.viz_pub = self.create_publisher(Image, "/camera/image_annotated", 10)

        self.get_logger().info(
            "DetectionNode ready — "
            f"rgb={image_topic}, camera_info={camera_info_topic}, depth={depth_topic}, pose={pose_topic}, "
            f"depth_localization_enabled={self.depth_localization_enabled}, "
            f"ground_projection_fallback={self.ground_projection_fallback}, "
            f"prefer_ground_projection={self.prefer_ground_projection}"
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def depth_callback(self, msg: Image) -> None:
        try:
            if msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
                depth *= 0.001
            else:
                self.get_logger().warning(f"Unsupported depth encoding: {msg.encoding}")
                return
        except ValueError as exc:
            self.get_logger().error(f"Depth conversion error: {exc}")
            return

        self.depth_image = depth
        self.depth_encoding = msg.encoding
        self.depth_frame_id = msg.header.frame_id

    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_uav_pose = msg

    def image_callback(self, msg: Image) -> None:
        self.frame_index += 1
        if self.frame_skip > 0 and ((self.frame_index - 1) % (self.frame_skip + 1)) != 0:
            return

        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgra8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif msg.encoding != "bgr8":
                self.get_logger().error(f"Unsupported RGB encoding: {msg.encoding}")
                return
        except ValueError as exc:
            self.get_logger().error(f"Image conversion error: {exc}")
            return

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
        )[0]

        det2d_array = Detection2DArray()
        det2d_array.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)

        det3d_array = Detection3DArray()
        det3d_array.header = Header(stamp=msg.header.stamp, frame_id=self.world_frame)

        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header = det3d_array.header
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        annotated = frame.copy()
        detections_found = 0
        depth_localized = 0
        ground_localized = 0

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else "unknown"
            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            det2d = Detection2D()
            det2d.header = det2d_array.header
            det2d.bbox.center.position.x = cx
            det2d.bbox.center.position.y = cy
            det2d.bbox.size_x = bw
            det2d.bbox.size_y = bh

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = confidence
            det2d.results.append(hypothesis)
            det2d_array.detections.append(det2d)

            point_world = None
            depth_m = None
            range_m = None
            localization_source = ""
            if self.publish_3d:
                ground_projection = None
                if self.ground_projection_fallback:
                    ground_projection = self.project_bbox_to_ground_plane(
                        x1,
                        y1,
                        x2,
                        y2,
                        frame.shape[1],
                        frame.shape[0],
                    )

                if self.prefer_ground_projection and ground_projection is not None:
                    point_world, range_m = ground_projection
                    localization_source = "ground"

                if point_world is None and self.depth_localization_enabled:
                    depth_m = self.sample_depth(cx, cy, frame.shape[1], frame.shape[0])
                    if depth_m is not None:
                        point_world = self.project_pixel_to_world(cx, cy, depth_m)
                        range_m = depth_m
                        localization_source = "depth"

                if point_world is None and ground_projection is not None:
                    point_world, range_m = ground_projection
                    localization_source = "ground"

                if point_world is not None and range_m is not None:
                    det3d = self.make_detection_3d(
                        msg.header.stamp,
                        class_name,
                        confidence,
                        point_world,
                        range_m,
                        localization_source,
                    )
                    det3d_array.detections.append(det3d)
                    marker_array.markers.extend(
                        self.make_markers(
                            msg.header.stamp,
                            detections_found,
                            class_name,
                            confidence,
                            range_m,
                            point_world,
                            localization_source,
                        )
                    )
                    if localization_source == "depth":
                        depth_localized += 1
                    elif localization_source == "ground":
                        ground_localized += 1

            label = f"{class_name} {confidence:.2f}"
            if range_m is not None:
                suffix = "d" if localization_source == "depth" else "g"
                label += f" {range_m:.1f}m/{suffix}"

            self.draw_detection(annotated, x1, y1, x2, y2, label)
            detections_found += 1

        self.det_pub.publish(det2d_array)

        if self.publish_3d:
            self.det3d_pub.publish(det3d_array)
            self.marker_pub.publish(marker_array)

        if self.publish_viz:
            viz_msg = Image()
            viz_msg.header = det2d_array.header
            viz_msg.height = annotated.shape[0]
            viz_msg.width = annotated.shape[1]
            viz_msg.encoding = "bgr8"
            viz_msg.step = annotated.shape[1] * 3
            viz_msg.data = annotated.tobytes()
            self.viz_pub.publish(viz_msg)

        if detections_found > 0:
            self.get_logger().info(
                f"{detections_found} target(s) detected, {len(det3d_array.detections)} localized "
                f"(depth={depth_localized}, ground={ground_localized})"
            )

    def sample_depth(self, u: float, v: float, rgb_width: int, rgb_height: int) -> Optional[float]:
        if self.depth_image is None:
            return None

        depth_height, depth_width = self.depth_image.shape
        du = int(round(u * depth_width / rgb_width))
        dv = int(round(v * depth_height / rgb_height))

        if du < 0 or du >= depth_width or dv < 0 or dv >= depth_height:
            return None

        half_window = 2
        u0 = max(0, du - half_window)
        u1 = min(depth_width, du + half_window + 1)
        v0 = max(0, dv - half_window)
        v1 = min(depth_height, dv + half_window + 1)

        patch = self.depth_image[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.2) & (patch < 100.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def project_pixel_to_world(self, u: float, v: float, depth_m: float) -> Optional[np.ndarray]:
        if self.camera_info is None:
            return None
        if self.latest_uav_pose is None:
            return None

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        if fx == 0.0 or fy == 0.0:
            return None

        point_camera = np.array([
            (u - cx) * depth_m / fx,
            (v - cy) * depth_m / fy,
            depth_m,
        ])

        point_base = self.camera_translation_base + self.camera_rotation_base @ point_camera

        pose = self.latest_uav_pose.pose
        rot_world_base = quat_xyzw_to_rotmat(pose.orientation)
        trans_world_base = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ])
        return trans_world_base + rot_world_base @ point_base

    def project_bbox_to_ground_plane(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, float] | None:
        """Project bbox footpoint onto the configured ground plane."""
        if self.camera_info is None or self.latest_uav_pose is None:
            return None

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        pose = self.latest_uav_pose.pose
        rot_world_base = quat_xyzw_to_rotmat(pose.orientation)
        trans_world_base = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=float)
        camera_origin_world = trans_world_base + rot_world_base @ self.camera_translation_base
        camera_rotation_world = rot_world_base @ self.camera_rotation_base

        return project_bbox_footpoint_to_horizontal_plane(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=image_width,
            image_height=image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_origin_world=camera_origin_world,
            camera_rotation_world=camera_rotation_world,
            plane_z=self.ground_plane_z,
            min_distance_m=self.ground_projection_min_range_m,
            max_distance_m=self.ground_projection_max_range_m,
            bbox_ground_y_fraction=self.bbox_ground_y_fraction,
        )

    def make_detection_3d(
        self,
        stamp,
        class_name: str,
        confidence: float,
        point_world: np.ndarray,
        range_m: float,
        localization_source: str,
    ) -> Detection3D:
        detection = Detection3D()
        detection.header.stamp = stamp
        detection.header.frame_id = self.world_frame
        detection.id = localization_source

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_name
        hypothesis.hypothesis.score = confidence
        hypothesis.pose.pose.position.x = float(point_world[0])
        hypothesis.pose.pose.position.y = float(point_world[1])
        hypothesis.pose.pose.position.z = float(point_world[2])
        detection.results.append(hypothesis)

        detection.bbox.center.position.x = float(point_world[0])
        detection.bbox.center.position.y = float(point_world[1])
        detection.bbox.center.position.z = float(point_world[2])
        detection.bbox.center.orientation.w = 1.0
        detection.bbox.size.x = max(0.2, range_m * 0.05)
        detection.bbox.size.y = max(0.2, range_m * 0.05)
        detection.bbox.size.z = max(0.4, range_m * 0.08)
        return detection

    def make_markers(
        self,
        stamp,
        index: int,
        class_name: str,
        confidence: float,
        range_m: float,
        point_world: np.ndarray,
        localization_source: str,
    ) -> list[Marker]:
        sphere = Marker()
        sphere.header.stamp = stamp
        sphere.header.frame_id = self.world_frame
        sphere.ns = "detections_3d"
        sphere.id = index * 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(point_world[0])
        sphere.pose.position.y = float(point_world[1])
        sphere.pose.position.z = float(point_world[2])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.35
        sphere.scale.y = 0.35
        sphere.scale.z = 0.35
        if localization_source == "depth":
            sphere.color.r = 0.05
            sphere.color.g = 0.9
            sphere.color.b = 0.2
        else:
            sphere.color.r = 0.1
            sphere.color.g = 0.45
            sphere.color.b = 1.0
        sphere.color.a = 0.95

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = self.world_frame
        text.ns = "detections_3d_text"
        text.id = index * 2 + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(point_world[0])
        text.pose.position.y = float(point_world[1])
        text.pose.position.z = float(point_world[2] + 0.5)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.3
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 0.95
        text.text = f"{class_name} {confidence:.2f} {range_m:.1f}m {localization_source}"

        return [sphere, text]

    @staticmethod
    def draw_detection(image: np.ndarray, x1: float, y1: float, x2: float, y2: float, label: str) -> None:
        x1i = int(round(x1))
        y1i = int(round(y1))
        x2i = int(round(x2))
        y2i = int(round(y2))
        cv2.rectangle(image, (x1i, y1i), (x2i, y2i), (255, 80, 0), 2)

        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_top = max(0, y1i - label_h - baseline - 6)
        label_bottom = label_top + label_h + baseline + 6
        label_right = min(image.shape[1] - 1, x1i + label_w + 8)
        cv2.rectangle(image, (x1i, label_top), (label_right, label_bottom), (255, 80, 0), thickness=-1)
        cv2.putText(
            image,
            label,
            (x1i + 4, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
