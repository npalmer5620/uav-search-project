#!/usr/bin/env python3
"""
detection_node.py

ROS 2 node that subscribes to /camera/image_raw, runs YOLOv11 inference,
and publishes results as:
  - /detections          (uav_detection_msgs/msg/Detection2DArray — custom)
  - /camera/image_annotated  (sensor_msgs/msg/Image — for visualisation)

Parameters (set via ROS 2 params or launch file):
  model_path   : path to .pt weights file  (default: yolo11n.pt — auto-downloads)
  confidence   : detection threshold        (default: 0.4)
  device       : 'cpu' or 'cuda'           (default: 'cpu')
  image_topic  : input topic               (default: /camera/image_raw)
  publish_viz  : publish annotated image   (default: True)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

import cv2
import numpy as np
## from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed — run: pip install ultralytics --break-system-packages")


# COCO class names (indices 0-79)
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

# Classes we care about for search-and-rescue context
TARGET_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


class DetectionNode(Node):

    def __init__(self):
        super().__init__("detection_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("model_path",  "yolo11n.pt")
        self.declare_parameter("confidence",  0.4)
        self.declare_parameter("device",      "cpu")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("publish_viz", True)

        model_path  = self.get_parameter("model_path").value
        self.conf   = self.get_parameter("confidence").value
        device      = self.get_parameter("device").value
        image_topic = self.get_parameter("image_topic").value
        self.pub_viz = self.get_parameter("publish_viz").value

        # ── Load YOLO model ──────────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.get_logger().info("YOLO model loaded")

        ## self.bridge = CvBridge()

        # ── QoS ─────────────────────────────────────────────────────────────
        # Best-effort matches Gazebo bridge output
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self.det_pub = self.create_publisher(
            Detection2DArray,
            "/detections",
            10,
        )

        if self.pub_viz:
            self.viz_pub = self.create_publisher(
                Image,
                "/camera/image_annotated",
                10,
            )

        self.get_logger().info(
            f"DetectionNode ready — subscribing to {image_topic}"
        )

    # ── Callback ─────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        try:
            dtype = np.uint8
            frame = np.frombuffer(msg.data, dtype=dtype).reshape(
                msg.height, msg.width, -1
            )
            if msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgra8':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # bgr8 needs no conversion
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # ── Run inference ────────────────────────────────────────────────────
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        # ── Build Detection2DArray ───────────────────────────────────────────
        det_array = Detection2DArray()
        det_array.header = Header()
        det_array.header.stamp = msg.header.stamp
        det_array.header.frame_id = msg.header.frame_id

        h, w = frame.shape[:2]

        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else "unknown"

            # Filter to target classes only
            if class_name not in TARGET_CLASSES:
                continue

            # Bounding box in pixel coords
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            # Normalised offsets from image centre (-1..+1)
            cx_norm = (cx - w / 2.0) / (w / 2.0)
            cy_norm = (cy - h / 2.0) / (h / 2.0)

            det = Detection2D()
            det.header = det_array.header

            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = bw
            det.bbox.size_y = bh

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(class_id)
            hyp.hypothesis.score    = confidence
            det.results.append(hyp)

            det_array.detections.append(det)

            self.get_logger().debug(
                f"Detected {class_name} ({confidence:.2f}) "
                f"cx_norm={cx_norm:+.2f} cy_norm={cy_norm:+.2f} "
                f"bbox_area={bw*bh:.0f}px²"
            )

        self.det_pub.publish(det_array)

        n = len(det_array.detections)
        if n > 0:
            self.get_logger().info(f"{n} target(s) detected")

        # ── Annotated image ──────────────────────────────────────────────────
        if self.pub_viz:
            annotated = results.plot()  # returns BGR numpy array
            viz_msg = Image()
            viz_msg.header = det_array.header
            viz_msg.height = annotated.shape[0]
            viz_msg.width  = annotated.shape[1]
            viz_msg.encoding = "bgr8"
            viz_msg.step = annotated.shape[1] * 3
            viz_msg.data = annotated.tobytes()
            self.viz_pub.publish(viz_msg)


def main(args=None):
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
