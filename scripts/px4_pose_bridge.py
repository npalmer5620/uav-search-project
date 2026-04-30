#!/usr/bin/env python3
"""Publish Foxglove-friendly pose/TF output from PX4 vehicle odometry."""

import math
from typing import Iterable

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, TransformStamped
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


NED_TO_ENU = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])

FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def is_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def quat_wxyz_to_rotmat(q_wxyz: Iterable[float]) -> np.ndarray:
    w, x, y, z = q_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero-length quaternion")

    w /= n
    x /= n
    y /= n
    z /= n

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rotmat_to_quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rot))

    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


class Px4PoseBridge(Node):
    """Convert PX4 VehicleOdometry to PoseStamped, TF, and Marker outputs."""

    def __init__(self) -> None:
        super().__init__("px4_pose_bridge")

        self.declare_parameter("odom_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("pose_topic", "/uav/pose")
        self.declare_parameter("marker_topic", "/uav/drone_marker")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        odom_topic = self.get_parameter("odom_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        pub_qos = QoSProfile(depth=5)

        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, pub_qos)
        self.marker_pub = self.create_publisher(Marker, marker_topic, pub_qos)
        self.tf_broadcaster = TransformBroadcaster(self, qos=pub_qos)
        self.create_subscription(VehicleOdometry, odom_topic, self.odom_callback, odom_qos)

        self.last_pose_frame = None
        self.get_logger().info(
            f"Bridging {odom_topic} -> {pose_topic}, /tf, {marker_topic}"
        )

    def odom_callback(self, msg: VehicleOdometry) -> None:
        if not is_finite(msg.position) or not is_finite(msg.q):
            return

        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            if self.last_pose_frame != msg.pose_frame:
                self.get_logger().warning(
                    f"Unsupported pose_frame={msg.pose_frame}; expected NED. Skipping pose output."
                )
                self.last_pose_frame = msg.pose_frame
            return

        self.last_pose_frame = msg.pose_frame

        position_enu = NED_TO_ENU @ np.array(msg.position, dtype=float)
        rotation_ned_frd = quat_wxyz_to_rotmat(msg.q)
        rotation_enu_flu = NED_TO_ENU @ rotation_ned_frd @ FLU_TO_FRD
        quat_xyzw = rotmat_to_quat_xyzw(rotation_enu_flu)

        stamp = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.world_frame
        pose_msg.pose.position.x = float(position_enu[0])
        pose_msg.pose.position.y = float(position_enu[1])
        pose_msg.pose.position.z = float(position_enu[2])
        pose_msg.pose.orientation.x = quat_xyzw[0]
        pose_msg.pose.orientation.y = quat_xyzw[1]
        pose_msg.pose.orientation.z = quat_xyzw[2]
        pose_msg.pose.orientation.w = quat_xyzw[3]
        self.pose_pub.publish(pose_msg)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = pose_msg.pose.position.x
        transform.transform.translation.y = pose_msg.pose.position.y
        transform.transform.translation.z = pose_msg.pose.position.z
        transform.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "uav"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = 0.7
        marker.scale.y = 0.14
        marker.scale.z = 0.14
        marker.color.a = 0.95
        marker.color.r = 0.1
        marker.color.g = 0.75
        marker.color.b = 0.2
        self.marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Px4PoseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
