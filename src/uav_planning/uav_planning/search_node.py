#!/usr/bin/env python3
"""Standalone spiral search ROS 2 node.

Wraps SpiralGenerator in a ROS 2 node with the same phase state machine as
the original drone_sweep.py.  For the full detection-reactive system, use
MissionController instead — this node exists for standalone spiral testing.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from std_msgs.msg import String

from uav_planning.spiral_generator import SpiralGenerator


class SearchNode(Node):

    def __init__(self) -> None:
        super().__init__("search_node")

        # Declare configurable parameters (replaces hardcoded constants)
        self.declare_parameter("max_radius", 20.0)
        self.declare_parameter("spacing", 5.0)
        self.declare_parameter("angular_speed", 0.3)
        self.declare_parameter("altitude", -10.0)
        self.declare_parameter("preflight_ticks", 20)
        self.declare_parameter("takeoff_ticks", 150)
        self.declare_parameter("timer_period", 0.1)

        max_radius = float(self.get_parameter("max_radius").value)
        spacing = float(self.get_parameter("spacing").value)
        angular_speed = float(self.get_parameter("angular_speed").value)
        altitude = float(self.get_parameter("altitude").value)
        self.preflight_ticks = int(self.get_parameter("preflight_ticks").value)
        self.takeoff_ticks = int(self.get_parameter("takeoff_ticks").value)
        self.dt = float(self.get_parameter("timer_period").value)

        self.spiral = SpiralGenerator(max_radius, spacing, angular_speed, altitude)

        # PX4 QoS
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # PX4 publishers
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_pub
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_pub
        )

        # PX4 subscribers
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._local_pos_cb,
            qos_sub,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v3",
            self._vehicle_status_cb,
            qos_sub,
        )

        # State publisher
        self.state_pub = self.create_publisher(String, "~/state", 10)

        # State
        self.local_pos = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.tick = 0
        self.phase = "PREFLIGHT"
        self.cruise_altitude: float | None = None

        self.timer = self.create_timer(self.dt, self._timer_cb)

    # -- Callbacks ---------------------------------------------------------------

    def _local_pos_cb(self, msg: VehicleLocalPosition) -> None:
        self.local_pos = msg

    def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    # -- PX4 helpers -------------------------------------------------------------

    def _publish_offboard_mode(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def _send_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def _arm(self) -> None:
        self._send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def _engage_offboard(self) -> None:
        self._send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def _land(self) -> None:
        self._send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # -- Main loop ---------------------------------------------------------------

    def _timer_cb(self) -> None:
        self._publish_offboard_mode()

        state_msg = String()
        state_msg.data = self.phase
        self.state_pub.publish(state_msg)

        if self.phase == "PREFLIGHT":
            self._publish_setpoint(0.0, 0.0, self.spiral.altitude, 0.0)
            if self.tick == self.preflight_ticks:
                self._engage_offboard()
            if self.tick == self.preflight_ticks + 5:
                self._arm()
                self.phase = "TAKEOFF"

        elif self.phase == "TAKEOFF":
            self._publish_setpoint(0.0, 0.0, self.spiral.altitude, 0.0)
            if self.tick >= self.preflight_ticks + self.takeoff_ticks:
                self.cruise_altitude = self.local_pos.z
                self.get_logger().info(f"Locked altitude: {self.cruise_altitude:.2f}")
                self.phase = "SPIRAL"

        elif self.phase == "SPIRAL":
            waypoint = self.spiral.step(self.dt)
            if waypoint is None:
                self.phase = "RTH"
                self.tick += 1
                return
            x, y, _z, yaw = waypoint
            self._publish_setpoint(x, y, self.cruise_altitude, yaw)

        elif self.phase == "RTH":
            self._publish_setpoint(0.0, 0.0, self.cruise_altitude, 0.0)
            dx = self.local_pos.x
            dy = self.local_pos.y
            if math.sqrt(dx * dx + dy * dy) < 1.0:
                self.phase = "LAND"

        elif self.phase == "LAND":
            self._land()
            self.phase = "DONE"

        elif self.phase == "DONE":
            self._publish_setpoint(0.0, 0.0, 0.0, 0.0)

        self.tick += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
