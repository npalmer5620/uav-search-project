#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)

# ── Spiral config ──────────────────────────────────────────────────────────────
MAX_RADIUS = 20.0     # meters (stop condition)
SPACING    = 5.0      # distance between loops

# Spiral equation: r = b * theta
B = SPACING / (2 * math.pi)

# Motion tuning
ANGULAR_SPEED = 0.3   # rad/sec (controls speed)

ALTITUDE = -10.0

# ── Timing ─────────────────────────────────────────────────────────────────────
PREFLIGHT_TICKS = 20
TAKEOFF_TICKS   = 150


class ContinuousSpiral(Node):

    def __init__(self):
        super().__init__('continuous_spiral')

        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_pub)
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_pub)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_pub)

        # Subscribers
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.local_position_callback,
            qos_sub
        )
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v3',
            self.vehicle_status_callback,
            qos_sub
        )

        # State
        self.local_pos        = VehicleLocalPosition()
        self.vehicle_status   = VehicleStatus()
        self.tick             = 0
        self.phase            = 'PREFLIGHT'
        self.theta            = 0.0
        self.cruise_altitude  = None

        self.dt = 0.1  # timer period

        self.timer = self.create_timer(self.dt, self.timer_callback)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def local_position_callback(self, msg):
        self.local_pos = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    # ── PX4 publishers ─────────────────────────────────────────────────────────

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
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

    def arm(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def engage_offboard(self):
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

    def land(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def timer_callback(self):
        self.publish_offboard_control_mode()

        # PREFLIGHT
        if self.phase == 'PREFLIGHT':
            self.publish_trajectory_setpoint(0.0, 0.0, ALTITUDE, 0.0)

            if self.tick == PREFLIGHT_TICKS:
                self.engage_offboard()
            if self.tick == PREFLIGHT_TICKS + 5:
                self.arm()
                self.phase = 'TAKEOFF'

        # TAKEOFF
        elif self.phase == 'TAKEOFF':
            self.publish_trajectory_setpoint(0.0, 0.0, ALTITUDE, 0.0)

            if self.tick >= PREFLIGHT_TICKS + TAKEOFF_TICKS:
                self.cruise_altitude = self.local_pos.z
                self.get_logger().info(
                    f'Locked altitude: {self.cruise_altitude:.2f}')
                self.phase = 'SPIRAL'

        # CONTINUOUS SPIRAL
        elif self.phase == 'SPIRAL':

            # Increase angle over time
            self.theta += ANGULAR_SPEED * self.dt

            # Spiral radius
            r = B * self.theta

            # Stop condition
            if r > MAX_RADIUS:
                self.phase = 'RTH'
                return

            # Convert to Cartesian (NED: x=north, y=east)
            x = r * math.cos(self.theta)
            y = r * math.sin(self.theta)

            # Velocity direction for yaw
            dx = B * math.cos(self.theta) - r * math.sin(self.theta)
            dy = B * math.sin(self.theta) + r * math.cos(self.theta)
            yaw = math.atan2(dy, dx)

            self.publish_trajectory_setpoint(
                x,
                y,
                self.cruise_altitude,
                yaw
            )

        # RETURN
        elif self.phase == 'RTH':
            self.publish_trajectory_setpoint(
                0.0, 0.0, self.cruise_altitude, 0.0)

            dx = self.local_pos.x
            dy = self.local_pos.y
            if math.sqrt(dx*dx + dy*dy) < 1.0:
                self.phase = 'LAND'

        # LAND
        elif self.phase == 'LAND':
            self.land()
            self.phase = 'DONE'

        # DONE
        elif self.phase == 'DONE':
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0, 0.0)

        self.tick += 1


def main(args=None):
    rclpy.init(args=args)
    node = ContinuousSpiral()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
