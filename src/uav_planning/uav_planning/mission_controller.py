#!/usr/bin/env python3
"""Detection-reactive mission controller.

Runs a grid (boustrophedon) search pattern.  When a 3-D detection arrives
above the confidence threshold, the drone pauses the grid, spirals around the
target to refine position and filter false positives, then resumes the search.

This node is the **sole publisher** to PX4 trajectory topics — the grid
generator is imported as a library, not a separate ROS node.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import os
from enum import Enum, auto
import socket
from statistics import fmean, median
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
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
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from uav_planning.grid_generator import GridGenerator
from uav_planning.spiral_generator import SpiralGenerator


@dataclass
class DetectionObservation:
    """Single deduplicated 3-D detection sample."""

    class_name: str
    confidence: float
    ned: tuple[float, float, float]
    enu: tuple[float, float, float]


@dataclass
class TrackedTarget:
    """Rolling multi-frame target estimate used for planner promotion."""

    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    hits: int = 0
    positions_ned: deque[tuple[float, float, float]] = field(default_factory=deque)
    positions_enu: deque[tuple[float, float, float]] = field(default_factory=deque)
    confidences: deque[float] = field(default_factory=deque)

    def add_observation(self, obs: DetectionObservation, now_s: float) -> None:
        self.last_seen = now_s
        self.hits += 1
        self.positions_ned.append(obs.ned)
        self.positions_enu.append(obs.enu)
        self.confidences.append(obs.confidence)

    def age(self, now_s: float) -> float:
        return now_s - self.first_seen

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(fmean(self.confidences))

    @property
    def filtered_ned(self) -> tuple[float, float, float]:
        return tuple(
            float(median(values)) for values in zip(*self.positions_ned, strict=False)
        )

    @property
    def filtered_enu(self) -> tuple[float, float, float]:
        return tuple(
            float(median(values)) for values in zip(*self.positions_enu, strict=False)
        )


class Phase(Enum):
    PREFLIGHT = auto()
    ARMING = auto()
    TAKEOFF = auto()
    HANDOFF = auto()
    SEARCH = auto()
    INVESTIGATE = auto()
    RTH = auto()
    LAND = auto()
    DONE = auto()


class MissionController(Node):

    @staticmethod
    def _default_shell_takeoff_enabled() -> bool:
        env_value = os.getenv("USE_PX4_SHELL_TAKEOFF")
        if env_value is not None:
            return env_value.strip().lower() in {"1", "true", "yes", "on"}
        return os.path.exists("/.dockerenv")

    @staticmethod
    def _default_px4_command_host() -> str:
        env_value = os.getenv("PX4_COMMAND_HOST")
        if env_value:
            return env_value
        return "sim" if os.path.exists("/.dockerenv") else "127.0.0.1"

    def __init__(self) -> None:
        super().__init__("mission_controller")

        # -- Parameters ----------------------------------------------------------
        # Grid search
        self.declare_parameter("grid.width", 40.0)
        self.declare_parameter("grid.height", 40.0)
        self.declare_parameter("grid.spacing", 5.0)
        self.declare_parameter("grid.speed", 2.0)
        self.declare_parameter("grid.altitude", -10.0)
        self.declare_parameter("grid.origin_x", 0.0)
        self.declare_parameter("grid.origin_y", 0.0)

        # Investigation behaviour
        self.declare_parameter("detection_confidence_threshold", 0.6)
        self.declare_parameter("investigate_duration", 10.0)
        self.declare_parameter("investigate_radius", 3.0)
        self.declare_parameter("investigate_spiral_spacing", 1.0)
        self.declare_parameter("investigate_spiral_speed", 0.5)
        self.declare_parameter("cooldown_distance", 3.0)
        self.declare_parameter("max_investigations", 0)  # 0 = unlimited
        self.declare_parameter("tracking.match_distance", 2.5)
        self.declare_parameter("tracking.duplicate_distance", 1.5)
        self.declare_parameter("tracking.track_timeout", 1.0)
        self.declare_parameter("tracking.history_size", 5)
        self.declare_parameter("tracking.min_hits", 3)
        self.declare_parameter("tracking.min_age", 0.5)
        self.declare_parameter("tracking.min_confidence", 0.65)

        # Takeoff stability
        self.declare_parameter("takeoff_altitude_tolerance", 0.5)  # metres
        self.declare_parameter("takeoff_velocity_tolerance", 0.3)  # m/s
        self.declare_parameter("takeoff_stable_ticks", 20)  # consecutive stable ticks required
        self.declare_parameter("takeoff_handoff_altitude", -2.5)  # metres in NED
        self.declare_parameter(
            "takeoff_timeout_proceed_altitude_tolerance", 1.0
        )  # metres
        self.declare_parameter("offboard_handoff_ticks", 10)
        self.declare_parameter("preflight_settle_ticks", 0)
        self.declare_parameter("shell_takeoff_arm_delay", 5.0)
        self.declare_parameter("takeoff_timeout_seconds", 60.0)
        self.declare_parameter("handoff_timeout_seconds", 60.0)
        self.declare_parameter(
            "use_px4_shell_takeoff",
            self._default_shell_takeoff_enabled(),
        )
        self.declare_parameter(
            "px4_command_host",
            self._default_px4_command_host(),
        )
        self.declare_parameter("px4_command_port", 14600)
        self.declare_parameter("px4_command_timeout", 1.0)

        # Timing
        self.declare_parameter("preflight_ticks", 20)
        self.declare_parameter("takeoff_ticks", 150)
        self.declare_parameter("timer_period", 0.1)

        # Read parameters
        grid_width = float(self.get_parameter("grid.width").value)
        grid_height = float(self.get_parameter("grid.height").value)
        grid_spacing = float(self.get_parameter("grid.spacing").value)
        grid_speed = float(self.get_parameter("grid.speed").value)
        grid_altitude = float(self.get_parameter("grid.altitude").value)
        grid_origin_x = float(self.get_parameter("grid.origin_x").value)
        grid_origin_y = float(self.get_parameter("grid.origin_y").value)

        self.search_altitude = grid_altitude

        self.confidence_threshold = float(
            self.get_parameter("detection_confidence_threshold").value
        )
        self.investigate_duration = float(
            self.get_parameter("investigate_duration").value
        )
        self.investigate_radius = float(
            self.get_parameter("investigate_radius").value
        )
        self.investigate_spiral_spacing = float(
            self.get_parameter("investigate_spiral_spacing").value
        )
        self.investigate_spiral_speed = float(
            self.get_parameter("investigate_spiral_speed").value
        )
        self.cooldown_distance = float(
            self.get_parameter("cooldown_distance").value
        )
        self.max_investigations = int(
            self.get_parameter("max_investigations").value
        )
        self.tracking_match_distance = float(
            self.get_parameter("tracking.match_distance").value
        )
        self.tracking_duplicate_distance = float(
            self.get_parameter("tracking.duplicate_distance").value
        )
        self.tracking_timeout = float(
            self.get_parameter("tracking.track_timeout").value
        )
        self.tracking_history_size = max(
            1, int(self.get_parameter("tracking.history_size").value)
        )
        self.tracking_min_hits = max(
            1, int(self.get_parameter("tracking.min_hits").value)
        )
        self.tracking_min_age = max(
            0.0, float(self.get_parameter("tracking.min_age").value)
        )
        self.tracking_min_confidence = float(
            self.get_parameter("tracking.min_confidence").value
        )
        self.takeoff_alt_tol = float(
            self.get_parameter("takeoff_altitude_tolerance").value
        )
        self.takeoff_vel_tol = float(
            self.get_parameter("takeoff_velocity_tolerance").value
        )
        self.takeoff_stable_required = int(
            self.get_parameter("takeoff_stable_ticks").value
        )
        self.takeoff_handoff_altitude = float(
            self.get_parameter("takeoff_handoff_altitude").value
        )
        self.takeoff_timeout_proceed_alt_tol = float(
            self.get_parameter("takeoff_timeout_proceed_altitude_tolerance").value
        )
        self.offboard_handoff_ticks = int(
            self.get_parameter("offboard_handoff_ticks").value
        )
        self.preflight_settle_ticks = int(
            self.get_parameter("preflight_settle_ticks").value
        )
        self.shell_takeoff_arm_delay_s = max(
            0.0,
            float(self.get_parameter("shell_takeoff_arm_delay").value),
        )
        self.takeoff_timeout_s = max(
            0.1,
            float(self.get_parameter("takeoff_timeout_seconds").value),
        )
        self.handoff_timeout_s = max(
            0.1,
            float(self.get_parameter("handoff_timeout_seconds").value),
        )
        self.use_px4_shell_takeoff = bool(
            self.get_parameter("use_px4_shell_takeoff").value
        )
        self.px4_command_host = str(
            self.get_parameter("px4_command_host").value
        ).strip()
        self.px4_command_port = int(
            self.get_parameter("px4_command_port").value
        )
        self.px4_command_timeout = float(
            self.get_parameter("px4_command_timeout").value
        )
        self.preflight_ticks = int(self.get_parameter("preflight_ticks").value)
        self.takeoff_ticks = int(self.get_parameter("takeoff_ticks").value)
        self.dt = float(self.get_parameter("timer_period").value)
        if self.dt > 0.0:
            self.shell_takeoff_arm_delay_ticks = max(
                0,
                int(math.ceil(self.shell_takeoff_arm_delay_s / self.dt)),
            )
        else:
            self.shell_takeoff_arm_delay_ticks = 0

        # -- Grid generator ------------------------------------------------------
        self.grid = GridGenerator(
            width=grid_width,
            height=grid_height,
            spacing=grid_spacing,
            speed=grid_speed,
            altitude=grid_altitude,
            origin=(grid_origin_x, grid_origin_y),
        )

        # -- PX4 QoS -------------------------------------------------------------
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
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -- PX4 publishers ------------------------------------------------------
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_pub
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_cmd
        )

        # -- PX4 subscribers -----------------------------------------------------
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

        # -- Detection subscriber ------------------------------------------------
        self.create_subscription(
            Detection3DArray, "/detections_3d", self._detection_cb, 10
        )

        # -- Observability publishers --------------------------------------------
        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.target_pub = self.create_publisher(PoseStamped, "~/investigate_target", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.event_pub = self.create_publisher(String, "~/events", 10)

        # -- State ---------------------------------------------------------------
        self._phase = Phase.PREFLIGHT
        self.local_pos = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.tick = 0
        self.preflight_ready_tick: int | None = None
        self.arming_tick: int = 0
        self.arm_ready_tick: int | None = None
        self.takeoff_start_tick: int = 0
        self.takeoff_start_time: float = 0.0
        self.handoff_start_tick: int = 0
        self.handoff_start_time: float = 0.0
        self.takeoff_stable_count = 0
        self.cruise_altitude: float | None = None
        self.handoff_hold_xy: tuple[float, float] | None = None
        self.shell_takeoff_requested = False
        self.arm_retry_ticks = 10
        self.takeoff_retry_ticks = 20
        self.offboard_retry_ticks = 10
        self.command_burst_count = 10
        self.command_burst_delay_s = 0.05

        # Investigation state
        self.investigate_target: tuple[float, float, float] | None = None  # NED
        self.investigate_target_enu: tuple[float, float, float] | None = None
        self.investigate_class: str = ""
        self.investigate_confidence: float = 0.0
        self.investigate_start_time = None
        self.investigate_spiral: SpiralGenerator | None = None
        self.investigate_approaching = False
        # Each entry: (ned_x, ned_y, enu_x, enu_y, enu_z, class_name, confidence)
        self.investigated_locations: list[tuple[float, float, float, float, float, str, float]] = []
        self.investigation_count = 0
        self.tracked_targets: list[TrackedTarget] = []
        self.next_track_id = 1

        self.timer = self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f"MissionController ready — grid {grid_width}x{grid_height}m, "
            f"spacing={grid_spacing}m, speed={grid_speed}m/s, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"investigate_duration={self.investigate_duration}s, "
            f"tracking_min_hits={self.tracking_min_hits}, "
            f"tracking_min_confidence={self.tracking_min_confidence}, "
            f"takeoff_mode={'px4_shell' if self.use_px4_shell_takeoff else 'vehicle_command'}, "
            f"shell_takeoff_arm_delay={self.shell_takeoff_arm_delay_s:.1f}s, "
            f"takeoff_timeout={self.takeoff_timeout_s:.1f}s, "
            f"handoff_timeout={self.handoff_timeout_s:.1f}s, "
            "timing_source=ros_clock"
        )

    # -- Phase property (logs transitions) ---------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @phase.setter
    def phase(self, new: Phase) -> None:
        if new != self._phase:
            self.get_logger().info(f"Phase: {self._phase.name} -> {new.name}")
        self._phase = new

    # -- Callbacks ---------------------------------------------------------------

    def _local_pos_cb(self, msg: VehicleLocalPosition) -> None:
        self.local_pos = msg

    def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _detection_cb(self, msg: Detection3DArray) -> None:
        """Update tracked targets and promote only stable ones for investigation."""
        if self.phase != Phase.SEARCH:
            return

        now_s = self._now_seconds()
        observations = self._cluster_observations(self._extract_observations(msg))
        self._update_tracked_targets(observations, now_s)

        promotable = self._select_promotable_target(now_s)
        if promotable is not None:
            self._begin_investigation(promotable)

    def _near_investigated(self, x: float, y: float) -> bool:
        for ned_x, ned_y, *_rest in self.investigated_locations:
            if math.hypot(x - ned_x, y - ned_y) < self.cooldown_distance:
                return True
        return False

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _elapsed_seconds(self, start_time_s: float) -> float:
        if start_time_s <= 0.0:
            return 0.0
        return max(0.0, self._now_seconds() - start_time_s)

    def _extract_observations(self, msg: Detection3DArray) -> list[DetectionObservation]:
        observations: list[DetectionObservation] = []
        for det in msg.detections:
            if not det.results:
                continue

            hyp = det.results[0]
            confidence = float(hyp.hypothesis.score)
            if confidence < self.confidence_threshold:
                continue

            # Detection is in ENU "map" frame (from px4_pose_bridge).
            # PX4 TrajectorySetpoint uses NED.
            # From px4_pose_bridge.py NED_TO_ENU = [[0,1,0],[1,0,0],[0,0,-1]]
            # Inverse: NED_x = ENU_y, NED_y = ENU_x, NED_z = -ENU_z
            enu_x = float(hyp.pose.pose.position.x)
            enu_y = float(hyp.pose.pose.position.y)
            enu_z = float(hyp.pose.pose.position.z)
            ned_x = enu_y
            ned_y = enu_x
            ned_z = -enu_z

            if self._near_investigated(ned_x, ned_y):
                continue

            observations.append(
                DetectionObservation(
                    class_name=hyp.hypothesis.class_id,
                    confidence=confidence,
                    ned=(ned_x, ned_y, ned_z),
                    enu=(enu_x, enu_y, enu_z),
                )
            )
        return observations

    def _cluster_observations(
        self, observations: list[DetectionObservation]
    ) -> list[DetectionObservation]:
        clustered: list[dict[str, object]] = []

        for obs in sorted(observations, key=lambda item: item.confidence, reverse=True):
            best_cluster = None
            best_distance = float("inf")

            for cluster in clustered:
                if cluster["class_name"] != obs.class_name:
                    continue
                cluster_ned = cluster["ned"]
                distance = math.dist(obs.ned, cluster_ned)
                if distance < self.tracking_duplicate_distance and distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster

            if best_cluster is None:
                clustered.append(
                    {
                        "class_name": obs.class_name,
                        "weight_sum": obs.confidence,
                        "confidence_sum": obs.confidence,
                        "count": 1,
                        "ned": obs.ned,
                        "enu": obs.enu,
                    }
                )
                continue

            count = int(best_cluster["count"]) + 1
            prev_weight = float(best_cluster["weight_sum"])
            weight_sum = prev_weight + obs.confidence
            best_cluster["weight_sum"] = weight_sum
            best_cluster["confidence_sum"] = float(best_cluster["confidence_sum"]) + obs.confidence
            best_cluster["count"] = count
            best_cluster["ned"] = tuple(
                (
                    prev_weight * prev_coord + obs.confidence * obs_coord
                ) / weight_sum
                for prev_coord, obs_coord in zip(best_cluster["ned"], obs.ned, strict=False)
            )
            best_cluster["enu"] = tuple(
                (
                    prev_weight * prev_coord + obs.confidence * obs_coord
                ) / weight_sum
                for prev_coord, obs_coord in zip(best_cluster["enu"], obs.enu, strict=False)
            )

        return [
            DetectionObservation(
                class_name=str(cluster["class_name"]),
                confidence=float(cluster["confidence_sum"]) / int(cluster["count"]),
                ned=tuple(float(value) for value in cluster["ned"]),
                enu=tuple(float(value) for value in cluster["enu"]),
            )
            for cluster in clustered
        ]

    def _update_tracked_targets(
        self, observations: list[DetectionObservation], now_s: float
    ) -> None:
        self.tracked_targets = [
            track
            for track in self.tracked_targets
            if now_s - track.last_seen <= self.tracking_timeout
        ]

        for obs in observations:
            track = self._find_matching_track(obs)
            if track is None:
                track = TrackedTarget(
                    track_id=self.next_track_id,
                    class_name=obs.class_name,
                    first_seen=now_s,
                    last_seen=now_s,
                    positions_ned=deque(maxlen=self.tracking_history_size),
                    positions_enu=deque(maxlen=self.tracking_history_size),
                    confidences=deque(maxlen=self.tracking_history_size),
                )
                self.next_track_id += 1
                self.tracked_targets.append(track)
            track.add_observation(obs, now_s)

    def _find_matching_track(
        self, obs: DetectionObservation
    ) -> TrackedTarget | None:
        best_track = None
        best_distance = float("inf")

        for track in self.tracked_targets:
            if track.class_name != obs.class_name:
                continue
            distance = math.dist(obs.ned, track.filtered_ned)
            if distance < self.tracking_match_distance and distance < best_distance:
                best_track = track
                best_distance = distance

        return best_track

    def _select_promotable_target(self, now_s: float) -> TrackedTarget | None:
        candidates = [
            track
            for track in self.tracked_targets
            if track.hits >= self.tracking_min_hits
            and track.age(now_s) >= self.tracking_min_age
            and track.mean_confidence >= self.tracking_min_confidence
            and not self._near_investigated(track.filtered_ned[0], track.filtered_ned[1])
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda track: (
                track.mean_confidence,
                track.hits,
                -math.hypot(track.filtered_ned[0] - self.local_pos.x, track.filtered_ned[1] - self.local_pos.y),
            ),
        )

    def _begin_investigation(self, track: TrackedTarget) -> None:
        ned_x, ned_y, ned_z = track.filtered_ned
        enu_x, enu_y, enu_z = track.filtered_enu

        self.investigate_target = (ned_x, ned_y, ned_z)
        self.investigate_target_enu = (enu_x, enu_y, enu_z)
        self.investigate_class = track.class_name
        self.investigate_confidence = track.mean_confidence
        self.investigate_start_time = self.get_clock().now()
        self.investigate_approaching = True
        # Create a small spiral for the investigation pass
        self.investigate_spiral = SpiralGenerator(
            max_radius=self.investigate_radius,
            spacing=self.investigate_spiral_spacing,
            angular_speed=self.investigate_spiral_speed,
            altitude=self.search_altitude,
        )
        self.phase = Phase.INVESTIGATE
        self.tracked_targets.clear()

        event = (
            f"Investigating stable {track.class_name} track #{track.track_id} "
            f"(hits={track.hits}, conf={track.mean_confidence:.2f}) at "
            f"({enu_x:.1f}, {enu_y:.1f}, {enu_z:.1f})"
        )
        self._publish_event(event)
        self.get_logger().info(event)

    # -- PX4 helpers -------------------------------------------------------------

    def _publish_offboard_mode(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.velocity = [float("nan")] * 3
        msg.acceleration = [float("nan")] * 3
        msg.jerk = [float("nan")] * 3
        msg.yaw = float(yaw)
        msg.yawspeed = float("nan")
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(msg)

    def _send_command(
        self,
        command: int,
        *,
        from_external: bool = True,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = from_external
        for _ in range(self.command_burst_count):
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.vehicle_command_pub.publish(msg)
            if self.command_burst_delay_s > 0.0:
                time.sleep(self.command_burst_delay_s)

    def _arm(self) -> None:
        self._send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            from_external=False,
            param1=1.0,
        )

    def _engage_offboard(self) -> None:
        self._send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

    def _land(self) -> None:
        self._send_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
            from_external=False,
        )

    def _is_armed(self) -> bool:
        return self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def _is_offboard(self) -> bool:
        return self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _has_takeoff_reference(self) -> bool:
        return (
            self.local_pos.xy_valid
            and self.local_pos.z_valid
            and self.local_pos.xy_global
            and self.local_pos.z_global
            and math.isfinite(self.local_pos.ref_lat)
            and math.isfinite(self.local_pos.ref_lon)
            and math.isfinite(self.local_pos.ref_alt)
        )

    def _native_takeoff_target_altitude(self) -> float:
        return max(self.search_altitude, self.takeoff_handoff_altitude)

    def _send_native_takeoff(self) -> tuple[float, float]:
        target_altitude = self._native_takeoff_target_altitude()
        target_amsl = float(self.local_pos.ref_alt - target_altitude)
        self._send_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            from_external=False,
            param5=float(self.local_pos.ref_lat),
            param6=float(self.local_pos.ref_lon),
            param7=target_amsl,
        )
        return target_altitude, target_amsl

    def _send_px4_shell_command(self, command: str) -> bool:
        command = command.strip()
        if not command:
            return False

        try:
            with socket.create_connection(
                (self.px4_command_host, self.px4_command_port),
                timeout=self.px4_command_timeout,
            ) as sock:
                sock.sendall((command + "\n").encode("utf-8"))
        except OSError as exc:
            self.get_logger().error(
                f"Failed to send PX4 shell command '{command}': {exc}"
            )
            return False

        return True

    def _current_yaw(self) -> float:
        if math.isfinite(self.local_pos.heading):
            return float(self.local_pos.heading)
        return 0.0

    def _px4_time_us(self) -> int:
        if getattr(self.local_pos, "timestamp", 0) > 0:
            return int(self.local_pos.timestamp)
        if getattr(self.vehicle_status, "timestamp", 0) > 0:
            return int(self.vehicle_status.timestamp)
        return 0

    # -- Visualization ----------------------------------------------------------

    def _publish_event(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.event_pub.publish(msg)

    def _ned_to_enu(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Convert NED coordinates to ENU for Foxglove markers."""
        return (y, x, -z)

    def _yaw_from_current_position(self, x: float, y: float) -> float:
        """Point the vehicle at a world-frame target from its current NED position."""
        return math.atan2(y - self.local_pos.y, x - self.local_pos.x)

    def _publish_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # -- Completed investigation markers (green sphere + text) ---------------
        for i, (_nx, _ny, ex, ey, ez, cls, conf) in enumerate(self.investigated_locations):
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = "map"
            sphere.ns = "investigated"
            sphere.id = i * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = ex
            sphere.pose.position.y = ey
            sphere.pose.position.z = ez
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.6
            sphere.scale.y = 0.6
            sphere.scale.z = 0.6
            sphere.color.r = 0.1
            sphere.color.g = 0.85
            sphere.color.b = 0.2
            sphere.color.a = 0.85

            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = "map"
            text.ns = "investigated_text"
            text.id = i * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = ex
            text.pose.position.y = ey
            text.pose.position.z = ez + 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.35
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.95
            text.text = f"#{i + 1} {cls} {conf:.0%}"

            ma.markers.extend([sphere, text])

        # -- Active investigation target (orange pulsing sphere + ring) ----------
        if self.phase == Phase.INVESTIGATE and self.investigate_target_enu is not None:
            ex, ey, ez = self.investigate_target_enu

            target_sphere = Marker()
            target_sphere.header.stamp = stamp
            target_sphere.header.frame_id = "map"
            target_sphere.ns = "active_target"
            target_sphere.id = 0
            target_sphere.type = Marker.SPHERE
            target_sphere.action = Marker.ADD
            target_sphere.pose.position.x = ex
            target_sphere.pose.position.y = ey
            target_sphere.pose.position.z = ez
            target_sphere.pose.orientation.w = 1.0
            target_sphere.scale.x = 0.8
            target_sphere.scale.y = 0.8
            target_sphere.scale.z = 0.8
            target_sphere.color.r = 1.0
            target_sphere.color.g = 0.6
            target_sphere.color.b = 0.0
            target_sphere.color.a = 0.9

            target_text = Marker()
            target_text.header.stamp = stamp
            target_text.header.frame_id = "map"
            target_text.ns = "active_target_text"
            target_text.id = 0
            target_text.type = Marker.TEXT_VIEW_FACING
            target_text.action = Marker.ADD
            target_text.pose.position.x = ex
            target_text.pose.position.y = ey
            target_text.pose.position.z = ez + 1.0
            target_text.pose.orientation.w = 1.0
            target_text.scale.z = 0.4
            target_text.color.r = 1.0
            target_text.color.g = 0.8
            target_text.color.b = 0.1
            target_text.color.a = 1.0
            elapsed = (
                self.get_clock().now() - self.investigate_start_time
            ).nanoseconds / 1e9
            remaining = max(0.0, self.investigate_duration - elapsed)
            target_text.text = (
                f"INVESTIGATING {self.investigate_class} "
                f"{self.investigate_confidence:.0%} [{remaining:.0f}s]"
            )

            # Orbit ring (cylinder at ground level)
            if self.investigate_radius > 0.0:
                ring = Marker()
                ring.header.stamp = stamp
                ring.header.frame_id = "map"
                ring.ns = "orbit_ring"
                ring.id = 0
                ring.type = Marker.CYLINDER
                ring.action = Marker.ADD
                ring.pose.position.x = ex
                ring.pose.position.y = ey
                ring.pose.position.z = ez
                ring.pose.orientation.w = 1.0
                ring.scale.x = self.investigate_radius * 2.0
                ring.scale.y = self.investigate_radius * 2.0
                ring.scale.z = 0.05
                ring.color.r = 1.0
                ring.color.g = 0.6
                ring.color.b = 0.0
                ring.color.a = 0.3
                ma.markers.append(ring)

            ma.markers.extend([target_sphere, target_text])
        else:
            # Delete active target markers when not investigating
            for ns in ("active_target", "active_target_text", "orbit_ring"):
                delete = Marker()
                delete.header.stamp = stamp
                delete.header.frame_id = "map"
                delete.ns = ns
                delete.id = 0
                delete.action = Marker.DELETE
                ma.markers.append(delete)

        # -- Drone state text (follows drone position) ---------------------------
        drone_state = Marker()
        drone_state.header.stamp = stamp
        drone_state.header.frame_id = "map"
        drone_state.ns = "drone_state"
        drone_state.id = 0
        drone_state.type = Marker.TEXT_VIEW_FACING
        drone_state.action = Marker.ADD
        # Convert drone NED position to ENU for display
        drone_enu = self._ned_to_enu(
            self.local_pos.x, self.local_pos.y, self.local_pos.z
        )
        drone_state.pose.position.x = drone_enu[0]
        drone_state.pose.position.y = drone_enu[1]
        drone_state.pose.position.z = drone_enu[2] + 1.5
        drone_state.pose.orientation.w = 1.0
        drone_state.scale.z = 0.4
        drone_state.color.r = 0.3
        drone_state.color.g = 0.9
        drone_state.color.b = 1.0
        drone_state.color.a = 0.95
        state_detail = self.phase.name
        if self.phase == Phase.SEARCH:
            state_detail += f" {self.grid.progress:.0%}"
        elif self.phase == Phase.INVESTIGATE:
            state_detail += f" ({self.investigate_class})"
        drone_state.text = state_detail
        ma.markers.append(drone_state)

        self.marker_pub.publish(ma)

    # -- Main loop ---------------------------------------------------------------

    def _timer_cb(self) -> None:
        if self.phase in (
            Phase.HANDOFF,
            Phase.SEARCH,
            Phase.INVESTIGATE,
            Phase.RTH,
        ):
            self._publish_offboard_mode()

        # Publish state for observability
        state_msg = String()
        state_msg.data = self.phase.name
        self.state_pub.publish(state_msg)

        if self.phase == Phase.PREFLIGHT:
            # Wait for PX4 to pass all preflight checks before starting the
            # takeoff sequence. Native vehicle-command takeoff additionally
            # requires a valid global reference.
            if self.tick < self.preflight_ticks:
                pass
            elif not self.vehicle_status.pre_flight_checks_pass:
                self.preflight_ready_tick = None
                if self.tick % 50 == 0:
                    self.get_logger().info("Waiting for PX4 preflight checks...")
            elif (
                not self.use_px4_shell_takeoff
                and not self._has_takeoff_reference()
            ):
                self.preflight_ready_tick = None
                if self.tick % 50 == 0:
                    self.get_logger().info(
                        "Waiting for PX4 global position reference for native takeoff..."
                    )
            else:
                if self.preflight_ready_tick is None:
                    self.preflight_ready_tick = self.tick

                settle_ticks = self.preflight_settle_ticks if self.use_px4_shell_takeoff else 0
                ticks_ready = self.tick - self.preflight_ready_tick
                if ticks_ready < settle_ticks:
                    if ticks_ready == 0 or ticks_ready % 50 == 0:
                        remaining_s = (settle_ticks - ticks_ready) * self.dt
                        self.get_logger().info(
                            "PX4 preflight checks passed — waiting "
                            f"{remaining_s:.0f}s for estimator settle before takeoff"
                        )
                else:
                    self.get_logger().info(
                        "PX4 preflight checks passed — starting "
                        f"{'PX4 shell' if self.use_px4_shell_takeoff else 'native'} takeoff sequence"
                    )
                    self.phase = Phase.ARMING
                    self.arming_tick = self.tick
                    self.arm_ready_tick = None
                    self.shell_takeoff_requested = False

        elif self.phase == Phase.ARMING:
            if self._is_armed():
                if self.arm_ready_tick is None:
                    self.arm_ready_tick = self.tick

                if self.use_px4_shell_takeoff:
                    ticks_armed = self.tick - self.arm_ready_tick
                    if ticks_armed < self.shell_takeoff_arm_delay_ticks:
                        if ticks_armed == 0 or ticks_armed % 10 == 0:
                            remaining_s = (
                                self.shell_takeoff_arm_delay_ticks - ticks_armed
                            ) * self.dt
                            self.get_logger().info(
                                "PX4 armed — waiting "
                                f"{remaining_s:.1f}s before shell takeoff"
                            )
                    elif not self.shell_takeoff_requested:
                        if not self._send_px4_shell_command("commander takeoff"):
                            event = (
                                "Failed to send PX4 shell takeoff command; aborting mission."
                            )
                            self._publish_event(event)
                            self.get_logger().error(event)
                            self.phase = Phase.LAND
                        else:
                            target_altitude = self._native_takeoff_target_altitude()
                            self.get_logger().info(
                                "PX4 armed — requesting PX4 shell takeoff to "
                                f"{target_altitude:.2f}m handoff altitude"
                            )
                            self.shell_takeoff_requested = True
                            self.phase = Phase.TAKEOFF
                            self.takeoff_start_tick = self.tick
                            self.takeoff_start_time = self._now_seconds()
                            self.takeoff_stable_count = 0
                else:
                    target_altitude, target_amsl = self._send_native_takeoff()
                    self.get_logger().info(
                        "PX4 armed — requesting native takeoff to "
                        f"{target_altitude:.2f}m (AMSL {target_amsl:.2f}m)"
                    )
                    self.phase = Phase.TAKEOFF
                    self.takeoff_start_tick = self.tick
                    self.takeoff_start_time = self._now_seconds()
                    self.takeoff_stable_count = 0
            else:
                self.arm_ready_tick = None
                if (
                    self.tick == self.arming_tick
                    or (self.tick - self.arming_tick) % self.arm_retry_ticks == 0
                ):
                    if self.use_px4_shell_takeoff:
                        if self._send_px4_shell_command("commander arm -f"):
                            if self.tick == self.arming_tick:
                                self.get_logger().info("PX4 shell arm command sent")
                    else:
                        self._arm()
                        if self.tick == self.arming_tick:
                            self.get_logger().info("Arm command sent")

                arming_timeout_ticks = self.takeoff_ticks
                if self.use_px4_shell_takeoff:
                    arming_timeout_ticks += self.shell_takeoff_arm_delay_ticks

                if self.tick >= self.arming_tick + arming_timeout_ticks:
                    event = (
                        "Failed to arm PX4 for "
                        f"{'shell' if self.use_px4_shell_takeoff else 'native'} takeoff; "
                        "aborting mission."
                    )
                    self._publish_event(event)
                    self.get_logger().error(event)
                    self.phase = Phase.LAND

        elif self.phase == Phase.TAKEOFF:
            target_altitude = self._native_takeoff_target_altitude()
            elapsed_takeoff_s = self._elapsed_seconds(self.takeoff_start_time)

            # Wait for PX4's native takeoff to get the vehicle cleanly airborne
            # before switching into offboard-controlled climb/search.
            alt_error = abs(self.local_pos.z - target_altitude)
            vz = abs(self.local_pos.vz)
            at_altitude = alt_error < self.takeoff_alt_tol
            velocity_settled = vz < self.takeoff_vel_tol

            if at_altitude and velocity_settled:
                self.takeoff_stable_count += 1
            else:
                self.takeoff_stable_count = 0

            # Periodic takeoff telemetry (every 2s)
            if self.tick % 20 == 0:
                ticks_in_takeoff = self.tick - self.takeoff_start_tick
                self.get_logger().info(
                    f"TAKEOFF {'shell' if self.use_px4_shell_takeoff else 'native'} "
                    f"z={self.local_pos.z:.2f}m target={target_altitude:.2f}m "
                    f"vz={self.local_pos.vz:.2f}m/s nav_state={self.vehicle_status.nav_state} "
                    f"stable={self.takeoff_stable_count}/{self.takeoff_stable_required} "
                    f"tick={ticks_in_takeoff}/{self.takeoff_ticks} "
                    f"elapsed={elapsed_takeoff_s:.1f}/{self.takeoff_timeout_s:.1f}s"
                )

            if self.takeoff_stable_count >= self.takeoff_stable_required:
                self.handoff_hold_xy = (
                    float(self.local_pos.x),
                    float(self.local_pos.y),
                )
                self.get_logger().info(
                    f"{'PX4 shell' if self.use_px4_shell_takeoff else 'Native'} takeoff complete — reached handoff altitude "
                    f"{target_altitude:.2f}m, preparing offboard takeover"
                )
                self.phase = Phase.HANDOFF
                self.handoff_start_tick = self.tick
                self.handoff_start_time = self._now_seconds()
                self.takeoff_stable_count = 0
            else:
                ticks_in_takeoff = self.tick - self.takeoff_start_tick
                if self.use_px4_shell_takeoff:
                    if (
                        ticks_in_takeoff > 0
                        and ticks_in_takeoff % self.takeoff_retry_ticks == 0
                        and self._is_armed()
                        and self.vehicle_status.nav_state
                        != VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF
                        and self.vehicle_status.takeoff_time == 0
                        and self.local_pos.z > target_altitude + self.takeoff_alt_tol
                    ):
                        if self._send_px4_shell_command("commander takeoff"):
                            self.get_logger().warn(
                                "Retrying PX4 shell takeoff command"
                            )
                elif (
                    ticks_in_takeoff > 0
                    and ticks_in_takeoff % self.takeoff_retry_ticks == 0
                    and self._is_armed()
                    and self.vehicle_status.takeoff_time == 0
                    and self._has_takeoff_reference()
                ):
                    _target_altitude, target_amsl = self._send_native_takeoff()
                    self.get_logger().warn(
                        "Retrying native takeoff request toward AMSL "
                        f"{target_amsl:.2f}m"
                    )

            if (
                self.phase == Phase.TAKEOFF
                and self.takeoff_start_time > 0.0
                and elapsed_takeoff_s >= self.takeoff_timeout_s
            ):
                event = (
                    f"{'PX4 shell' if self.use_px4_shell_takeoff else 'Native'} takeoff failed to reach handoff altitude; aborting mission "
                    f"at z={self.local_pos.z:.2f}m "
                    f"(target={target_altitude:.2f}m, alt_error={alt_error:.2f}m, "
                    f"vz={vz:.2f}m/s)"
                )
                self._publish_event(event)
                self.get_logger().error(event)
                self.phase = Phase.LAND

        elif self.phase == Phase.HANDOFF:
            if self.handoff_hold_xy is None:
                self.handoff_hold_xy = (
                    float(self.local_pos.x),
                    float(self.local_pos.y),
                )

            elapsed_handoff_s = self._elapsed_seconds(self.handoff_start_time)

            hold_x, hold_y = self.handoff_hold_xy
            self._publish_setpoint(
                hold_x,
                hold_y,
                self.search_altitude,
                self._current_yaw(),
            )

            ticks_in_handoff = self.tick - self.handoff_start_tick
            if (
                ticks_in_handoff >= self.offboard_handoff_ticks
                and not self._is_offboard()
                and ticks_in_handoff % self.offboard_retry_ticks == 0
            ):
                self._engage_offboard()
                if ticks_in_handoff == self.offboard_handoff_ticks:
                    self.get_logger().info(
                        "Takeoff complete — requesting offboard takeover"
                    )

            alt_error = abs(self.local_pos.z - self.search_altitude)
            vz = abs(self.local_pos.vz)
            at_altitude = alt_error < self.takeoff_alt_tol
            velocity_settled = vz < self.takeoff_vel_tol

            if self._is_offboard() and at_altitude and velocity_settled:
                self.takeoff_stable_count += 1
            else:
                self.takeoff_stable_count = 0

            if self.tick % 20 == 0:
                self.get_logger().info(
                    f"HANDOFF z={self.local_pos.z:.2f}m target={self.search_altitude:.2f}m "
                    f"vz={self.local_pos.vz:.2f}m/s nav_state={self.vehicle_status.nav_state} "
                    f"stable={self.takeoff_stable_count}/{self.takeoff_stable_required} "
                    f"tick={ticks_in_handoff}/{self.takeoff_ticks} "
                    f"elapsed={elapsed_handoff_s:.1f}/{self.handoff_timeout_s:.1f}s"
                )

            if self._is_offboard() and self.takeoff_stable_count >= self.takeoff_stable_required:
                self.cruise_altitude = self.search_altitude
                self.get_logger().info(
                    f"Offboard handoff complete — reached commanded altitude "
                    f"{self.search_altitude:.2f}m, stable for "
                    f"{self.takeoff_stable_required} ticks"
                )
                self.phase = Phase.SEARCH
            elif (
                self.handoff_start_time > 0.0
                and elapsed_handoff_s >= self.handoff_timeout_s
            ):
                if (
                    self._is_offboard()
                    and
                    alt_error <= self.takeoff_timeout_proceed_alt_tol
                    and velocity_settled
                ):
                    self.cruise_altitude = self.search_altitude
                    self.get_logger().warn(
                        "Offboard handoff timeout reached, but altitude is close enough to "
                        f"target to proceed safely: z={self.local_pos.z:.2f}m, "
                        f"target={self.search_altitude:.2f}m, alt_error={alt_error:.2f}m, "
                        f"vz={vz:.2f}m/s"
                    )
                    self.phase = Phase.SEARCH
                else:
                    event = (
                        "Offboard handoff failed to reach mission altitude; aborting mission "
                        f"at z={self.local_pos.z:.2f}m "
                        f"(target={self.search_altitude:.2f}m, "
                        f"alt_error={alt_error:.2f}m, vz={vz:.2f}m/s)"
                    )
                    self._publish_event(event)
                    self.get_logger().error(event)
                    self.phase = Phase.LAND

        elif self.phase == Phase.SEARCH:
            if self.cruise_altitude is None:
                self.get_logger().error(
                    "SEARCH entered without a valid cruise altitude; aborting mission."
                )
                self.phase = Phase.LAND
                self.tick += 1
                return

            # Only advance the grid when the drone is near the current target.
            # This prevents the cursor from running away while the drone
            # transits to the grid start or catches up after an investigation.
            current = self.grid.current_position
            if current is None:
                self.phase = Phase.RTH
                self.tick += 1
                return

            cx, cy, _cz, _cyaw = current
            dist_to_target = math.hypot(
                self.local_pos.x - cx, self.local_pos.y - cy
            )
            if dist_to_target < self.grid.spacing:
                waypoint = self.grid.step(self.dt)
                if waypoint is None:
                    self.phase = Phase.RTH
                    self.tick += 1
                    return
                x, y, _z, yaw = waypoint
            else:
                # Fly toward the current grid target without advancing
                x, y, _z = cx, cy, _cz
                yaw = self._yaw_from_current_position(x, y)

            # Periodic search telemetry (every 5s)
            if self.tick % 50 == 0:
                self.get_logger().info(
                    f"SEARCH grid={self.grid.progress:.0%} "
                    f"pos=({self.local_pos.x:.1f},{self.local_pos.y:.1f}) "
                    f"target=({x:.1f},{y:.1f}) dist={dist_to_target:.1f}m "
                    f"z={self.local_pos.z:.1f}m "
                    f"{'advancing' if dist_to_target < self.grid.spacing else 'transiting'}"
                )

            self._publish_setpoint(x, y, self.cruise_altitude, yaw)

        elif self.phase == Phase.INVESTIGATE:
            if self.cruise_altitude is None:
                self.get_logger().error(
                    "INVESTIGATE entered without a valid cruise altitude; aborting mission."
                )
                self.phase = Phase.LAND
                self.tick += 1
                return

            elapsed = (
                self.get_clock().now() - self.investigate_start_time
            ).nanoseconds / 1e9

            spiral_done = (
                self.investigate_spiral is not None
                and self.investigate_spiral.complete
            )
            timed_out = elapsed >= self.investigate_duration

            if spiral_done or timed_out:
                tx, ty, _tz = self.investigate_target
                ex, ey, ez = self.investigate_target_enu
                self.investigated_locations.append(
                    (tx, ty, ex, ey, ez, self.investigate_class, self.investigate_confidence)
                )
                self.investigation_count += 1
                reason = "spiral complete" if spiral_done else "timeout"
                event = (
                    f"Investigation #{self.investigation_count} complete "
                    f"({self.investigate_class}, {reason}). Resuming grid."
                )
                self._publish_event(event)
                self.get_logger().info(event)
                if (
                    self.max_investigations > 0
                    and self.investigation_count >= self.max_investigations
                ):
                    self.phase = Phase.RTH
                else:
                    self.phase = Phase.SEARCH
                self.tick += 1
                return

            tx, ty, tz = self.investigate_target
            distance_to_target = math.hypot(tx - self.local_pos.x, ty - self.local_pos.y)

            if self.investigate_approaching and distance_to_target > self.investigate_radius:
                # Approach the detection directly before starting the spiral
                yaw = self._yaw_from_current_position(tx, ty)
                self._publish_setpoint(tx, ty, self.cruise_altitude, yaw)
            else:
                # Spiral around the target to refine position / reject false positives
                self.investigate_approaching = False
                wp = self.investigate_spiral.step(self.dt)
                if wp is not None:
                    sx, sy, _sz, _syaw = wp
                    # Offset spiral output by target position
                    ox = tx + sx
                    oy = ty + sy
                    yaw = self._yaw_from_current_position(tx, ty)
                    self._publish_setpoint(ox, oy, self.cruise_altitude, yaw)

            # Publish target pose for visualization
            target_msg = PoseStamped()
            target_msg.header.stamp = self.get_clock().now().to_msg()
            target_msg.header.frame_id = "map"
            if self.investigate_target_enu is not None:
                ex, ey, ez = self.investigate_target_enu
            else:
                ex, ey, ez = self._ned_to_enu(tx, ty, tz)
            target_msg.pose.position.x = float(ex)
            target_msg.pose.position.y = float(ey)
            target_msg.pose.position.z = float(ez)
            self.target_pub.publish(target_msg)

        elif self.phase == Phase.RTH:
            if self.cruise_altitude is None:
                self.get_logger().error(
                    "RTH entered without a valid cruise altitude; landing in place."
                )
                self.phase = Phase.LAND
                self.tick += 1
                return

            self._publish_setpoint(0.0, 0.0, self.cruise_altitude, 0.0)
            dx = self.local_pos.x
            dy = self.local_pos.y
            if math.hypot(dx, dy) < 1.0:
                self.phase = Phase.LAND

        elif self.phase == Phase.LAND:
            self._land()
            self.phase = Phase.DONE

        elif self.phase == Phase.DONE:
            self._publish_setpoint(0.0, 0.0, 0.0, 0.0)

        self._publish_markers()
        self.tick += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionController()
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
