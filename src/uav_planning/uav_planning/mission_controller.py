#!/usr/bin/env python3
"""Detection-reactive mission controller.

Runs a spiral search pattern by default.  When a 3-D detection arrives above
the confidence threshold, the drone pauses the spiral, orbits (or hovers over)
the target for a configurable duration, then resumes the search.

This node is the **sole publisher** to PX4 trajectory topics — the spiral
generator is imported as a library, not a separate ROS node.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from enum import Enum, auto
from statistics import fmean, median

import rclpy
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
    TAKEOFF = auto()
    SEARCH = auto()
    INVESTIGATE = auto()
    RTH = auto()
    LAND = auto()
    DONE = auto()


class MissionController(Node):

    def __init__(self) -> None:
        super().__init__("mission_controller")

        # -- Parameters ----------------------------------------------------------
        # Spiral
        self.declare_parameter("spiral.max_radius", 20.0)
        self.declare_parameter("spiral.spacing", 5.0)
        self.declare_parameter("spiral.angular_speed", 0.3)
        self.declare_parameter("spiral.altitude", -10.0)

        # Investigation behaviour
        self.declare_parameter("detection_confidence_threshold", 0.6)
        self.declare_parameter("investigate_duration", 10.0)
        self.declare_parameter("investigate_radius", 3.0)
        self.declare_parameter("investigate_orbit_speed", 0.5)
        self.declare_parameter("cooldown_distance", 3.0)
        self.declare_parameter("max_investigations", 0)  # 0 = unlimited
        self.declare_parameter("tracking.match_distance", 2.5)
        self.declare_parameter("tracking.duplicate_distance", 1.5)
        self.declare_parameter("tracking.track_timeout", 1.0)
        self.declare_parameter("tracking.history_size", 5)
        self.declare_parameter("tracking.min_hits", 3)
        self.declare_parameter("tracking.min_age", 0.5)
        self.declare_parameter("tracking.min_confidence", 0.65)

        # Timing
        self.declare_parameter("preflight_ticks", 20)
        self.declare_parameter("takeoff_ticks", 150)
        self.declare_parameter("timer_period", 0.1)

        # Read parameters
        max_radius = float(self.get_parameter("spiral.max_radius").value)
        spacing = float(self.get_parameter("spiral.spacing").value)
        angular_speed = float(self.get_parameter("spiral.angular_speed").value)
        altitude = float(self.get_parameter("spiral.altitude").value)

        self.confidence_threshold = float(
            self.get_parameter("detection_confidence_threshold").value
        )
        self.investigate_duration = float(
            self.get_parameter("investigate_duration").value
        )
        self.investigate_radius = float(
            self.get_parameter("investigate_radius").value
        )
        self.investigate_orbit_speed = float(
            self.get_parameter("investigate_orbit_speed").value
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
        self.preflight_ticks = int(self.get_parameter("preflight_ticks").value)
        self.takeoff_ticks = int(self.get_parameter("takeoff_ticks").value)
        self.dt = float(self.get_parameter("timer_period").value)

        # -- Spiral generator ----------------------------------------------------
        self.spiral = SpiralGenerator(max_radius, spacing, angular_speed, altitude)

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

        # -- PX4 publishers ------------------------------------------------------
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_pub
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_pub
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
        self.phase = Phase.PREFLIGHT
        self.local_pos = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.tick = 0
        self.cruise_altitude: float | None = None

        # Investigation state
        self.investigate_target: tuple[float, float, float] | None = None  # NED
        self.investigate_target_enu: tuple[float, float, float] | None = None
        self.investigate_class: str = ""
        self.investigate_confidence: float = 0.0
        self.investigate_start_time = None
        self.investigate_orbit_theta = 0.0
        self.investigate_orbit_active = False
        # Each entry: (ned_x, ned_y, enu_x, enu_y, enu_z, class_name, confidence)
        self.investigated_locations: list[tuple[float, float, float, float, float, str, float]] = []
        self.investigation_count = 0
        self.tracked_targets: list[TrackedTarget] = []
        self.next_track_id = 1

        self.timer = self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f"MissionController ready — spiral max_radius={max_radius}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"investigate_duration={self.investigate_duration}s, "
            f"tracking_min_hits={self.tracking_min_hits}, "
            f"tracking_min_confidence={self.tracking_min_confidence}"
        )

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
        self.investigate_orbit_theta = 0.0
        self.investigate_orbit_active = False
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
            state_detail += f" {self.spiral.progress:.0%}"
        elif self.phase == Phase.INVESTIGATE:
            state_detail += f" ({self.investigate_class})"
        drone_state.text = state_detail
        ma.markers.append(drone_state)

        self.marker_pub.publish(ma)

    # -- Main loop ---------------------------------------------------------------

    def _timer_cb(self) -> None:
        self._publish_offboard_mode()

        # Publish state for observability
        state_msg = String()
        state_msg.data = self.phase.name
        self.state_pub.publish(state_msg)

        if self.phase == Phase.PREFLIGHT:
            self._publish_setpoint(0.0, 0.0, self.spiral.altitude, 0.0)
            if self.tick == self.preflight_ticks:
                self._engage_offboard()
            if self.tick == self.preflight_ticks + 5:
                self._arm()
                self.phase = Phase.TAKEOFF

        elif self.phase == Phase.TAKEOFF:
            self._publish_setpoint(0.0, 0.0, self.spiral.altitude, 0.0)
            if self.tick >= self.preflight_ticks + self.takeoff_ticks:
                self.cruise_altitude = self.local_pos.z
                self.get_logger().info(f"Locked altitude: {self.cruise_altitude:.2f}")
                self.phase = Phase.SEARCH

        elif self.phase == Phase.SEARCH:
            waypoint = self.spiral.step(self.dt)
            if waypoint is None:
                self.phase = Phase.RTH
                self.tick += 1
                return
            x, y, _z, yaw = waypoint
            self._publish_setpoint(x, y, self.cruise_altitude, yaw)

        elif self.phase == Phase.INVESTIGATE:
            elapsed = (
                self.get_clock().now() - self.investigate_start_time
            ).nanoseconds / 1e9

            if elapsed >= self.investigate_duration:
                tx, ty, _tz = self.investigate_target
                ex, ey, ez = self.investigate_target_enu
                self.investigated_locations.append(
                    (tx, ty, ex, ey, ez, self.investigate_class, self.investigate_confidence)
                )
                self.investigation_count += 1
                event = (
                    f"Investigation #{self.investigation_count} complete "
                    f"({self.investigate_class}). Resuming spiral."
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

            if self.investigate_radius > 0.0 and distance_to_target > self.investigate_radius:
                # Approach the detection directly before starting any orbit.
                self.investigate_orbit_active = False
                yaw = self._yaw_from_current_position(tx, ty)
                self._publish_setpoint(tx, ty, self.cruise_altitude, yaw)
            elif self.investigate_radius > 0.0:
                # Once near the target, orbit from the current bearing instead of
                # snapping to an arbitrary point on the circle.
                if not self.investigate_orbit_active:
                    self.investigate_orbit_theta = math.atan2(
                        self.local_pos.y - ty, self.local_pos.x - tx
                    )
                    self.investigate_orbit_active = True

                self.investigate_orbit_theta += self.investigate_orbit_speed * self.dt
                ox = tx + self.investigate_radius * math.cos(self.investigate_orbit_theta)
                oy = ty + self.investigate_radius * math.sin(self.investigate_orbit_theta)
                yaw = self._yaw_from_current_position(tx, ty)
                self._publish_setpoint(ox, oy, self.cruise_altitude, yaw)
            else:
                # Hover directly over target
                yaw = self._yaw_from_current_position(tx, ty)
                self._publish_setpoint(tx, ty, self.cruise_altitude, yaw)

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
