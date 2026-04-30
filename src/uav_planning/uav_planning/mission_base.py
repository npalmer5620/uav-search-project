#!/usr/bin/env python3
"""Shared MAVSDK mission controller base for search implementations."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from statistics import fmean, median
from typing import Callable

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from uav_planning.mavsdk_backend import MavsdkBackend
from uav_planning.spiral_generator import SpiralGenerator


@dataclass
class LocalPosition:
    """Small NED state mirror populated from MAVSDK telemetry."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    xy_valid: bool = False
    z_valid: bool = False
    v_xy_valid: bool = False
    v_z_valid: bool = False
    xy_global: bool = False
    z_global: bool = False


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
    HANDOFF = auto()
    SEARCH = auto()
    INVESTIGATE = auto()
    RTH = auto()
    LAND = auto()
    DONE = auto()


class MissionControllerBase(Node):
    """Common MAVSDK mission flow with pluggable SEARCH behavior."""

    def __init__(
        self,
        node_name: str,
        *,
        extra_parameter_declarations: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(node_name)

        if extra_parameter_declarations is not None:
            extra_parameter_declarations()

        # -- Parameters ------------------------------------------------------
        self.declare_parameter("grid.width", 40.0)
        self.declare_parameter("grid.height", 40.0)
        self.declare_parameter("grid.spacing", 5.0)
        self.declare_parameter("grid.speed", 2.0)
        self.declare_parameter("grid.altitude", -10.0)
        self.declare_parameter("grid.origin_x", 0.0)
        self.declare_parameter("grid.origin_y", 0.0)

        self.declare_parameter("detection_confidence_threshold", 0.6)
        self.declare_parameter("investigate_duration", 10.0)
        self.declare_parameter("investigate_radius", 3.0)
        self.declare_parameter("investigate_spiral_spacing", 1.0)
        self.declare_parameter("investigate_spiral_speed", 0.5)
        self.declare_parameter("cooldown_distance", 3.0)
        self.declare_parameter("max_investigations", 0)
        self.declare_parameter("tracking.match_distance", 2.5)
        self.declare_parameter("tracking.duplicate_distance", 1.5)
        self.declare_parameter("tracking.track_timeout", 1.0)
        self.declare_parameter("tracking.history_size", 5)
        self.declare_parameter("tracking.min_hits", 3)
        self.declare_parameter("tracking.min_age", 0.5)
        self.declare_parameter("tracking.min_confidence", 0.65)

        self.declare_parameter("takeoff_altitude_tolerance", 0.5)
        self.declare_parameter("takeoff_velocity_tolerance", 0.3)
        self.declare_parameter("takeoff_stable_ticks", 20)
        self.declare_parameter("takeoff_timeout_proceed_altitude_tolerance", 1.0)
        self.declare_parameter("offboard_handoff_ticks", 10)
        self.declare_parameter("takeoff_timeout_seconds", 60.0)
        self.declare_parameter("handoff_timeout_seconds", 60.0)
        self.declare_parameter("mavsdk.system_address", "udpin://0.0.0.0:14540")

        self.declare_parameter("preflight_ticks", 20)
        self.declare_parameter("takeoff_ticks", 150)
        self.declare_parameter("timer_period", 0.1)

        self.search_width = float(self.get_parameter("grid.width").value)
        self.search_height = float(self.get_parameter("grid.height").value)
        self.search_spacing = float(self.get_parameter("grid.spacing").value)
        self.search_speed = float(self.get_parameter("grid.speed").value)
        self.search_altitude = float(self.get_parameter("grid.altitude").value)
        self.search_origin_x = float(self.get_parameter("grid.origin_x").value)
        self.search_origin_y = float(self.get_parameter("grid.origin_y").value)
        self.search_x_min = self.search_origin_x - self.search_height / 2.0
        self.search_x_max = self.search_origin_x + self.search_height / 2.0
        self.search_y_min = self.search_origin_y - self.search_width / 2.0
        self.search_y_max = self.search_origin_y + self.search_width / 2.0

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
        self.takeoff_timeout_proceed_alt_tol = float(
            self.get_parameter("takeoff_timeout_proceed_altitude_tolerance").value
        )
        self.offboard_handoff_ticks = int(
            self.get_parameter("offboard_handoff_ticks").value
        )
        self.takeoff_timeout_s = max(
            0.1,
            float(self.get_parameter("takeoff_timeout_seconds").value),
        )
        self.handoff_timeout_s = max(
            0.1,
            float(self.get_parameter("handoff_timeout_seconds").value),
        )
        self.mavsdk_system_address = str(
            self.get_parameter("mavsdk.system_address").value
        ).strip()
        self.mavsdk_backend = MavsdkBackend(
            system_address=self.mavsdk_system_address,
            logger=self.get_logger(),
        )
        self.mavsdk_backend.start()

        self.preflight_ticks = int(self.get_parameter("preflight_ticks").value)
        self.takeoff_ticks = int(self.get_parameter("takeoff_ticks").value)
        self.dt = float(self.get_parameter("timer_period").value)

        self._init_search_controller()

        self.create_subscription(
            Detection3DArray, "/detections_3d", self._detection_cb, 10
        )

        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.target_pub = self.create_publisher(PoseStamped, "~/investigate_target", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.event_pub = self.create_publisher(String, "~/events", 10)

        self._phase = Phase.PREFLIGHT
        self.local_pos = LocalPosition()
        self.tick = 0
        self.preflight_ready_tick: int | None = None
        self.arming_tick = 0
        self.arm_ready_tick: int | None = None
        self.takeoff_start_tick = 0
        self.takeoff_start_time = 0.0
        self.handoff_start_tick = 0
        self.handoff_start_time = 0.0
        self.takeoff_stable_count = 0
        self.cruise_altitude: float | None = None
        self.handoff_hold_xy: tuple[float, float] | None = None
        self.arm_retry_ticks = 10
        self.offboard_retry_ticks = 10
        self.search_elapsed_s = 0.0

        self.investigate_target: tuple[float, float, float] | None = None
        self.investigate_target_enu: tuple[float, float, float] | None = None
        self.investigate_class = ""
        self.investigate_confidence = 0.0
        self.investigate_start_time = None
        self.investigate_spiral: SpiralGenerator | None = None
        self.investigate_approaching = False
        self.investigated_locations: list[
            tuple[float, float, float, float, float, str, float]
        ] = []
        self.investigation_count = 0
        self.tracked_targets: list[TrackedTarget] = []
        self.next_track_id = 1

        self.timer = self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f"{self.__class__.__name__} ready — search {self.search_width}x{self.search_height}m, "
            f"spacing={self.search_spacing}m, speed={self.search_speed}m/s, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"investigate_duration={self.investigate_duration}s, "
            f"tracking_min_hits={self.tracking_min_hits}, "
            f"tracking_min_confidence={self.tracking_min_confidence}, "
            "flight_control=mavsdk, "
            f"mavsdk_system_address={self.mavsdk_system_address}, "
            f"takeoff_timeout={self.takeoff_timeout_s:.1f}s, "
            f"handoff_timeout={self.handoff_timeout_s:.1f}s, "
            "timing_source=ros_clock"
        )

    def destroy_node(self) -> bool:
        if self.mavsdk_backend is not None:
            self.mavsdk_backend.close()
        return super().destroy_node()

    # -- Search hooks ---------------------------------------------------------

    def _init_search_controller(self) -> None:
        raise NotImplementedError

    def _search_step(self) -> None:
        raise NotImplementedError

    def _search_state_detail(self) -> str | None:
        return None

    def _on_search_enter(self) -> None:
        """Hook for subclasses that need to react to SEARCH entry."""

    def _on_search_exit(self) -> None:
        """Hook for subclasses that need to react to SEARCH exit."""

    # -- Phase property -------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @phase.setter
    def phase(self, new: Phase) -> None:
        previous = self._phase
        if new != previous:
            self.get_logger().info(f"Phase: {previous.name} -> {new.name}")
            if previous == Phase.SEARCH:
                self._on_search_exit()
            self._phase = new
            if new == Phase.SEARCH:
                self._on_search_enter()
            return
        self._phase = new

    # -- Shared geometry helpers ---------------------------------------------

    def clip_to_search_area(self, x: float, y: float) -> tuple[float, float, float]:
        """Clamp a NED XY target into the configured search rectangle."""
        clipped_x = min(max(float(x), self.search_x_min), self.search_x_max)
        clipped_y = min(max(float(y), self.search_y_min), self.search_y_max)
        overflow = math.hypot(clipped_x - x, clipped_y - y)
        return clipped_x, clipped_y, overflow

    # -- Callbacks ------------------------------------------------------------

    def _sync_mavsdk_local_position(self) -> None:
        status = self.mavsdk_backend.status
        if not status.position_velocity_valid:
            return

        self.local_pos.x = status.north_m
        self.local_pos.y = status.east_m
        self.local_pos.z = status.down_m
        self.local_pos.vx = status.velocity_north_m_s
        self.local_pos.vy = status.velocity_east_m_s
        self.local_pos.vz = status.velocity_down_m_s
        self.local_pos.xy_valid = True
        self.local_pos.z_valid = True
        self.local_pos.v_xy_valid = True
        self.local_pos.v_z_valid = True
        self.local_pos.xy_global = status.global_position_ok
        self.local_pos.z_global = status.global_position_ok

    def _detection_cb(self, msg: Detection3DArray) -> None:
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
                if (
                    distance < self.tracking_duplicate_distance
                    and distance < best_distance
                ):
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
            best_cluster["confidence_sum"] = (
                float(best_cluster["confidence_sum"]) + obs.confidence
            )
            best_cluster["count"] = count
            best_cluster["ned"] = tuple(
                (
                    prev_weight * prev_coord + obs.confidence * obs_coord
                ) / weight_sum
                for prev_coord, obs_coord in zip(
                    best_cluster["ned"], obs.ned, strict=False
                )
            )
            best_cluster["enu"] = tuple(
                (
                    prev_weight * prev_coord + obs.confidence * obs_coord
                ) / weight_sum
                for prev_coord, obs_coord in zip(
                    best_cluster["enu"], obs.enu, strict=False
                )
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
                -math.hypot(
                    track.filtered_ned[0] - self.local_pos.x,
                    track.filtered_ned[1] - self.local_pos.y,
                ),
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

    # -- MAVSDK helpers -------------------------------------------------------

    def _publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        self.mavsdk_backend.set_position_ned(x, y, z, yaw)

    def _arm(self) -> None:
        self.mavsdk_backend.arm()

    def _engage_offboard(self) -> None:
        self.mavsdk_backend.start_offboard()

    def _land(self) -> None:
        self.mavsdk_backend.land()

    def _is_armed(self) -> bool:
        return self.mavsdk_backend.status.connected and self.mavsdk_backend.status.armed

    def _is_offboard(self) -> bool:
        return self.mavsdk_backend.status.connected and self.mavsdk_backend.status.offboard

    def _preflight_checks_passed(self) -> bool:
        status = self.mavsdk_backend.status
        if status.last_error:
            return False
        return (
            status.connected
            and status.health_all_ok
            and status.position_velocity_valid
        )

    def _current_yaw(self) -> float:
        return 0.0

    # -- Visualization --------------------------------------------------------

    def _publish_event(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.event_pub.publish(msg)

    def _ned_to_enu(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return (y, x, -z)

    def _yaw_from_current_position(self, x: float, y: float) -> float:
        return math.atan2(y - self.local_pos.y, x - self.local_pos.x)

    def _publish_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()

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
            for ns in ("active_target", "active_target_text", "orbit_ring"):
                delete = Marker()
                delete.header.stamp = stamp
                delete.header.frame_id = "map"
                delete.ns = ns
                delete.id = 0
                delete.action = Marker.DELETE
                ma.markers.append(delete)

        drone_state = Marker()
        drone_state.header.stamp = stamp
        drone_state.header.frame_id = "map"
        drone_state.ns = "drone_state"
        drone_state.id = 0
        drone_state.type = Marker.TEXT_VIEW_FACING
        drone_state.action = Marker.ADD
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
            detail = self._search_state_detail()
            if detail:
                state_detail += f" {detail}"
        elif self.phase == Phase.INVESTIGATE:
            state_detail += f" ({self.investigate_class})"
        drone_state.text = state_detail
        ma.markers.append(drone_state)

        self.marker_pub.publish(ma)

    # -- Main loop ------------------------------------------------------------

    def _timer_cb(self) -> None:
        self._sync_mavsdk_local_position()

        state_msg = String()
        state_msg.data = self.phase.name
        self.state_pub.publish(state_msg)

        if self.phase == Phase.PREFLIGHT:
            if self.tick < self.preflight_ticks:
                pass
            elif not self._preflight_checks_passed():
                self.preflight_ready_tick = None
                if self.tick % 50 == 0:
                    status = self.mavsdk_backend.status
                    if status.last_error:
                        detail = f" ({status.last_error})"
                    else:
                        detail = (
                            " "
                            f"(connected={status.connected}, "
                            f"armable={status.armable}, "
                            f"local={status.local_position_ok}, "
                            f"global={status.global_position_ok}, "
                            f"home={status.home_position_ok}, "
                            f"ned={status.position_velocity_valid})"
                        )
                    self.get_logger().info(
                        f"Waiting for PX4 preflight checks{detail}..."
                    )
            else:
                if self.preflight_ready_tick is None:
                    self.preflight_ready_tick = self.tick

                ticks_ready = self.tick - self.preflight_ready_tick
                if ticks_ready >= 0:
                    self.get_logger().info(
                        "PX4 preflight checks passed — starting MAVSDK offboard takeoff sequence"
                    )
                    self.phase = Phase.ARMING
                    self.arming_tick = self.tick
                    self.arm_ready_tick = None

        elif self.phase == Phase.ARMING:
            if self._is_armed():
                if self.arm_ready_tick is None:
                    self.arm_ready_tick = self.tick

                self.handoff_hold_xy = (
                    float(self.local_pos.x),
                    float(self.local_pos.y),
                )
                hold_x, hold_y = self.handoff_hold_xy
                self._publish_setpoint(
                    hold_x,
                    hold_y,
                    self.search_altitude,
                    self._current_yaw(),
                )
                self.get_logger().info(
                    "PX4 armed — starting MAVSDK offboard climb to "
                    f"{self.search_altitude:.2f}m"
                )
                self.phase = Phase.HANDOFF
                self.handoff_start_tick = self.tick - self.offboard_handoff_ticks
                self.handoff_start_time = self._now_seconds()
                self.takeoff_stable_count = 0
            else:
                self.arm_ready_tick = None
                if (
                    self.tick == self.arming_tick
                    or (self.tick - self.arming_tick) % self.arm_retry_ticks == 0
                ):
                    self._arm()
                    if self.tick == self.arming_tick:
                        self.get_logger().info("MAVSDK arm command sent")

                arming_timeout_ticks = self.takeoff_ticks

                if self.tick >= self.arming_tick + arming_timeout_ticks:
                    event = (
                        "Failed to arm PX4 for MAVSDK offboard takeoff; "
                        "aborting mission."
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
                    f"vz={self.local_pos.vz:.2f}m/s "
                    f"offboard={self._is_offboard()} "
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
                    and alt_error <= self.takeoff_timeout_proceed_alt_tol
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

            self.search_elapsed_s += self.dt
            self._search_step()

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
                self.investigate_spiral is not None and self.investigate_spiral.complete
            )
            timed_out = elapsed >= self.investigate_duration

            if spiral_done or timed_out:
                tx, ty, _tz = self.investigate_target
                ex, ey, ez = self.investigate_target_enu
                self.investigated_locations.append(
                    (
                        tx,
                        ty,
                        ex,
                        ey,
                        ez,
                        self.investigate_class,
                        self.investigate_confidence,
                    )
                )
                self.investigation_count += 1
                reason = "spiral complete" if spiral_done else "timeout"
                event = (
                    f"Investigation #{self.investigation_count} complete "
                    f"({self.investigate_class}, {reason}). Resuming search."
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
                yaw = self._yaw_from_current_position(tx, ty)
                self._publish_setpoint(tx, ty, self.cruise_altitude, yaw)
            else:
                self.investigate_approaching = False
                wp = self.investigate_spiral.step(self.dt)
                if wp is not None:
                    sx, sy, _sz, _syaw = wp
                    ox = tx + sx
                    oy = ty + sy
                    yaw = self._yaw_from_current_position(tx, ty)
                    self._publish_setpoint(ox, oy, self.cruise_altitude, yaw)

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
