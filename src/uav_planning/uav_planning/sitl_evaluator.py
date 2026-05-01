#!/usr/bin/env python3
"""Quantitative SITL evaluator for Gazebo search runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


TARGET_CLASS_HINTS = {
    "person": ("person", "female", "male", "visitor", "walking"),
    "truck": ("truck",),
    "bus": ("bus",),
    "motorcycle": ("motorcycle",),
    "bicycle": ("bicycle",),
    "car": ("car", "suv", "hatchback", "vehicle"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def parse_pose_xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    values = [float(item) for item in text.split()]
    while len(values) < 3:
        values.append(0.0)
    return (values[0], values[1], values[2])


def infer_target_class(*values: str) -> str | None:
    text = " ".join(values).lower()
    for class_name, hints in TARGET_CLASS_HINTS.items():
        if any(hint in text for hint in hints):
            return class_name
    return None


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quat_to_yaw_enu(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass(frozen=True)
class TruthTarget:
    target_id: int
    name: str
    class_name: str
    enu_x: float
    enu_y: float
    enu_z: float
    ned_x: float
    ned_y: float
    is_victim: bool
    in_search_area: bool = True

    def horizontal_error_enu(self, x: float, y: float) -> float:
        return math.hypot(float(x) - self.enu_x, float(y) - self.enu_y)


@dataclass
class TargetMetrics:
    detection_count: int = 0
    first_detection_s: float | None = None
    last_detection_s: float | None = None
    first_detection_error_m: float | None = None
    best_detection_error_m: float | None = None
    max_confidence: float = 0.0
    visible_count: int = 0
    first_visible_s: float | None = None
    report_count: int = 0
    first_report_s: float | None = None
    first_report_error_m: float | None = None
    best_report_error_m: float | None = None


@dataclass(frozen=True)
class SearchGeometry:
    width_m: float
    height_m: float
    origin_north_m: float
    origin_east_m: float
    cell_size_m: float

    @property
    def rows(self) -> int:
        return max(1, int(math.ceil(self.height_m / max(self.cell_size_m, 1e-6))))

    @property
    def cols(self) -> int:
        return max(1, int(math.ceil(self.width_m / max(self.cell_size_m, 1e-6))))

    @property
    def north_limits(self) -> tuple[float, float]:
        return (
            self.origin_north_m - self.height_m / 2.0,
            self.origin_north_m + self.height_m / 2.0,
        )

    @property
    def east_limits(self) -> tuple[float, float]:
        return (
            self.origin_east_m - self.width_m / 2.0,
            self.origin_east_m + self.width_m / 2.0,
        )

    def contains_ned(self, north_m: float, east_m: float) -> bool:
        north_min, north_max = self.north_limits
        east_min, east_max = self.east_limits
        return north_min <= north_m <= north_max and east_min <= east_m <= east_max

    def grid_index(self, north_m: float, east_m: float) -> tuple[int, int] | None:
        if not self.contains_ned(north_m, east_m):
            return None
        north_min, north_max = self.north_limits
        east_min, east_max = self.east_limits
        row = int(
            min(
                self.rows - 1,
                max(0, ((north_m - north_min) / max(north_max - north_min, 1e-6)) * self.rows),
            )
        )
        col = int(
            min(
                self.cols - 1,
                max(0, ((east_m - east_min) / max(east_max - east_min, 1e-6)) * self.cols),
            )
        )
        return row, col

    def iter_cell_centers(self):
        north_min, north_max = self.north_limits
        east_min, east_max = self.east_limits
        for row in range(self.rows):
            north = north_min + (row + 0.5) * ((north_max - north_min) / self.rows)
            for col in range(self.cols):
                east = east_min + (col + 0.5) * ((east_max - east_min) / self.cols)
                yield row, col, north, east


@dataclass(frozen=True)
class ForwardFrustum:
    horizontal_fov_rad: float = 1.274
    image_width_px: int = 640
    image_height_px: int = 480
    depth_near_m: float = 0.2
    depth_far_m: float = 19.1
    target_height_m: float = 1.4

    @property
    def vertical_fov_rad(self) -> float:
        aspect = self.image_height_px / max(float(self.image_width_px), 1.0)
        return 2.0 * math.atan(math.tan(self.horizontal_fov_rad / 2.0) * aspect)

    def is_visible_ned(
        self,
        *,
        drone_north_m: float,
        drone_east_m: float,
        drone_yaw_ned_rad: float,
        altitude_ned_m: float,
        target_north_m: float,
        target_east_m: float,
        target_height_m: float | None = None,
    ) -> bool:
        dx = target_north_m - drone_north_m
        dy = target_east_m - drone_east_m
        horizontal_range = math.hypot(dx, dy)
        if horizontal_range < self.depth_near_m or horizontal_range > self.depth_far_m:
            return False

        bearing = wrap_angle_rad(math.atan2(dy, dx) - drone_yaw_ned_rad)
        if abs(bearing) > self.horizontal_fov_rad / 2.0:
            return False

        target_height = self.target_height_m if target_height_m is None else target_height_m
        vertical_drop = max(0.0, abs(altitude_ned_m) - max(0.0, target_height))
        down_angle = math.atan2(vertical_drop, max(horizontal_range, 1e-6))
        return down_angle <= self.vertical_fov_rad / 2.0


def load_world_targets(world_path: str | Path, *, target_filter: str = "people") -> list[TruthTarget]:
    resolved = resolve_repo_path(world_path)
    tree = ET.parse(resolved)
    root = tree.getroot()
    targets: list[TruthTarget] = []

    def should_keep(name: str, class_name: str, is_victim: bool) -> bool:
        selected = target_filter.strip().lower()
        if selected in ("people", "persons", "person"):
            return class_name == "person"
        if selected in ("victims", "victim"):
            return class_name == "person" and is_victim
        if selected in ("all", "all_detectable", "detectable"):
            return class_name in TARGET_CLASS_HINTS
        if selected:
            return selected in name.lower() or selected == class_name
        return class_name == "person"

    for include in root.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        uri = (include.findtext("uri") or "").strip()
        class_name = infer_target_class(uri, name)
        if class_name is None:
            continue
        is_victim = "victim" in name.lower()
        if not should_keep(name, class_name, is_victim):
            continue
        enu_x, enu_y, enu_z = parse_pose_xyz(include.findtext("pose"))
        targets.append(
            TruthTarget(
                target_id=len(targets),
                name=name or f"{class_name}_{len(targets)}",
                class_name=class_name,
                enu_x=enu_x,
                enu_y=enu_y,
                enu_z=enu_z,
                ned_x=enu_y,
                ned_y=enu_x,
                is_victim=is_victim,
            )
        )

    return targets


class SitlEvaluator(Node):
    """Collect ground-truth localization and mission metrics during SITL runs."""

    def __init__(self) -> None:
        super().__init__("sitl_evaluator")

        self.declare_parameter("world_path", "worlds/search_area.sdf")
        self.declare_parameter("output_dir", "artifacts/sitl_eval")
        self.declare_parameter("run_id", "")
        self.declare_parameter("policy_label", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("target_filter", "people")
        self.declare_parameter("search_bounds_only", True)
        self.declare_parameter("grid.width", 40.0)
        self.declare_parameter("grid.height", 40.0)
        self.declare_parameter("grid.origin_x", 0.0)
        self.declare_parameter("grid.origin_y", 0.0)
        self.declare_parameter("coverage_cell_size_m", 4.0)
        self.declare_parameter("match_radius_m", 4.0)
        self.declare_parameter("report_radius_m", 4.0)
        self.declare_parameter("sample_period_s", 2.0)
        self.declare_parameter("summary_period_s", 10.0)
        self.declare_parameter("duration_s", 0.0)
        self.declare_parameter("stop_on_terminal_phase", True)

        self.world_path = str(self.get_parameter("world_path").value)
        self.policy_label = str(self.get_parameter("policy_label").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.target_filter = str(self.get_parameter("target_filter").value)
        self.search_bounds_only = bool(self.get_parameter("search_bounds_only").value)
        self.match_radius_m = max(0.1, float(self.get_parameter("match_radius_m").value))
        self.report_radius_m = max(0.1, float(self.get_parameter("report_radius_m").value))
        self.sample_period_s = max(0.5, float(self.get_parameter("sample_period_s").value))
        self.summary_period_s = max(1.0, float(self.get_parameter("summary_period_s").value))
        self.duration_s = max(0.0, float(self.get_parameter("duration_s").value))
        self.stop_on_terminal_phase = bool(self.get_parameter("stop_on_terminal_phase").value)

        self.geometry = SearchGeometry(
            width_m=float(self.get_parameter("grid.width").value),
            height_m=float(self.get_parameter("grid.height").value),
            origin_north_m=float(self.get_parameter("grid.origin_x").value),
            origin_east_m=float(self.get_parameter("grid.origin_y").value),
            cell_size_m=max(0.5, float(self.get_parameter("coverage_cell_size_m").value)),
        )
        self.frustum = ForwardFrustum()

        loaded_targets = [
            replace(
                target,
                in_search_area=self.geometry.contains_ned(target.ned_x, target.ned_y),
            )
            for target in load_world_targets(self.world_path, target_filter=self.target_filter)
        ]
        if self.search_bounds_only:
            self.targets = [
                target
                for target in loaded_targets
                if self.geometry.contains_ned(target.ned_x, target.ned_y)
            ]
        else:
            self.targets = loaded_targets
        self.target_metrics = {target.target_id: TargetMetrics() for target in self.targets}

        run_id = str(self.get_parameter("run_id").value).strip()
        if not run_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            suffix = self.policy_label.strip() or "sitl"
            run_id = f"{stamp}_{suffix}"
        output_root = resolve_repo_path(str(self.get_parameter("output_dir").value))
        self.run_dir = output_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.truth_path = self.run_dir / "ground_truth.json"
        self.truth_path.write_text(
            json.dumps([asdict(target) for target in self.targets], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        self.completed = False
        self.node_start_s: float | None = None
        self.mission_start_s: float | None = None
        self.last_sample_s: float | None = None
        self.last_summary_s: float | None = None
        self.active_controller = ""
        self.phase = "UNKNOWN"
        self.phase_start_s: float | None = None
        self.phase_durations: dict[str, float] = {}
        self.last_pose: tuple[float, float, float] | None = None
        self.last_pose_s: float | None = None
        self.path_length_m = 0.0
        self.search_path_length_m = 0.0
        self.min_altitude_m: float | None = None
        self.max_altitude_m: float | None = None
        self.last_pose_summary: dict[str, float] = {}
        self.path_cells: set[tuple[int, int]] = set()
        self.visible_cells: set[tuple[int, int]] = set()
        self.processed_report_markers: set[str] = set()
        self.total_detection_messages = 0
        self.total_detections = 0
        self.matched_detections = 0
        self.false_detections = 0
        self.total_reports = 0
        self.matched_reports = 0
        self.false_reports = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(PoseStamped, "/uav/pose", self._pose_cb, qos)
        self.create_subscription(Detection3DArray, "/detections_3d", self._detections_cb, 10)
        self.create_subscription(String, "/mission_controller/state", self._state_cb_factory("mission_controller"), 10)
        self.create_subscription(String, "/rl_mission_controller/state", self._state_cb_factory("rl_mission_controller"), 10)
        self.create_subscription(String, "/mission_controller/events", self._event_cb_factory("mission_controller"), 10)
        self.create_subscription(String, "/rl_mission_controller/events", self._event_cb_factory("rl_mission_controller"), 10)
        self.create_subscription(MarkerArray, "/mission_controller/markers", self._markers_cb_factory("mission_controller"), 10)
        self.create_subscription(MarkerArray, "/rl_mission_controller/markers", self._markers_cb_factory("rl_mission_controller"), 10)
        self.summary_pub = self.create_publisher(String, "/sitl_eval/summary", 10)
        self.create_timer(0.5, self._timer_cb)

        self._log_event(
            "evaluator_started",
            {
                "run_dir": str(self.run_dir),
                "world_path": self.world_path,
                "target_filter": self.target_filter,
                "target_count": len(self.targets),
                "search_bounds_only": self.search_bounds_only,
            },
        )
        self.get_logger().info(
            f"SITL evaluator writing {self.run_dir} "
            f"for {len(self.targets)} '{self.target_filter}' target(s)"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _elapsed_s(self) -> float:
        now_s = self._now_s()
        anchor = self.mission_start_s if self.mission_start_s is not None else self.node_start_s
        if anchor is None:
            return 0.0
        return max(0.0, now_s - anchor)

    def _log_event(self, event_type: str, payload: dict) -> None:
        record = {
            "type": event_type,
            "sim_time_s": self._now_s(),
            "elapsed_s": self._elapsed_s(),
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _state_cb_factory(self, controller: str):
        def callback(msg: String) -> None:
            phase = str(msg.data).strip() or "UNKNOWN"
            now_s = self._now_s()
            if self.active_controller != controller:
                self.active_controller = controller
            if phase == self.phase:
                return
            if self.phase_start_s is not None:
                self.phase_durations[self.phase] = self.phase_durations.get(self.phase, 0.0) + max(
                    0.0,
                    now_s - self.phase_start_s,
                )
            old_phase = self.phase
            self.phase = phase
            self.phase_start_s = now_s
            if self.mission_start_s is None and phase == "SEARCH":
                self.mission_start_s = now_s
            self._log_event(
                "phase",
                {"controller": controller, "old_phase": old_phase, "new_phase": phase},
            )
            if self.stop_on_terminal_phase and phase == "DONE":
                self.completed = True

        return callback

    def _event_cb_factory(self, controller: str):
        def callback(msg: String) -> None:
            self._log_event(
                "controller_event",
                {"controller": controller, "message": str(msg.data)},
            )

        return callback

    def _markers_cb_factory(self, controller: str):
        def callback(msg: MarkerArray) -> None:
            for marker in msg.markers:
                if marker.ns != "investigated" or marker.action != Marker.ADD:
                    continue
                key = f"{controller}:{marker.id}"
                if key in self.processed_report_markers:
                    continue
                self.processed_report_markers.add(key)
                self._process_report(
                    controller=controller,
                    report_id=marker.id,
                    enu_x=float(marker.pose.position.x),
                    enu_y=float(marker.pose.position.y),
                    enu_z=float(marker.pose.position.z),
                )

        return callback

    def _pose_cb(self, msg: PoseStamped) -> None:
        now_s = self._now_s()
        if self.node_start_s is None and now_s > 0.0:
            self.node_start_s = now_s

        enu_x = float(msg.pose.position.x)
        enu_y = float(msg.pose.position.y)
        enu_z = float(msg.pose.position.z)
        ned_x = enu_y
        ned_y = enu_x
        ned_z = -enu_z
        altitude_m = enu_z
        self.min_altitude_m = altitude_m if self.min_altitude_m is None else min(self.min_altitude_m, altitude_m)
        self.max_altitude_m = altitude_m if self.max_altitude_m is None else max(self.max_altitude_m, altitude_m)
        self.last_pose_summary = {
            "enu_x": enu_x,
            "enu_y": enu_y,
            "enu_z": enu_z,
            "ned_x": ned_x,
            "ned_y": ned_y,
            "ned_z": ned_z,
        }

        if self.last_pose is not None:
            dx = enu_x - self.last_pose[0]
            dy = enu_y - self.last_pose[1]
            dz = enu_z - self.last_pose[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if math.isfinite(distance) and distance < 25.0:
                self.path_length_m += distance
                if self.mission_start_s is not None and self.phase in ("SEARCH", "INVESTIGATE"):
                    self.search_path_length_m += distance
        self.last_pose = (enu_x, enu_y, enu_z)
        self.last_pose_s = now_s

        cell = self.geometry.grid_index(ned_x, ned_y)
        if cell is not None and self.mission_start_s is not None:
            self.path_cells.add(cell)

        orientation = msg.pose.orientation
        yaw_enu = quat_to_yaw_enu(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        yaw_ned = wrap_angle_rad(math.pi / 2.0 - yaw_enu)
        if self.mission_start_s is not None:
            visible_now: list[int] = []
            for target in self.targets:
                if self.frustum.is_visible_ned(
                    drone_north_m=ned_x,
                    drone_east_m=ned_y,
                    drone_yaw_ned_rad=yaw_ned,
                    altitude_ned_m=ned_z,
                    target_north_m=target.ned_x,
                    target_east_m=target.ned_y,
                ):
                    visible_now.append(target.target_id)
            for row, col, north, east in self.geometry.iter_cell_centers():
                if self.frustum.is_visible_ned(
                    drone_north_m=ned_x,
                    drone_east_m=ned_y,
                    drone_yaw_ned_rad=yaw_ned,
                    altitude_ned_m=ned_z,
                    target_north_m=north,
                    target_east_m=east,
                ):
                    self.visible_cells.add((row, col))
            for target_id in visible_now:
                metrics = self.target_metrics[target_id]
                metrics.visible_count += 1
                if metrics.first_visible_s is None:
                    metrics.first_visible_s = self._elapsed_s()

    def _detections_cb(self, msg: Detection3DArray) -> None:
        self.total_detection_messages += 1
        matched_in_msg = 0
        false_in_msg = 0
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            class_name = str(hyp.hypothesis.class_id)
            confidence = float(hyp.hypothesis.score)
            enu_x = float(hyp.pose.pose.position.x)
            enu_y = float(hyp.pose.pose.position.y)
            enu_z = float(hyp.pose.pose.position.z)
            self.total_detections += 1
            target, error_m = self._nearest_target(enu_x, enu_y, class_name=class_name)
            if target is not None and error_m <= self.match_radius_m:
                self.matched_detections += 1
                matched_in_msg += 1
                metrics = self.target_metrics[target.target_id]
                elapsed_s = self._elapsed_s()
                metrics.detection_count += 1
                metrics.last_detection_s = elapsed_s
                metrics.max_confidence = max(metrics.max_confidence, confidence)
                if metrics.first_detection_s is None:
                    metrics.first_detection_s = elapsed_s
                    metrics.first_detection_error_m = error_m
                if metrics.best_detection_error_m is None or error_m < metrics.best_detection_error_m:
                    metrics.best_detection_error_m = error_m
            else:
                self.false_detections += 1
                false_in_msg += 1

        if matched_in_msg or false_in_msg:
            self._log_event(
                "detections",
                {
                    "detections": len(msg.detections),
                    "matched": matched_in_msg,
                    "false": false_in_msg,
                },
            )

    def _process_report(
        self,
        *,
        controller: str,
        report_id: int,
        enu_x: float,
        enu_y: float,
        enu_z: float,
    ) -> None:
        self.total_reports += 1
        target, error_m = self._nearest_target(enu_x, enu_y, class_name="person")
        payload = {
            "controller": controller,
            "report_id": report_id,
            "enu_x": enu_x,
            "enu_y": enu_y,
            "enu_z": enu_z,
            "matched": False,
            "error_m": None,
            "target": None,
        }
        if target is not None and error_m <= self.report_radius_m:
            self.matched_reports += 1
            metrics = self.target_metrics[target.target_id]
            elapsed_s = self._elapsed_s()
            metrics.report_count += 1
            if metrics.first_report_s is None:
                metrics.first_report_s = elapsed_s
                metrics.first_report_error_m = error_m
            if metrics.best_report_error_m is None or error_m < metrics.best_report_error_m:
                metrics.best_report_error_m = error_m
            payload.update(
                {
                    "matched": True,
                    "error_m": error_m,
                    "target": target.name,
                    "target_id": target.target_id,
                }
            )
        else:
            self.false_reports += 1
        self._log_event("reported_location", payload)

    def _nearest_target(
        self,
        enu_x: float,
        enu_y: float,
        *,
        class_name: str,
    ) -> tuple[TruthTarget | None, float]:
        best_target: TruthTarget | None = None
        best_error = float("inf")
        for target in self.targets:
            if class_name and target.class_name != class_name:
                continue
            error_m = target.horizontal_error_enu(enu_x, enu_y)
            if error_m < best_error:
                best_error = error_m
                best_target = target
        return best_target, best_error

    def _target_records(self) -> list[dict]:
        records: list[dict] = []
        for target in self.targets:
            metrics = self.target_metrics[target.target_id]
            records.append(
                {
                    **asdict(target),
                    **asdict(metrics),
                    "detected": metrics.first_detection_s is not None,
                    "reported": metrics.first_report_s is not None,
                    "visible_before_detection": (
                        metrics.first_visible_s is not None
                        and metrics.first_detection_s is not None
                        and metrics.first_visible_s <= metrics.first_detection_s
                    ),
                }
            )
        return records

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return float(statistics.fmean(values)) if values else None

    @staticmethod
    def _median(values: list[float]) -> float | None:
        return float(statistics.median(values)) if values else None

    def _summary(self) -> dict:
        targets = self._target_records()
        detection_errors = [
            float(item["best_detection_error_m"])
            for item in targets
            if item["best_detection_error_m"] is not None
        ]
        report_errors = [
            float(item["best_report_error_m"])
            for item in targets
            if item["best_report_error_m"] is not None
        ]
        first_detection_times = [
            float(item["first_detection_s"])
            for item in targets
            if item["first_detection_s"] is not None
        ]
        first_report_times = [
            float(item["first_report_s"])
            for item in targets
            if item["first_report_s"] is not None
        ]
        total_cells = self.geometry.rows * self.geometry.cols
        phase_durations = dict(self.phase_durations)
        if self.phase_start_s is not None:
            phase_durations[self.phase] = phase_durations.get(self.phase, 0.0) + max(
                0.0,
                self._now_s() - self.phase_start_s,
            )

        return {
            "run_dir": str(self.run_dir),
            "policy_label": self.policy_label,
            "model_path": self.model_path,
            "world_path": self.world_path,
            "target_filter": self.target_filter,
            "search_bounds_only": self.search_bounds_only,
            "stop_on_terminal_phase": self.stop_on_terminal_phase,
            "active_controller": self.active_controller,
            "phase": self.phase,
            "elapsed_s": self._elapsed_s(),
            "mission_started": self.mission_start_s is not None,
            "phase_durations_s": phase_durations,
            "ground_truth_targets": len(targets),
            "targets_visible": sum(1 for item in targets if item["first_visible_s"] is not None),
            "targets_detected": sum(1 for item in targets if item["detected"]),
            "targets_reported": sum(1 for item in targets if item["reported"]),
            "visible_not_detected": sum(
                1
                for item in targets
                if item["first_visible_s"] is not None and not item["detected"]
            ),
            "detected_not_reported": sum(
                1 for item in targets if item["detected"] and not item["reported"]
            ),
            "time_to_first_detection_s": min(first_detection_times) if first_detection_times else None,
            "time_to_first_report_s": min(first_report_times) if first_report_times else None,
            "mean_first_detection_time_s": self._mean(first_detection_times),
            "mean_first_report_time_s": self._mean(first_report_times),
            "mean_best_detection_error_m": self._mean(detection_errors),
            "median_best_detection_error_m": self._median(detection_errors),
            "mean_best_report_error_m": self._mean(report_errors),
            "median_best_report_error_m": self._median(report_errors),
            "detection_messages": self.total_detection_messages,
            "detections_total": self.total_detections,
            "detections_matched": self.matched_detections,
            "detections_false": self.false_detections,
            "reported_locations_total": self.total_reports,
            "reported_locations_matched": self.matched_reports,
            "reported_locations_false": self.false_reports,
            "path_length_m": self.path_length_m,
            "search_path_length_m": self.search_path_length_m,
            "path_coverage_fraction": len(self.path_cells) / max(float(total_cells), 1.0),
            "visible_coverage_fraction": len(self.visible_cells) / max(float(total_cells), 1.0),
            "coverage_rows": self.geometry.rows,
            "coverage_cols": self.geometry.cols,
            "min_altitude_m": self.min_altitude_m,
            "max_altitude_m": self.max_altitude_m,
            "last_pose": self.last_pose_summary,
            "targets": targets,
        }

    def _write_summary(self) -> dict:
        summary = self._summary()
        tmp_path = self.summary_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.summary_path)
        return summary

    def _timer_cb(self) -> None:
        now_s = self._now_s()
        if self.node_start_s is None and now_s > 0.0:
            self.node_start_s = now_s

        elapsed_s = self._elapsed_s()
        if self.last_sample_s is None or now_s - self.last_sample_s >= self.sample_period_s:
            summary = self._summary()
            sample = {
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "targets",
                    "phase_durations_s",
                    "last_pose",
                }
            }
            with self.samples_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sample, sort_keys=True) + "\n")
            self.last_sample_s = now_s

        if self.last_summary_s is None or now_s - self.last_summary_s >= self.summary_period_s:
            summary = self._write_summary()
            msg = String()
            msg.data = json.dumps(
                {
                    "elapsed_s": summary["elapsed_s"],
                    "phase": summary["phase"],
                    "targets_detected": summary["targets_detected"],
                    "targets_reported": summary["targets_reported"],
                    "ground_truth_targets": summary["ground_truth_targets"],
                    "mean_best_detection_error_m": summary["mean_best_detection_error_m"],
                    "visible_coverage_fraction": summary["visible_coverage_fraction"],
                },
                sort_keys=True,
            )
            self.summary_pub.publish(msg)
            self.last_summary_s = now_s

        if self.duration_s > 0.0 and elapsed_s >= self.duration_s:
            self.completed = True

    def finalize(self) -> None:
        summary = self._write_summary()
        self._log_event(
            "evaluator_finished",
            {
                "targets_detected": summary["targets_detected"],
                "targets_reported": summary["targets_reported"],
                "elapsed_s": summary["elapsed_s"],
            },
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SitlEvaluator()
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
