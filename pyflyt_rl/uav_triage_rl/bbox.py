"""Analytic bounding-box detector for fast non-YOLO training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from uav_triage_rl.perception import CameraConfig, DetectionObservation, PerceptionConfig
from uav_triage_rl.world import VictimSpec


@dataclass(frozen=True)
class BBoxSimConfig:
    """Controls simulated detector box instability."""

    body_length_m: float = 1.25
    body_width_m: float = 0.42
    body_height_m: float = 0.25
    center_jitter_px: float = 6.0
    size_jitter_fraction: float = 0.12
    temporal_jitter_alpha: float = 0.65
    dropout_rate: float = 0.08
    duplicate_rate: float = 0.0
    localization_noise_m: float = 0.35
    confidence_mean: float = 0.78
    confidence_std: float = 0.12
    min_box_area_px2: float = 12.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "BBoxSimConfig":
        raw = dict(raw or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw.items() if key in allowed})


@dataclass(frozen=True)
class SimulatedBBox:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str
    truth_id: int | None
    world_xy: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "xyxy": [float(value) for value in self.xyxy],
            "confidence": float(self.confidence),
            "class_name": self.class_name,
            "truth_id": self.truth_id,
            "world_xy": [float(self.world_xy[0]), float(self.world_xy[1])],
        }


class BBoxPerception:
    """Projects known victims into camera boxes with configurable detector noise."""

    def __init__(
        self,
        *,
        camera: CameraConfig,
        perception: PerceptionConfig,
        config: BBoxSimConfig,
    ) -> None:
        self.camera = camera
        self.perception = perception
        self.config = config
        self._jitter_state: dict[int, tuple[float, float]] = {}
        self.last_bboxes: list[SimulatedBBox] = []

    def reset(self) -> None:
        self._jitter_state.clear()
        self.last_bboxes = []

    def visible_cells(
        self,
        *,
        geometry: Any,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
    ) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for row, col, cx, cy in geometry.iter_cell_centers():
            if self.is_visible(
                drone_x=drone_x,
                drone_y=drone_y,
                drone_z=drone_z,
                drone_yaw=drone_yaw,
                target_x=cx,
                target_y=cy,
            ):
                cells.append((row, col))
        return cells

    def is_visible(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
        target_x: float,
        target_y: float,
        target_height_m: float | None = None,
    ) -> bool:
        projected = self._project_point(
            point=np.array([target_x, target_y, target_height_m or 0.0], dtype=float),
            drone_x=drone_x,
            drone_y=drone_y,
            drone_z=drone_z,
            drone_yaw=drone_yaw,
        )
        if projected is None:
            return False
        u, v, _depth = projected
        return 0.0 <= u <= self.camera.image_width_px - 1 and 0.0 <= v <= self.camera.image_height_px - 1

    def detect(
        self,
        victims: list[VictimSpec],
        *,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
        rng: np.random.Generator,
        investigating: bool,
        visible_cells: list[tuple[int, int]],
        geometry: Any,
    ) -> list[DetectionObservation]:
        observations: list[DetectionObservation] = []
        bboxes: list[SimulatedBBox] = []
        probability = (
            self.perception.investigate_detection_probability
            if investigating
            else self.perception.detection_probability
        )
        probability = max(0.0, min(1.0, probability - self.config.dropout_rate))

        for victim in victims:
            bbox = self._victim_bbox(
                victim,
                drone_x=drone_x,
                drone_y=drone_y,
                drone_z=drone_z,
                drone_yaw=drone_yaw,
                rng=rng,
            )
            if bbox is None or rng.random() > probability:
                continue
            bboxes.append(bbox)
            observations.append(self._observation_from_bbox(victim, bbox, rng=rng))
            if self.config.duplicate_rate > 0.0 and rng.random() < self.config.duplicate_rate:
                duplicate = self._jitter_bbox(
                    bbox,
                    truth_id=victim.id + 10_000,
                    rng=rng,
                    extra_scale=1.6,
                )
                bboxes.append(duplicate)
                observations.append(self._observation_from_bbox(victim, duplicate, rng=rng))

        if self.perception.false_positive_rate > 0.0 and visible_cells:
            if rng.random() < self.perception.false_positive_rate:
                row, col = visible_cells[int(rng.integers(0, len(visible_cells)))]
                x, y = geometry.grid_to_world(row, col)
                fp_box = self._false_positive_bbox(x, y, rng=rng)
                bboxes.append(fp_box)
                observations.append(
                    DetectionObservation(
                        x=x,
                        y=y,
                        confidence=fp_box.confidence,
                        class_name=fp_box.class_name,
                        truth_id=None,
                        bbox_xyxy=fp_box.xyxy,
                        source="bbox_false_positive",
                    )
                )

        self.last_bboxes = bboxes
        return observations

    def _camera_basis(self, drone_yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array([math.cos(drone_yaw), math.sin(drone_yaw), 0.0], dtype=float)
        right = np.array([-math.sin(drone_yaw), math.cos(drone_yaw), 0.0], dtype=float)
        down = np.array([0.0, 0.0, -1.0], dtype=float)
        pitch = math.radians(self.camera.camera_pitch_deg)
        optical = math.cos(pitch) * forward + math.sin(pitch) * down
        optical /= max(float(np.linalg.norm(optical)), 1e-9)
        image_down = np.cross(right, optical)
        image_down /= max(float(np.linalg.norm(image_down)), 1e-9)
        return right, image_down, optical

    def _project_point(
        self,
        *,
        point: np.ndarray,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
    ) -> tuple[float, float, float] | None:
        right, image_down, optical = self._camera_basis(drone_yaw)
        rel = point - np.array([drone_x, drone_y, drone_z], dtype=float)
        z_cam = float(np.dot(rel, optical))
        if z_cam < self.camera.min_range_m or z_cam > self.camera.max_range_m:
            return None
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, image_down))
        fx = (self.camera.image_width_px / 2.0) / max(math.tan(self.camera.horizontal_fov_rad / 2.0), 1e-9)
        fy = (self.camera.image_height_px / 2.0) / max(math.tan(self.camera.vertical_fov_rad / 2.0), 1e-9)
        u = self.camera.image_width_px / 2.0 + fx * (x_cam / z_cam)
        v = self.camera.image_height_px / 2.0 + fy * (y_cam / z_cam)
        return u, v, z_cam

    def _victim_corners(self, victim: VictimSpec) -> list[np.ndarray]:
        length = max(0.1, self.config.body_length_m)
        width = max(0.1, self.config.body_width_m)
        height = max(0.05, self.config.body_height_m)
        forward = np.array([math.cos(victim.yaw), math.sin(victim.yaw), 0.0], dtype=float)
        side = np.array([-math.sin(victim.yaw), math.cos(victim.yaw), 0.0], dtype=float)
        center = np.array([victim.x, victim.y, 0.0], dtype=float)
        points: list[np.ndarray] = []
        for fwd in (-length / 2.0, length / 2.0):
            for lat in (-width / 2.0, width / 2.0):
                for z in (0.03, height):
                    point = center + forward * fwd + side * lat
                    point[2] = z
                    points.append(point)
        return points

    def _victim_bbox(
        self,
        victim: VictimSpec,
        *,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        drone_yaw: float,
        rng: np.random.Generator,
    ) -> SimulatedBBox | None:
        projected = [
            self._project_point(
                point=point,
                drone_x=drone_x,
                drone_y=drone_y,
                drone_z=drone_z,
                drone_yaw=drone_yaw,
            )
            for point in self._victim_corners(victim)
        ]
        valid = [point for point in projected if point is not None]
        if not valid:
            return None

        xs = [point[0] for point in valid]
        ys = [point[1] for point in valid]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(self.camera.image_width_px - 1), max(xs))
        y2 = min(float(self.camera.image_height_px - 1), max(ys))
        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) * (y2 - y1) < self.config.min_box_area_px2:
            return None

        confidence = float(
            np.clip(
                rng.normal(self.config.confidence_mean, self.config.confidence_std),
                0.05,
                0.99,
            )
        )
        bbox = SimulatedBBox(
            xyxy=(x1, y1, x2, y2),
            confidence=confidence,
            class_name="person",
            truth_id=victim.id,
            world_xy=(victim.x, victim.y),
        )
        return self._jitter_bbox(bbox, truth_id=victim.id, rng=rng)

    def _jitter_bbox(
        self,
        bbox: SimulatedBBox,
        *,
        truth_id: int,
        rng: np.random.Generator,
        extra_scale: float = 1.0,
    ) -> SimulatedBBox:
        x1, y1, x2, y2 = bbox.xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        alpha = max(0.0, min(0.98, self.config.temporal_jitter_alpha))
        prev_x, prev_y = self._jitter_state.get(truth_id, (0.0, 0.0))
        fresh_x = float(rng.normal(0.0, self.config.center_jitter_px * extra_scale))
        fresh_y = float(rng.normal(0.0, self.config.center_jitter_px * extra_scale))
        jitter_x = alpha * prev_x + (1.0 - alpha) * fresh_x
        jitter_y = alpha * prev_y + (1.0 - alpha) * fresh_y
        self._jitter_state[truth_id] = (jitter_x, jitter_y)

        scale_w = float(np.clip(1.0 + rng.normal(0.0, self.config.size_jitter_fraction), 0.45, 1.8))
        scale_h = float(np.clip(1.0 + rng.normal(0.0, self.config.size_jitter_fraction), 0.45, 1.8))
        cx += jitter_x
        cy += jitter_y
        min_size = 2.0
        max_width = max(min_size, self.camera.image_width_px - 1.0)
        max_height = max(min_size, self.camera.image_height_px - 1.0)
        width = min(max(min_size, width * scale_w), max_width)
        height = min(max(min_size, height * scale_h), max_height)
        cx = float(np.clip(cx, width / 2.0, self.camera.image_width_px - 1.0 - width / 2.0))
        cy = float(np.clip(cy, height / 2.0, self.camera.image_height_px - 1.0 - height / 2.0))
        new_x1 = cx - width / 2.0
        new_y1 = cy - height / 2.0
        new_x2 = cx + width / 2.0
        new_y2 = cy + height / 2.0
        return SimulatedBBox(
            xyxy=(new_x1, new_y1, new_x2, new_y2),
            confidence=bbox.confidence,
            class_name=bbox.class_name,
            truth_id=bbox.truth_id,
            world_xy=bbox.world_xy,
        )

    def _false_positive_bbox(self, x: float, y: float, *, rng: np.random.Generator) -> SimulatedBBox:
        width = float(rng.uniform(10.0, 48.0))
        height = float(rng.uniform(10.0, 48.0))
        cx = float(rng.uniform(width / 2.0, self.camera.image_width_px - width / 2.0))
        cy = float(rng.uniform(height / 2.0, self.camera.image_height_px - height / 2.0))
        return SimulatedBBox(
            xyxy=(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0),
            confidence=float(rng.uniform(0.28, 0.65)),
            class_name="person",
            truth_id=None,
            world_xy=(x, y),
        )

    def _observation_from_bbox(
        self,
        victim: VictimSpec,
        bbox: SimulatedBBox,
        *,
        rng: np.random.Generator,
    ) -> DetectionObservation:
        noise = max(0.0, self.config.localization_noise_m)
        return DetectionObservation(
            x=float(victim.x + rng.normal(0.0, noise)),
            y=float(victim.y + rng.normal(0.0, noise)),
            confidence=bbox.confidence,
            class_name=bbox.class_name,
            truth_id=victim.id,
            bbox_xyxy=bbox.xyxy,
            source="bbox",
        )
