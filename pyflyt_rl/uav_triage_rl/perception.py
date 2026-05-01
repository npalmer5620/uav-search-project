"""Synthetic training perception and lightweight detection memory."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from uav_triage_rl.world import VictimSpec


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class CameraConfig:
    image_width_px: int = 320
    image_height_px: int = 240
    horizontal_fov_deg: float = 90.0
    camera_pitch_deg: float = 55.0
    min_range_m: float = 0.5
    max_range_m: float = 22.0
    target_height_m: float = 0.25

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "CameraConfig":
        return cls(**dict(raw or {}))

    @property
    def horizontal_fov_rad(self) -> float:
        return math.radians(self.horizontal_fov_deg)

    @property
    def vertical_fov_rad(self) -> float:
        aspect = self.image_height_px / max(float(self.image_width_px), 1.0)
        return 2.0 * math.atan(math.tan(self.horizontal_fov_rad / 2.0) * aspect)


@dataclass(frozen=True)
class PerceptionConfig:
    mode: str = "bbox"
    detection_probability: float = 0.88
    investigate_detection_probability: float = 0.98
    false_positive_rate: float = 0.01
    detection_noise_m: float = 0.45
    match_distance_m: float = 2.5
    min_hits: int = 2
    track_timeout_steps: int = 18
    min_mean_confidence: float = 0.55

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "PerceptionConfig":
        raw = dict(raw or {})
        raw.pop("bbox", None)
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in raw.items() if key in allowed})
        if config.mode not in {"bbox", "point"}:
            raise ValueError("perception.mode must be 'bbox' or 'point'")
        return config


@dataclass(frozen=True)
class DetectionObservation:
    x: float
    y: float
    confidence: float
    class_name: str = "person"
    truth_id: int | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    source: str = "point"


@dataclass
class DetectionTrack:
    x: float
    y: float
    confidence_sum: float
    hits: int
    first_seen_step: int
    last_seen_step: int
    truth_votes: dict[int, int] = field(default_factory=dict)
    confirmed: bool = False

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / max(1, self.hits)

    @property
    def dominant_truth_id(self) -> int | None:
        if not self.truth_votes:
            return None
        return max(self.truth_votes.items(), key=lambda item: item[1])[0]

    def update(
        self,
        observation: DetectionObservation,
        *,
        step: int,
        alpha: float = 0.35,
        count_hit: bool = True,
    ) -> None:
        self.x = (1.0 - alpha) * self.x + alpha * observation.x
        self.y = (1.0 - alpha) * self.y + alpha * observation.y
        if count_hit:
            self.confidence_sum += float(observation.confidence)
            self.hits += 1
        elif self.hits > 0:
            self.confidence_sum = max(
                self.confidence_sum,
                float(observation.confidence) * self.hits,
            )
        self.last_seen_step = step
        if count_hit and observation.truth_id is not None:
            self.truth_votes[observation.truth_id] = self.truth_votes.get(observation.truth_id, 0) + 1


class DetectionMemory:
    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config
        self.tracks: list[DetectionTrack] = []
        self.confirmed_truth_ids: set[int] = set()

    def reset(self) -> None:
        self.tracks.clear()
        self.confirmed_truth_ids.clear()

    def update(self, observations: list[DetectionObservation], *, step: int) -> tuple[int, int]:
        self.tracks = [
            track
            for track in self.tracks
            if step - track.last_seen_step <= self.config.track_timeout_steps
        ]
        updated = 0
        for obs in observations:
            best_track = None
            best_distance = float("inf")
            for track in self.tracks:
                distance = math.hypot(track.x - obs.x, track.y - obs.y)
                if distance < best_distance and distance <= self.config.match_distance_m:
                    best_track = track
                    best_distance = distance
            if best_track is None:
                votes = {obs.truth_id: 1} if obs.truth_id is not None else {}
                self.tracks.append(
                    DetectionTrack(
                        x=obs.x,
                        y=obs.y,
                        confidence_sum=obs.confidence,
                        hits=1,
                        first_seen_step=step,
                        last_seen_step=step,
                        truth_votes=votes,
                    )
                )
            else:
                best_track.update(obs, step=step, count_hit=best_track.last_seen_step != step)
                updated += 1

        newly_confirmed = 0
        for track in self.tracks:
            if track.confirmed:
                continue
            if track.hits < self.config.min_hits:
                continue
            if track.mean_confidence < self.config.min_mean_confidence:
                continue
            track.confirmed = True
            truth_id = track.dominant_truth_id
            if truth_id is not None and truth_id not in self.confirmed_truth_ids:
                self.confirmed_truth_ids.add(truth_id)
                newly_confirmed += 1
        return updated, newly_confirmed

    def best_unconfirmed(self) -> DetectionTrack | None:
        candidates = [track for track in self.tracks if not track.confirmed]
        if not candidates:
            return None
        return max(candidates, key=lambda track: (track.mean_confidence, track.hits, -track.last_seen_step))

    def top_tracks(self, limit: int = 3) -> list[DetectionTrack]:
        return sorted(
            self.tracks,
            key=lambda track: (track.confirmed, track.mean_confidence, track.hits, track.last_seen_step),
            reverse=True,
        )[:limit]


class SyntheticPerception:
    """Fast victim visibility model for training without YOLO in the step loop."""

    def __init__(self, camera: CameraConfig, config: PerceptionConfig) -> None:
        self.camera = camera
        self.config = config

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
        dx = float(target_x) - float(drone_x)
        dy = float(target_y) - float(drone_y)
        horizontal_range = math.hypot(dx, dy)
        if horizontal_range < self.camera.min_range_m or horizontal_range > self.camera.max_range_m:
            return False

        bearing = wrap_angle(math.atan2(dy, dx) - float(drone_yaw))
        if abs(bearing) > self.camera.horizontal_fov_rad / 2.0:
            return False

        target_height = self.camera.target_height_m if target_height_m is None else target_height_m
        vertical_drop = max(0.0, float(drone_z) - max(0.0, target_height))
        down_angle = math.atan2(vertical_drop, max(horizontal_range, 1e-6))
        camera_pitch = math.radians(self.camera.camera_pitch_deg)
        return abs(down_angle - camera_pitch) <= self.camera.vertical_fov_rad / 2.0

    def visible_cells(self, *, geometry: Any, drone_x: float, drone_y: float, drone_z: float,
                      drone_yaw: float) -> list[tuple[int, int]]:
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
        probability = (
            self.config.investigate_detection_probability
            if investigating
            else self.config.detection_probability
        )
        for victim in victims:
            if not self.is_visible(
                drone_x=drone_x,
                drone_y=drone_y,
                drone_z=drone_z,
                drone_yaw=drone_yaw,
                target_x=victim.x,
                target_y=victim.y,
                target_height_m=victim.height_m,
            ):
                continue
            if rng.random() > probability:
                continue
            noise = max(0.0, float(self.config.detection_noise_m))
            observations.append(
                DetectionObservation(
                    x=float(victim.x + rng.normal(0.0, noise)),
                    y=float(victim.y + rng.normal(0.0, noise)),
                    confidence=float(rng.uniform(0.58, 0.96)),
                    class_name="person",
                    truth_id=victim.id,
                )
            )

        if self.config.false_positive_rate > 0.0 and visible_cells:
            if rng.random() < self.config.false_positive_rate:
                row, col = visible_cells[int(rng.integers(0, len(visible_cells)))]
                x, y = geometry.grid_to_world(row, col)
                observations.append(
                    DetectionObservation(
                        x=x,
                        y=y,
                        confidence=float(rng.uniform(0.35, 0.70)),
                        class_name="person",
                        truth_id=None,
                    )
                )
        return observations
