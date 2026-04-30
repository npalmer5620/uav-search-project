"""Detection-track memory and confirmation logic for V2 search."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from uav_rl.rl_common import wrap_angle_rad


@dataclass(frozen=True)
class DetectionObservation:
    class_name: str
    confidence: float
    x: float
    y: float
    z: float = 0.0
    truth_id: int | None = None


@dataclass
class DetectionTrack:
    track_id: int
    class_name: str
    first_seen_step: int
    last_seen_step: int
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    viewpoints: list[tuple[float, float, float]] = field(default_factory=list)
    truth_ids: list[int | None] = field(default_factory=list)
    investigated: bool = False
    confirmed: bool = False

    def add(
        self,
        obs: DetectionObservation,
        *,
        step: int,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        max_history: int,
    ) -> None:
        self.last_seen_step = int(step)
        self.positions.append((float(obs.x), float(obs.y), float(obs.z)))
        self.confidences.append(float(obs.confidence))
        self.viewpoints.append((float(drone_x), float(drone_y), float(drone_yaw)))
        self.truth_ids.append(obs.truth_id)
        if len(self.positions) > max_history:
            self.positions = self.positions[-max_history:]
            self.confidences = self.confidences[-max_history:]
            self.viewpoints = self.viewpoints[-max_history:]
            self.truth_ids = self.truth_ids[-max_history:]

    @property
    def hits(self) -> int:
        return len(self.confidences)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))

    @property
    def filtered_position(self) -> tuple[float, float, float]:
        if not self.positions:
            return (0.0, 0.0, 0.0)
        arr = np.asarray(self.positions, dtype=np.float32)
        return tuple(float(v) for v in np.median(arr, axis=0))

    @property
    def dominant_truth_id(self) -> int | None:
        values = [value for value in self.truth_ids if value is not None]
        if not values:
            return None
        return max(set(values), key=values.count)

    def age_steps(self, step: int) -> int:
        return max(0, int(step) - self.last_seen_step)

    def max_viewpoint_separation(self) -> float:
        if len(self.viewpoints) < 2:
            return 0.0
        best = 0.0
        for idx, (ax, ay, _ayaw) in enumerate(self.viewpoints):
            for bx, by, _byaw in self.viewpoints[idx + 1 :]:
                best = max(best, math.hypot(ax - bx, ay - by))
        return best

    def max_yaw_span(self) -> float:
        if len(self.viewpoints) < 2:
            return 0.0
        yaws = [view[2] for view in self.viewpoints]
        best = 0.0
        for idx, yaw_a in enumerate(yaws):
            for yaw_b in yaws[idx + 1 :]:
                best = max(best, abs(wrap_angle_rad(yaw_a - yaw_b)))
        return best

    def max_position_spread(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        best = 0.0
        for idx, (ax, ay, az) in enumerate(self.positions):
            for bx, by, bz in self.positions[idx + 1 :]:
                best = max(best, math.dist((ax, ay, az), (bx, by, bz)))
        return best


@dataclass(frozen=True)
class ConfirmationConfig:
    match_distance_m: float = 2.5
    max_history: int = 8
    track_timeout_steps: int = 20
    min_hits: int = 3
    min_mean_confidence: float = 0.65
    min_viewpoint_separation_m: float = 4.0
    min_yaw_span_rad: float = math.radians(25.0)
    max_position_spread_m: float = 2.0


class DetectionMemory:
    def __init__(self, config: ConfirmationConfig | None = None) -> None:
        self.config = config or ConfirmationConfig()
        self.tracks: list[DetectionTrack] = []
        self.next_track_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1

    def prune(self, step: int) -> None:
        timeout = max(1, int(self.config.track_timeout_steps))
        self.tracks = [
            track
            for track in self.tracks
            if track.confirmed or track.age_steps(step) <= timeout
        ]

    def update(
        self,
        observations: list[DetectionObservation],
        *,
        step: int,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
    ) -> tuple[int, int]:
        self.prune(step)
        new_tracks = 0
        updated_existing = 0
        for obs in observations:
            track = self._find_match(obs)
            if track is None:
                track = DetectionTrack(
                    track_id=self.next_track_id,
                    class_name=obs.class_name,
                    first_seen_step=step,
                    last_seen_step=step,
                )
                self.next_track_id += 1
                self.tracks.append(track)
                new_tracks += 1
            else:
                updated_existing += 1
            track.add(
                obs,
                step=step,
                drone_x=drone_x,
                drone_y=drone_y,
                drone_yaw=drone_yaw,
                max_history=self.config.max_history,
            )
        return new_tracks, updated_existing

    def _find_match(self, obs: DetectionObservation) -> DetectionTrack | None:
        best_track = None
        best_distance = float("inf")
        for track in self.tracks:
            if track.class_name != obs.class_name or track.confirmed:
                continue
            tx, ty, tz = track.filtered_position
            distance = math.dist((tx, ty, tz), (obs.x, obs.y, obs.z))
            if distance <= self.config.match_distance_m and distance < best_distance:
                best_track = track
                best_distance = distance
        return best_track

    def confirm_ready_tracks(self) -> list[DetectionTrack]:
        ready: list[DetectionTrack] = []
        for track in self.tracks:
            if track.confirmed:
                continue
            if track.hits < self.config.min_hits:
                continue
            if track.mean_confidence < self.config.min_mean_confidence:
                continue
            has_view_diversity = (
                track.max_viewpoint_separation() >= self.config.min_viewpoint_separation_m
                or track.max_yaw_span() >= self.config.min_yaw_span_rad
            )
            if not has_view_diversity:
                continue
            if track.max_position_spread() > self.config.max_position_spread_m:
                continue
            track.confirmed = True
            ready.append(track)
        return ready

    def best_track(self, *, include_confirmed: bool = False) -> DetectionTrack | None:
        candidates = [
            track
            for track in self.tracks
            if (include_confirmed or not track.confirmed) and not track.investigated
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda track: (
                track.mean_confidence,
                track.hits,
                track.last_seen_step,
            ),
        )

    def top_k_features(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        step: int,
        k: int = 3,
        max_range_m: float = 60.0,
        age_norm_steps: int = 100,
    ) -> np.ndarray:
        features = np.zeros((k, 7), dtype=np.float32)
        ranked = sorted(
            self.tracks,
            key=lambda track: (track.confirmed, -track.mean_confidence, -track.hits),
        )
        for idx, track in enumerate(ranked[:k]):
            tx, ty, _tz = track.filtered_position
            dx = tx - drone_x
            dy = ty - drone_y
            distance = math.hypot(dx, dy)
            bearing = wrap_angle_rad(math.atan2(dy, dx) - drone_yaw)
            features[idx] = np.array(
                [
                    np.clip(track.mean_confidence, 0.0, 1.0),
                    np.clip(distance / max(max_range_m, 1.0), 0.0, 1.0),
                    math.sin(bearing),
                    math.cos(bearing),
                    np.clip(track.age_steps(step) / max(float(age_norm_steps), 1.0), 0.0, 1.0),
                    1.0 if track.confirmed else 0.0,
                    1.0 if track.investigated else 0.0,
                ],
                dtype=np.float32,
            )
        return features

    def unconfirmed_count(self) -> int:
        return sum(1 for track in self.tracks if not track.confirmed)
