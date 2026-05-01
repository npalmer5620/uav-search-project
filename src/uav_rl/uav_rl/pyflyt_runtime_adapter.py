"""Gazebo runtime adapter for policies trained in ``pyflyt_rl``.

The standalone PyFlyt stack uses a leaner observation contract than the
existing Gazebo V2 task: a 4-channel belief patch, 9 state scalars, a
10-action one-hot, and 3 compact detection tracks. This module mirrors that
contract without importing PyFlyt, so the ROS/Gazebo controller can run a
saved PyFlyt DQN policy inside the Docker SITL stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
import math
from pathlib import Path
from typing import Any

import numpy as np

from uav_rl.detection_memory import DetectionMemory, DetectionTrack


PYFLYT_CHANNEL_OBSERVED = 0
PYFLYT_CHANNEL_VICTIM_SCORE = 1
PYFLYT_CHANNEL_OBSTACLE = 2
PYFLYT_CHANNEL_UNCERTAINTY = 3
PYFLYT_BELIEF_CHANNELS = 4


class PyFlytSearchAction(IntEnum):
    FRONTIER_BEST = 0
    FRONTIER_SECOND = 1
    FRONTIER_THIRD = 2
    INVESTIGATE_BEST = 3
    INVESTIGATE_OFFSET = 4
    HIGH_INFO_BEST = 5
    HIGH_INFO_SECOND = 6
    RETURN_CENTER = 7
    ESCAPE_STUCK = 8
    HOVER_SCAN = 9


PYFLYT_ACTION_NAMES = {
    PyFlytSearchAction.FRONTIER_BEST: "frontier_best",
    PyFlytSearchAction.FRONTIER_SECOND: "frontier_second",
    PyFlytSearchAction.FRONTIER_THIRD: "frontier_third",
    PyFlytSearchAction.INVESTIGATE_BEST: "investigate_best",
    PyFlytSearchAction.INVESTIGATE_OFFSET: "investigate_offset",
    PyFlytSearchAction.HIGH_INFO_BEST: "high_info_best",
    PyFlytSearchAction.HIGH_INFO_SECOND: "high_info_second",
    PyFlytSearchAction.RETURN_CENTER: "return_center",
    PyFlytSearchAction.ESCAPE_STUCK: "escape_stuck",
    PyFlytSearchAction.HOVER_SCAN: "hover_scan",
}


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class PyFlytCameraConfig:
    image_width_px: int = 320
    image_height_px: int = 240
    horizontal_fov_deg: float = 90.0
    camera_pitch_deg: float = 55.0
    min_range_m: float = 0.5
    max_range_m: float = 22.0
    target_height_m: float = 0.25

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "PyFlytCameraConfig":
        if raw is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in dict(raw).items() if key in allowed})

    @property
    def horizontal_fov_rad(self) -> float:
        return math.radians(self.horizontal_fov_deg)

    @property
    def vertical_fov_rad(self) -> float:
        aspect = self.image_height_px / max(float(self.image_width_px), 1.0)
        return 2.0 * math.atan(math.tan(self.horizontal_fov_rad / 2.0) * aspect)

    def is_visible(
        self,
        *,
        drone_x: float,
        drone_y: float,
        altitude_m: float,
        drone_yaw: float,
        target_x: float,
        target_y: float,
        target_height_m: float | None = None,
    ) -> bool:
        dx = float(target_x) - float(drone_x)
        dy = float(target_y) - float(drone_y)
        horizontal_range = math.hypot(dx, dy)
        if horizontal_range < self.min_range_m or horizontal_range > self.max_range_m:
            return False

        bearing = wrap_angle(math.atan2(dy, dx) - float(drone_yaw))
        if abs(bearing) > self.horizontal_fov_rad / 2.0:
            return False

        target_height = self.target_height_m if target_height_m is None else target_height_m
        vertical_drop = max(0.0, float(altitude_m) - max(0.0, target_height))
        down_angle = math.atan2(vertical_drop, max(horizontal_range, 1e-6))
        camera_pitch = math.radians(self.camera_pitch_deg)
        return abs(down_angle - camera_pitch) <= self.vertical_fov_rad / 2.0

    def visible_cells(
        self,
        *,
        geometry: "PyFlytMapGeometry",
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        altitude_ned_z: float,
    ) -> list[tuple[int, int]]:
        altitude_m = abs(float(altitude_ned_z))
        cells: list[tuple[int, int]] = []
        for row, col, cx, cy in geometry.iter_cell_centers():
            if self.is_visible(
                drone_x=drone_x,
                drone_y=drone_y,
                altitude_m=altitude_m,
                drone_yaw=drone_yaw,
                target_x=cx,
                target_y=cy,
            ):
                cells.append((row, col))
        return cells


@dataclass(frozen=True)
class PyFlytMapGeometry:
    width_m: float = 40.0
    height_m: float = 40.0
    cell_size_m: float = 2.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    @property
    def rows(self) -> int:
        return max(1, int(math.ceil(self.height_m / max(self.cell_size_m, 1e-6))))

    @property
    def cols(self) -> int:
        return max(1, int(math.ceil(self.width_m / max(self.cell_size_m, 1e-6))))

    @property
    def x_limits(self) -> tuple[float, float]:
        return (self.origin_x - self.width_m / 2.0, self.origin_x + self.width_m / 2.0)

    @property
    def y_limits(self) -> tuple[float, float]:
        return (self.origin_y - self.height_m / 2.0, self.origin_y + self.height_m / 2.0)

    def clip_xy(self, x: float, y: float) -> tuple[float, float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        clipped_x = min(max(float(x), x_min), x_max)
        clipped_y = min(max(float(y), y_min), y_max)
        return clipped_x, clipped_y, math.hypot(clipped_x - x, clipped_y - y)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        col = int(
            np.clip(
                ((float(x) - x_min) / max(x_max - x_min, 1e-6)) * self.cols,
                0,
                self.cols - 1,
            )
        )
        row = int(
            np.clip(
                ((float(y) - y_min) / max(y_max - y_min, 1e-6)) * self.rows,
                0,
                self.rows - 1,
            )
        )
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        row = int(np.clip(row, 0, self.rows - 1))
        col = int(np.clip(col, 0, self.cols - 1))
        x = x_min + (float(col) + 0.5) * (x_max - x_min) / self.cols
        y = y_min + (float(row) + 0.5) * (y_max - y_min) / self.rows
        return float(x), float(y)

    def iter_cell_centers(self):
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = self.grid_to_world(row, col)
                yield row, col, x, y


@dataclass
class PyFlytBeliefUpdate:
    new_observed_cells: int = 0
    uncertainty_reduction: float = 0.0


class PyFlytBeliefMap:
    def __init__(self, geometry: PyFlytMapGeometry) -> None:
        self.geometry = geometry
        self.grid = np.zeros(
            (geometry.rows, geometry.cols, PYFLYT_BELIEF_CHANNELS),
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> None:
        self.grid.fill(0.0)
        self.grid[:, :, PYFLYT_CHANNEL_UNCERTAINTY] = 1.0

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.grid[:, :, PYFLYT_CHANNEL_OBSERVED]))

    def mark_visible(
        self,
        cells: list[tuple[int, int]],
        *,
        uncertainty_drop: float = 0.35,
    ) -> PyFlytBeliefUpdate:
        new_cells = 0
        uncertainty_reduction = 0.0
        for row, col in cells:
            if not (0 <= row < self.geometry.rows and 0 <= col < self.geometry.cols):
                continue
            old_observed = float(self.grid[row, col, PYFLYT_CHANNEL_OBSERVED])
            old_uncertainty = float(self.grid[row, col, PYFLYT_CHANNEL_UNCERTAINTY])
            if old_observed <= 0.0:
                new_cells += 1
            next_uncertainty = max(0.0, old_uncertainty - uncertainty_drop)
            self.grid[row, col, PYFLYT_CHANNEL_OBSERVED] = 1.0
            self.grid[row, col, PYFLYT_CHANNEL_UNCERTAINTY] = next_uncertainty
            uncertainty_reduction += max(0.0, old_uncertainty - next_uncertainty)
        return PyFlytBeliefUpdate(new_cells, uncertainty_reduction)

    def mark_obstacle(self, x: float, y: float, confidence: float = 1.0) -> None:
        row, col = self.geometry.world_to_grid(x, y)
        self.grid[row, col, PYFLYT_CHANNEL_OBSTACLE] = max(
            float(self.grid[row, col, PYFLYT_CHANNEL_OBSTACLE]),
            float(np.clip(confidence, 0.0, 1.0)),
        )
        self.grid[row, col, PYFLYT_CHANNEL_OBSERVED] = 1.0

    def update_detection(self, x: float, y: float, confidence: float) -> None:
        row, col = self.geometry.world_to_grid(x, y)
        self.grid[row, col, PYFLYT_CHANNEL_VICTIM_SCORE] = min(
            1.0,
            float(self.grid[row, col, PYFLYT_CHANNEL_VICTIM_SCORE])
            + float(np.clip(confidence, 0.0, 1.0)) * 0.25,
        )
        self.grid[row, col, PYFLYT_CHANNEL_OBSERVED] = 1.0
        self.grid[row, col, PYFLYT_CHANNEL_UNCERTAINTY] = max(
            0.0,
            float(self.grid[row, col, PYFLYT_CHANNEL_UNCERTAINTY]) - 0.25 * confidence,
        )

    def confirm_cell(self, x: float, y: float) -> None:
        row, col = self.geometry.world_to_grid(x, y)
        self.grid[row, col, PYFLYT_CHANNEL_VICTIM_SCORE] = 1.0
        self.grid[row, col, PYFLYT_CHANNEL_UNCERTAINTY] = 0.0

    def local_patch(self, x: float, y: float, *, side: int) -> np.ndarray:
        side = max(1, int(side))
        if side % 2 == 0:
            side += 1
        row, col = self.geometry.world_to_grid(x, y)
        radius = side // 2
        padded = np.pad(
            self.grid,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        row += radius
        col += radius
        return padded[row - radius : row + radius + 1, col - radius : col + radius + 1, :]


@dataclass(frozen=True)
class PyFlytActionConfig:
    scan_yaw_step_deg: float = 35.0
    investigate_standoff_m: float = 7.0

    @property
    def scan_yaw_step_rad(self) -> float:
        return math.radians(self.scan_yaw_step_deg)


@dataclass(frozen=True)
class PyFlytGoal:
    x: float
    y: float
    yaw: float
    name: str
    overflow_m: float = 0.0
    target_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class PyFlytAdapterConfig:
    width_m: float = 40.0
    height_m: float = 40.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    cell_size_m: float = 2.0
    search_altitude_m: float = 6.0
    max_speed_m_s: float = 3.0
    decision_period_s: float = 1.5
    max_episode_steps: int = 220
    patch_side: int = 11
    top_k_tracks: int = 3
    match_distance_m: float = 2.5
    min_hits: int = 2
    track_timeout_steps: int = 18
    min_mean_confidence: float = 0.55
    action: PyFlytActionConfig = field(default_factory=PyFlytActionConfig)
    camera: PyFlytCameraConfig = field(default_factory=PyFlytCameraConfig)

    @property
    def observation_size(self) -> int:
        patch_side = self.patch_side + (1 if self.patch_side % 2 == 0 else 0)
        return (
            patch_side * patch_side * PYFLYT_BELIEF_CHANNELS
            + 9
            + len(PyFlytSearchAction)
            + self.top_k_tracks * 5
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        fallback: "PyFlytAdapterConfig | None" = None,
    ) -> "PyFlytAdapterConfig":
        base = fallback or cls()
        if raw is None:
            return base
        data = dict(raw)
        env = dict(data.get("env", {}) or {})
        camera = dict(data.get("camera", {}) or {})
        actions = dict(data.get("actions", {}) or {})
        perception = dict(data.get("perception", {}) or {})

        return cls(
            width_m=float(env.get("width_m", base.width_m)),
            height_m=float(env.get("height_m", base.height_m)),
            origin_x=float(env.get("origin_x", base.origin_x)),
            origin_y=float(env.get("origin_y", base.origin_y)),
            cell_size_m=float(env.get("cell_size_m", base.cell_size_m)),
            search_altitude_m=float(env.get("search_altitude_m", base.search_altitude_m)),
            max_speed_m_s=float(env.get("max_speed_m_s", base.max_speed_m_s)),
            decision_period_s=float(env.get("decision_period_s", base.decision_period_s)),
            max_episode_steps=int(env.get("max_episode_steps", base.max_episode_steps)),
            patch_side=int(env.get("patch_side", base.patch_side)),
            top_k_tracks=int(env.get("top_k_tracks", base.top_k_tracks)),
            match_distance_m=float(
                perception.get("match_distance_m", base.match_distance_m)
            ),
            min_hits=int(perception.get("min_hits", base.min_hits)),
            track_timeout_steps=int(
                perception.get("track_timeout_steps", base.track_timeout_steps)
            ),
            min_mean_confidence=float(
                perception.get("min_mean_confidence", base.min_mean_confidence)
            ),
            action=PyFlytActionConfig(
                scan_yaw_step_deg=float(
                    actions.get("scan_yaw_step_deg", base.action.scan_yaw_step_deg)
                ),
                investigate_standoff_m=float(
                    actions.get(
                        "investigate_standoff_m",
                        base.action.investigate_standoff_m,
                    )
                ),
            ),
            camera=PyFlytCameraConfig.from_mapping(camera or base.camera.__dict__),
        )

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        fallback: "PyFlytAdapterConfig | None" = None,
    ) -> "PyFlytAdapterConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("PyYAML is required to load PyFlyt policy config") from exc
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Expected mapping in PyFlyt config: {path}")
        return cls.from_mapping(raw, fallback=fallback)


def load_pyflyt_adapter_config(
    *,
    model_path: Path,
    artifact_dir: Path,
    config_path: Path | None,
    fallback: PyFlytAdapterConfig,
) -> tuple[PyFlytAdapterConfig, Path | None]:
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(config_path)
    candidates.extend([artifact_dir / "config.yaml", model_path.parent / "config.yaml"])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return PyFlytAdapterConfig.from_yaml(resolved, fallback=fallback), resolved
    return fallback, None


class PyFlytRuntimeAdapter:
    def __init__(self, config: PyFlytAdapterConfig) -> None:
        self.config = config
        self.geometry = PyFlytMapGeometry(
            width_m=config.width_m,
            height_m=config.height_m,
            cell_size_m=config.cell_size_m,
            origin_x=config.origin_x,
            origin_y=config.origin_y,
        )
        self.belief = PyFlytBeliefMap(self.geometry)
        self.last_action = -1
        self.last_action_name = ""
        self.last_visible_new_cells = 0

    @property
    def observation_size(self) -> int:
        return self.config.observation_size

    @property
    def coverage_fraction(self) -> float:
        return self.belief.coverage_fraction

    def reset(self) -> None:
        self.belief.reset()
        self.last_action = -1
        self.last_action_name = ""
        self.last_visible_new_cells = 0

    def mark_view(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        altitude_ned_z: float,
    ) -> PyFlytBeliefUpdate:
        cells = self.config.camera.visible_cells(
            geometry=self.geometry,
            drone_x=drone_x,
            drone_y=drone_y,
            drone_yaw=drone_yaw,
            altitude_ned_z=altitude_ned_z,
        )
        update = self.belief.mark_visible(cells)
        self.last_visible_new_cells = update.new_observed_cells
        return update

    def mark_detection(self, x: float, y: float, confidence: float) -> None:
        self.belief.update_detection(x, y, confidence)

    def mark_confirmed(self, x: float, y: float) -> None:
        self.belief.confirm_cell(x, y)

    def mark_obstacle(self, x: float, y: float, confidence: float = 1.0) -> None:
        self.belief.mark_obstacle(x, y, confidence)

    def encode_observation(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        altitude_ned_z: float,
        vx: float,
        vy: float,
        step: int,
        memory: DetectionMemory,
    ) -> np.ndarray:
        half_w = max(self.config.width_m / 2.0, 1e-6)
        half_h = max(self.config.height_m / 2.0, 1e-6)
        speed_ref = max(self.config.max_speed_m_s, 1.0)
        coverage = self.belief.coverage_fraction
        state = np.array(
            [
                (float(x) - self.config.origin_x) / half_w,
                (float(y) - self.config.origin_y) / half_h,
                abs(float(altitude_ned_z)) / max(self.config.search_altitude_m, 1.0),
                math.cos(float(yaw)),
                math.sin(float(yaw)),
                float(vx) / speed_ref,
                float(vy) / speed_ref,
                coverage,
                float(step) / max(1, self.config.max_episode_steps),
            ],
            dtype=np.float32,
        )

        last_action = np.zeros(len(PyFlytSearchAction), dtype=np.float32)
        if 0 <= self.last_action < len(last_action):
            last_action[self.last_action] = 1.0

        track_features: list[float] = []
        for track in self._top_tracks(memory):
            tx, ty, _tz = track.filtered_position
            track_features.extend(
                [
                    (tx - float(x)) / max(self.config.width_m, 1.0),
                    (ty - float(y)) / max(self.config.height_m, 1.0),
                    track.mean_confidence,
                    min(
                        1.0,
                        track.age_steps(step) / max(1, self.config.track_timeout_steps),
                    ),
                    1.0 if track.confirmed else 0.0,
                ]
            )
        while len(track_features) < self.config.top_k_tracks * 5:
            track_features.append(0.0)

        obs = np.concatenate(
            [
                self.belief.local_patch(x, y, side=self.config.patch_side).reshape(-1),
                state,
                last_action,
                np.asarray(track_features, dtype=np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != (self.observation_size,):
            raise RuntimeError(
                f"PyFlyt adapter produced observation shape {obs.shape}, "
                f"expected {(self.observation_size,)}"
            )
        return obs

    def goal_for_action(
        self,
        *,
        action: int,
        x: float,
        y: float,
        yaw: float,
        memory: DetectionMemory,
    ) -> PyFlytGoal:
        selected = PyFlytSearchAction(int(action))
        self.last_action = int(selected)
        best_track = memory.best_track()

        if selected in (
            PyFlytSearchAction.INVESTIGATE_BEST,
            PyFlytSearchAction.INVESTIGATE_OFFSET,
        ) and best_track is not None:
            goal = self._goal_for_track(
                track=best_track,
                x=x,
                y=y,
                current_yaw=yaw,
                offset=selected == PyFlytSearchAction.INVESTIGATE_OFFSET,
            )
            self.last_action_name = goal.name
            return goal
        if selected in (
            PyFlytSearchAction.INVESTIGATE_BEST,
            PyFlytSearchAction.INVESTIGATE_OFFSET,
        ):
            selected = PyFlytSearchAction.FRONTIER_BEST

        if selected == PyFlytSearchAction.RETURN_CENTER:
            gx, gy, overflow = self.geometry.clip_xy(
                self.config.origin_x,
                self.config.origin_y,
            )
            goal = PyFlytGoal(
                x=gx,
                y=gy,
                yaw=math.atan2(self.config.origin_y - y, self.config.origin_x - x),
                name=PYFLYT_ACTION_NAMES[selected],
                overflow_m=overflow,
            )
            self.last_action_name = goal.name
            return goal

        if selected == PyFlytSearchAction.ESCAPE_STUCK:
            ranked = self._ranked_frontiers(x=x, y=y)
            if ranked:
                _score, row, col = ranked[min(len(ranked) - 1, 5)]
                goal = self._goal_for_cell(row=row, col=col, x=x, y=y, name="escape_stuck")
            else:
                gx, gy, overflow = self.geometry.clip_xy(
                    self.config.origin_x - (x - self.config.origin_x),
                    self.config.origin_y - (y - self.config.origin_y),
                )
                goal = PyFlytGoal(
                    x=gx,
                    y=gy,
                    yaw=math.atan2(gy - y, gx - x),
                    name="escape_stuck",
                    overflow_m=overflow,
                )
            self.last_action_name = goal.name
            return goal

        if selected == PyFlytSearchAction.HOVER_SCAN:
            goal = PyFlytGoal(
                x=float(x),
                y=float(y),
                yaw=wrap_angle(float(yaw) + self.config.action.scan_yaw_step_rad),
                name="hover_scan",
            )
            self.last_action_name = goal.name
            return goal

        high_info = selected in (
            PyFlytSearchAction.HIGH_INFO_BEST,
            PyFlytSearchAction.HIGH_INFO_SECOND,
        )
        ranked = self._ranked_frontiers(x=x, y=y, high_info=high_info)
        if ranked:
            rank = {
                PyFlytSearchAction.FRONTIER_BEST: 0,
                PyFlytSearchAction.FRONTIER_SECOND: 1,
                PyFlytSearchAction.FRONTIER_THIRD: 2,
                PyFlytSearchAction.HIGH_INFO_BEST: 0,
                PyFlytSearchAction.HIGH_INFO_SECOND: 1,
            }.get(selected, 0)
            _score, row, col = ranked[min(rank, len(ranked) - 1)]
            goal = self._goal_for_cell(
                row=row,
                col=col,
                x=x,
                y=y,
                name=PYFLYT_ACTION_NAMES[selected],
            )
            self.last_action_name = goal.name
            return goal

        goal = PyFlytGoal(x=float(x), y=float(y), yaw=float(yaw), name="hover_scan")
        self.last_action_name = goal.name
        return goal

    def _top_tracks(self, memory: DetectionMemory) -> list[DetectionTrack]:
        return sorted(
            memory.tracks,
            key=lambda track: (
                track.confirmed,
                track.mean_confidence,
                track.hits,
                track.last_seen_step,
            ),
            reverse=True,
        )[: self.config.top_k_tracks]

    def _ranked_frontiers(
        self,
        *,
        x: float,
        y: float,
        high_info: bool = False,
    ) -> list[tuple[float, int, int]]:
        ranked: list[tuple[float, int, int]] = []
        for row in range(self.geometry.rows):
            for col in range(self.geometry.cols):
                cell = self.belief.grid[row, col]
                if cell[PYFLYT_CHANNEL_OBSTACLE] > 0.5:
                    continue
                wx, wy = self.geometry.grid_to_world(row, col)
                distance = math.hypot(wx - x, wy - y)
                unobserved = 1.0 if cell[PYFLYT_CHANNEL_OBSERVED] < 0.5 else 0.0
                score = (
                    2.0 * unobserved
                    + 0.7 * float(cell[PYFLYT_CHANNEL_UNCERTAINTY])
                    + 2.5 * float(cell[PYFLYT_CHANNEL_VICTIM_SCORE])
                    - 0.035 * distance
                )
                if high_info:
                    score += 0.5 * float(cell[PYFLYT_CHANNEL_UNCERTAINTY])
                ranked.append((score, row, col))
        ranked.sort(reverse=True, key=lambda item: item[0])
        return ranked

    def _goal_for_cell(
        self,
        *,
        row: int,
        col: int,
        x: float,
        y: float,
        name: str,
    ) -> PyFlytGoal:
        gx, gy = self.geometry.grid_to_world(row, col)
        gx, gy, overflow = self.geometry.clip_xy(gx, gy)
        return PyFlytGoal(
            x=gx,
            y=gy,
            yaw=math.atan2(gy - y, gx - x),
            name=name,
            overflow_m=overflow,
        )

    def _goal_for_track(
        self,
        *,
        track: DetectionTrack,
        x: float,
        y: float,
        current_yaw: float,
        offset: bool,
    ) -> PyFlytGoal:
        tx, ty, _tz = track.filtered_position
        from_target = math.atan2(y - ty, x - tx)
        if not math.isfinite(from_target) or math.hypot(x - tx, y - ty) < 1e-6:
            from_target = wrap_angle(current_yaw + math.pi)
        if offset:
            from_target = wrap_angle(from_target + math.radians(55.0))
        gx = tx + self.config.action.investigate_standoff_m * math.cos(from_target)
        gy = ty + self.config.action.investigate_standoff_m * math.sin(from_target)
        gx, gy, overflow = self.geometry.clip_xy(gx, gy)
        return PyFlytGoal(
            x=gx,
            y=gy,
            yaw=math.atan2(ty - gy, tx - gx),
            name="investigate_offset" if offset else "investigate_best",
            overflow_m=overflow,
            target_xy=(tx, ty),
        )
