"""Macro-actions and heuristic goal selection for triage search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any

import numpy as np

from uav_triage_rl.perception import DetectionTrack, wrap_angle


CHANNEL_OBSERVED = 0
CHANNEL_VICTIM_SCORE = 1
CHANNEL_OBSTACLE = 2
CHANNEL_UNCERTAINTY = 3
BELIEF_CHANNELS = 4


class SearchAction(IntEnum):
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


ACTION_NAMES = {
    SearchAction.FRONTIER_BEST: "frontier_best",
    SearchAction.FRONTIER_SECOND: "frontier_second",
    SearchAction.FRONTIER_THIRD: "frontier_third",
    SearchAction.INVESTIGATE_BEST: "investigate_best",
    SearchAction.INVESTIGATE_OFFSET: "investigate_offset",
    SearchAction.HIGH_INFO_BEST: "high_info_best",
    SearchAction.HIGH_INFO_SECOND: "high_info_second",
    SearchAction.RETURN_CENTER: "return_center",
    SearchAction.ESCAPE_STUCK: "escape_stuck",
    SearchAction.HOVER_SCAN: "hover_scan",
}


@dataclass(frozen=True)
class ActionConfig:
    scan_yaw_step_deg: float = 35.0
    investigate_standoff_m: float = 7.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ActionConfig":
        return cls(**dict(raw or {}))

    @property
    def scan_yaw_step_rad(self) -> float:
        return math.radians(self.scan_yaw_step_deg)


@dataclass(frozen=True)
class Goal:
    x: float
    y: float
    yaw: float
    name: str
    overflow_m: float = 0.0
    target_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class MapGeometry:
    width_m: float
    height_m: float
    cell_size_m: float

    @property
    def rows(self) -> int:
        return max(1, int(math.ceil(self.height_m / self.cell_size_m)))

    @property
    def cols(self) -> int:
        return max(1, int(math.ceil(self.width_m / self.cell_size_m)))

    @property
    def x_limits(self) -> tuple[float, float]:
        return (-self.width_m / 2.0, self.width_m / 2.0)

    @property
    def y_limits(self) -> tuple[float, float]:
        return (-self.height_m / 2.0, self.height_m / 2.0)

    def clip_xy(self, x: float, y: float) -> tuple[float, float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        clipped_x = min(max(float(x), x_min), x_max)
        clipped_y = min(max(float(y), y_min), y_max)
        return clipped_x, clipped_y, math.hypot(clipped_x - x, clipped_y - y)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        col = int(np.clip(((float(x) - x_min) / max(x_max - x_min, 1e-6)) * self.cols, 0, self.cols - 1))
        row = int(np.clip(((float(y) - y_min) / max(y_max - y_min, 1e-6)) * self.rows, 0, self.rows - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        x = x_min + (float(col) + 0.5) * (x_max - x_min) / self.cols
        y = y_min + (float(row) + 0.5) * (y_max - y_min) / self.rows
        return x, y

    def iter_cell_centers(self):
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = self.grid_to_world(row, col)
                yield row, col, x, y


def ranked_frontiers(
    *,
    geometry: MapGeometry,
    belief: np.ndarray,
    x: float,
    y: float,
    high_info: bool = False,
) -> list[tuple[float, int, int]]:
    ranked: list[tuple[float, int, int]] = []
    for row in range(geometry.rows):
        for col in range(geometry.cols):
            cell = belief[row, col]
            if cell[CHANNEL_OBSTACLE] > 0.5:
                continue
            wx, wy = geometry.grid_to_world(row, col)
            distance = math.hypot(wx - x, wy - y)
            unobserved = 1.0 if cell[CHANNEL_OBSERVED] < 0.5 else 0.0
            score = (
                2.0 * unobserved
                + 0.7 * float(cell[CHANNEL_UNCERTAINTY])
                + 2.5 * float(cell[CHANNEL_VICTIM_SCORE])
                - 0.035 * distance
            )
            if high_info:
                score += 0.5 * float(cell[CHANNEL_UNCERTAINTY])
            ranked.append((score, row, col))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return ranked


def _goal_for_cell(
    *,
    geometry: MapGeometry,
    row: int,
    col: int,
    x: float,
    y: float,
    name: str,
) -> Goal:
    gx, gy = geometry.grid_to_world(row, col)
    gx, gy, overflow = geometry.clip_xy(gx, gy)
    yaw = math.atan2(gy - y, gx - x)
    return Goal(x=gx, y=gy, yaw=yaw, name=name, overflow_m=overflow)


def _goal_for_track(
    *,
    geometry: MapGeometry,
    track: DetectionTrack,
    x: float,
    y: float,
    current_yaw: float,
    config: ActionConfig,
    offset: bool,
) -> Goal:
    from_target = math.atan2(y - track.y, x - track.x)
    if not math.isfinite(from_target) or math.hypot(x - track.x, y - track.y) < 1e-6:
        from_target = wrap_angle(current_yaw + math.pi)
    if offset:
        from_target = wrap_angle(from_target + math.radians(55.0))
    gx = track.x + config.investigate_standoff_m * math.cos(from_target)
    gy = track.y + config.investigate_standoff_m * math.sin(from_target)
    gx, gy, overflow = geometry.clip_xy(gx, gy)
    yaw = math.atan2(track.y - gy, track.x - gx)
    return Goal(
        x=gx,
        y=gy,
        yaw=yaw,
        name="investigate_offset" if offset else "investigate_best",
        overflow_m=overflow,
        target_xy=(track.x, track.y),
    )


def select_goal(
    *,
    action: int,
    geometry: MapGeometry,
    belief: np.ndarray,
    tracks: list[DetectionTrack],
    x: float,
    y: float,
    yaw: float,
    config: ActionConfig,
) -> Goal:
    selected = SearchAction(int(action))
    best_track = next((track for track in tracks if not track.confirmed), None)

    if selected in (SearchAction.INVESTIGATE_BEST, SearchAction.INVESTIGATE_OFFSET) and best_track is not None:
        return _goal_for_track(
            geometry=geometry,
            track=best_track,
            x=x,
            y=y,
            current_yaw=yaw,
            config=config,
            offset=selected == SearchAction.INVESTIGATE_OFFSET,
        )
    if selected in (SearchAction.INVESTIGATE_BEST, SearchAction.INVESTIGATE_OFFSET):
        selected = SearchAction.FRONTIER_BEST

    if selected == SearchAction.RETURN_CENTER:
        gx, gy, overflow = geometry.clip_xy(0.0, 0.0)
        return Goal(x=gx, y=gy, yaw=math.atan2(-y, -x), name="return_center", overflow_m=overflow)

    if selected == SearchAction.ESCAPE_STUCK:
        ranked = ranked_frontiers(geometry=geometry, belief=belief, x=x, y=y)
        if ranked:
            # Choose a good cell that is not the nearest best cell, encouraging escape.
            idx = min(len(ranked) - 1, 5)
            _, row, col = ranked[idx]
            return _goal_for_cell(geometry=geometry, row=row, col=col, x=x, y=y, name="escape_stuck")
        gx, gy, overflow = geometry.clip_xy(-x, -y)
        return Goal(x=gx, y=gy, yaw=math.atan2(gy - y, gx - x), name="escape_stuck", overflow_m=overflow)

    if selected == SearchAction.HOVER_SCAN:
        return Goal(
            x=x,
            y=y,
            yaw=wrap_angle(yaw + config.scan_yaw_step_rad),
            name="hover_scan",
        )

    high_info = selected in (SearchAction.HIGH_INFO_BEST, SearchAction.HIGH_INFO_SECOND)
    ranked = ranked_frontiers(geometry=geometry, belief=belief, x=x, y=y, high_info=high_info)
    if ranked:
        rank = {
            SearchAction.FRONTIER_BEST: 0,
            SearchAction.FRONTIER_SECOND: 1,
            SearchAction.FRONTIER_THIRD: 2,
            SearchAction.HIGH_INFO_BEST: 0,
            SearchAction.HIGH_INFO_SECOND: 1,
        }.get(selected, 0)
        _, row, col = ranked[min(rank, len(ranked) - 1)]
        return _goal_for_cell(
            geometry=geometry,
            row=row,
            col=col,
            x=x,
            y=y,
            name=ACTION_NAMES.get(selected, "frontier_best"),
        )

    return Goal(x=x, y=y, yaw=wrap_angle(yaw + config.scan_yaw_step_rad), name="hover_scan")


def lawnmower_action(step_index: int) -> int:
    if step_index % 7 == 6:
        return int(SearchAction.HOVER_SCAN)
    if step_index % 3 == 1:
        return int(SearchAction.FRONTIER_SECOND)
    return int(SearchAction.FRONTIER_BEST)
