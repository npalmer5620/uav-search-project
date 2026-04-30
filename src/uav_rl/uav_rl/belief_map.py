"""Belief-map data structures for the V2 search policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CHANNEL_OBSERVED = 0
CHANNEL_OBSTACLE = 1
CHANNEL_VICTIM_SCORE = 2
CHANNEL_UNCERTAINTY = 3
CHANNEL_AGE = 4
CHANNEL_CONFIRMED = 5
BELIEF_CHANNEL_COUNT = 6


@dataclass(frozen=True)
class MapGeometry:
    width: float = 40.0
    height: float = 40.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    cell_size_m: float = 4.0

    @property
    def rows(self) -> int:
        return max(1, int(math.ceil(self.height / max(self.cell_size_m, 1e-6))))

    @property
    def cols(self) -> int:
        return max(1, int(math.ceil(self.width / max(self.cell_size_m, 1e-6))))

    @property
    def x_limits(self) -> tuple[float, float]:
        return (self.origin_x - self.height / 2.0, self.origin_x + self.height / 2.0)

    @property
    def y_limits(self) -> tuple[float, float]:
        return (self.origin_y - self.width / 2.0, self.origin_y + self.width / 2.0)

    @property
    def diagonal_m(self) -> float:
        return math.hypot(self.width, self.height)

    def in_bounds_xy(self, x: float, y: float) -> bool:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        return x_min <= x <= x_max and y_min <= y <= y_max

    def clip_xy(self, x: float, y: float) -> tuple[float, float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        clipped_x = min(max(float(x), x_min), x_max)
        clipped_y = min(max(float(y), y_min), y_max)
        return clipped_x, clipped_y, math.hypot(clipped_x - x, clipped_y - y)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        row = int(np.clip(((float(x) - x_min) / max(x_max - x_min, 1e-6)) * self.rows, 0, self.rows - 1))
        col = int(np.clip(((float(y) - y_min) / max(y_max - y_min, 1e-6)) * self.cols, 0, self.cols - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        row = int(np.clip(row, 0, self.rows - 1))
        col = int(np.clip(col, 0, self.cols - 1))
        x = x_min + (row + 0.5) * ((x_max - x_min) / self.rows)
        y = y_min + (col + 0.5) * ((y_max - y_min) / self.cols)
        return float(x), float(y)

    def iter_cell_centers(self):
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = self.grid_to_world(row, col)
                yield row, col, x, y


@dataclass(frozen=True)
class BeliefUpdate:
    new_observed_cells: int = 0
    uncertainty_reduction: float = 0.0
    revisited_cells: int = 0


class BeliefMap:
    """Grid memory used as the policy's search-state memory."""

    def __init__(self, geometry: MapGeometry, *, age_norm_s: float = 120.0) -> None:
        self.geometry = geometry
        self.age_norm_s = max(1.0, float(age_norm_s))
        self.grid = np.zeros(
            (geometry.rows, geometry.cols, BELIEF_CHANNEL_COUNT),
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> None:
        self.grid.fill(0.0)
        self.grid[:, :, CHANNEL_UNCERTAINTY] = 1.0

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.grid[:, :, CHANNEL_OBSERVED]))

    @property
    def mean_uncertainty(self) -> float:
        return float(np.mean(self.grid[:, :, CHANNEL_UNCERTAINTY]))

    def age(self, dt_s: float) -> None:
        observed = self.grid[:, :, CHANNEL_OBSERVED] > 0.0
        age_channel = self.grid[:, :, CHANNEL_AGE]
        age_channel[observed] += max(0.0, float(dt_s))

    def mark_visible(
        self,
        cells: list[tuple[int, int]],
        *,
        uncertainty_drop: float = 0.35,
    ) -> BeliefUpdate:
        new_observed = 0
        revisited = 0
        uncertainty_reduction = 0.0
        for row, col in cells:
            if not (0 <= row < self.geometry.rows and 0 <= col < self.geometry.cols):
                continue
            if self.grid[row, col, CHANNEL_OBSERVED] <= 0.0:
                new_observed += 1
            else:
                revisited += 1
            previous_uncertainty = float(self.grid[row, col, CHANNEL_UNCERTAINTY])
            next_uncertainty = max(0.0, previous_uncertainty - uncertainty_drop)
            uncertainty_reduction += previous_uncertainty - next_uncertainty
            self.grid[row, col, CHANNEL_OBSERVED] = 1.0
            self.grid[row, col, CHANNEL_UNCERTAINTY] = next_uncertainty
            self.grid[row, col, CHANNEL_AGE] = 0.0
        return BeliefUpdate(new_observed, uncertainty_reduction, revisited)

    def mark_obstacle(self, x: float, y: float, confidence: float = 1.0) -> None:
        if not self.geometry.in_bounds_xy(x, y):
            return
        row, col = self.geometry.world_to_grid(x, y)
        self.grid[row, col, CHANNEL_OBSTACLE] = max(
            float(self.grid[row, col, CHANNEL_OBSTACLE]),
            float(np.clip(confidence, 0.0, 1.0)),
        )
        self.grid[row, col, CHANNEL_OBSERVED] = 1.0

    def update_detection(self, x: float, y: float, confidence: float) -> tuple[int, int]:
        row, col = self.geometry.world_to_grid(x, y)
        conf = float(np.clip(confidence, 0.0, 1.0))
        prev = float(self.grid[row, col, CHANNEL_VICTIM_SCORE])
        self.grid[row, col, CHANNEL_VICTIM_SCORE] = max(prev, prev + conf * (1.0 - prev) * 0.6)
        self.grid[row, col, CHANNEL_UNCERTAINTY] = max(
            0.0,
            float(self.grid[row, col, CHANNEL_UNCERTAINTY]) - 0.25 * conf,
        )
        self.grid[row, col, CHANNEL_OBSERVED] = 1.0
        self.grid[row, col, CHANNEL_AGE] = 0.0
        return row, col

    def confirm_cell(self, x: float, y: float) -> None:
        row, col = self.geometry.world_to_grid(x, y)
        self.grid[row, col, CHANNEL_CONFIRMED] = 1.0
        self.grid[row, col, CHANNEL_VICTIM_SCORE] = 1.0
        self.grid[row, col, CHANNEL_UNCERTAINTY] = 0.0

    def local_patch(self, x: float, y: float, *, side: int) -> np.ndarray:
        side = max(1, int(side))
        if side % 2 == 0:
            side += 1
        center_row, center_col = self.geometry.world_to_grid(x, y)
        radius = side // 2
        patch = np.zeros((side, side, BELIEF_CHANNEL_COUNT), dtype=np.float32)
        patch[:, :, CHANNEL_UNCERTAINTY] = 1.0
        patch[:, :, CHANNEL_OBSTACLE] = 1.0

        for out_r in range(side):
            row = center_row + out_r - radius
            if row < 0 or row >= self.geometry.rows:
                continue
            for out_c in range(side):
                col = center_col + out_c - radius
                if col < 0 or col >= self.geometry.cols:
                    continue
                patch[out_r, out_c, :] = self.grid[row, col, :]

        patch[:, :, CHANNEL_AGE] = np.clip(
            patch[:, :, CHANNEL_AGE] / self.age_norm_s,
            0.0,
            1.0,
        )
        return patch

    def frontier_cells(self) -> list[tuple[int, int]]:
        observed = self.grid[:, :, CHANNEL_OBSERVED] > 0.0
        obstacle = self.grid[:, :, CHANNEL_OBSTACLE] > 0.5
        frontiers: list[tuple[int, int]] = []
        for row in range(self.geometry.rows):
            for col in range(self.geometry.cols):
                if observed[row, col] or obstacle[row, col]:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = row + dr
                    nc = col + dc
                    if 0 <= nr < self.geometry.rows and 0 <= nc < self.geometry.cols and observed[nr, nc]:
                        frontiers.append((row, col))
                        break
        return frontiers

    def nearest_frontier(self, x: float, y: float) -> tuple[float, float] | None:
        frontiers = self.frontier_cells()
        if not frontiers:
            return None
        return min(
            (self.geometry.grid_to_world(row, col) for row, col in frontiers),
            key=lambda pt: math.hypot(pt[0] - x, pt[1] - y),
        )

    def best_victim_cell(self, x: float, y: float) -> tuple[float, float] | None:
        victim = self.grid[:, :, CHANNEL_VICTIM_SCORE]
        confirmed = self.grid[:, :, CHANNEL_CONFIRMED] > 0.0
        best_score = -1.0
        best_cell: tuple[int, int] | None = None
        for row in range(self.geometry.rows):
            for col in range(self.geometry.cols):
                if confirmed[row, col]:
                    continue
                score = float(victim[row, col])
                if score <= 0.0:
                    continue
                cx, cy = self.geometry.grid_to_world(row, col)
                distance_penalty = 0.02 * math.hypot(cx - x, cy - y)
                ranked = score - distance_penalty
                if ranked > best_score:
                    best_score = ranked
                    best_cell = (row, col)
        if best_cell is None:
            return None
        return self.geometry.grid_to_world(*best_cell)
