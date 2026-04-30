"""Forward-facing RGBD camera geometry used by the V2 search policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from uav_rl.rl_common import wrap_angle_rad


@dataclass(frozen=True)
class ForwardCameraModel:
    """Simple level, forward-facing pinhole camera model.

    The PX4 ``gz_x500_depth`` model uses an OakD-style depth camera pointed
    forward. For a level camera, ground targets are only visible when their
    downward angle from the optical axis fits inside the lower half of the
    vertical FOV and their range fits inside the depth clip range.
    """

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

    @property
    def min_ground_range_m_at_4m(self) -> float:
        return self.min_visible_ground_range(-4.0)

    @property
    def min_ground_range_m_at_6m(self) -> float:
        return self.min_visible_ground_range(-6.0)

    @property
    def min_ground_range_m_at_10m(self) -> float:
        return self.min_visible_ground_range(-10.0)

    def min_visible_ground_range(self, altitude_ned_z: float) -> float:
        height = abs(float(altitude_ned_z))
        half_vfov = max(self.vertical_fov_rad / 2.0, 1e-6)
        return height / max(math.tan(half_vfov), 1e-6)

    def ground_visibility_band(self, altitude_ned_z: float) -> tuple[float, float]:
        min_range = max(self.depth_near_m, self.min_visible_ground_range(altitude_ned_z))
        return min_range, self.depth_far_m

    def is_ground_point_visible(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        altitude_ned_z: float,
        point_x: float,
        point_y: float,
        target_height_m: float | None = None,
    ) -> bool:
        dx = float(point_x) - float(drone_x)
        dy = float(point_y) - float(drone_y)
        horizontal_range = math.hypot(dx, dy)
        if horizontal_range < self.depth_near_m or horizontal_range > self.depth_far_m:
            return False

        bearing = wrap_angle_rad(math.atan2(dy, dx) - float(drone_yaw))
        if abs(bearing) > self.horizontal_fov_rad / 2.0:
            return False

        target_height = self.target_height_m if target_height_m is None else target_height_m
        vertical_drop = max(0.0, abs(float(altitude_ned_z)) - max(0.0, target_height))
        down_angle = math.atan2(vertical_drop, max(horizontal_range, 1e-6))
        return down_angle <= self.vertical_fov_rad / 2.0

    def visible_cells(
        self,
        *,
        geometry,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        altitude_ned_z: float,
    ) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for row, col, cx, cy in geometry.iter_cell_centers():
            if self.is_ground_point_visible(
                drone_x=drone_x,
                drone_y=drone_y,
                drone_yaw=drone_yaw,
                altitude_ned_z=altitude_ned_z,
                point_x=cx,
                point_y=cy,
            ):
                cells.append((row, col))
        return cells

    def visible_points(
        self,
        *,
        drone_x: float,
        drone_y: float,
        drone_yaw: float,
        altitude_ned_z: float,
        points: Iterable[tuple[float, float]],
        target_height_m: float | None = None,
    ) -> list[int]:
        visible: list[int] = []
        for idx, (point_x, point_y) in enumerate(points):
            if self.is_ground_point_visible(
                drone_x=drone_x,
                drone_y=drone_y,
                drone_yaw=drone_yaw,
                altitude_ned_z=altitude_ned_z,
                point_x=point_x,
                point_y=point_y,
                target_height_m=target_height_m,
            ):
                visible.append(idx)
        return visible
