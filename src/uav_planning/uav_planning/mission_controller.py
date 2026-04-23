#!/usr/bin/env python3
"""Grid-search mission controller."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException

from uav_planning.grid_generator import GridGenerator
from uav_planning.mission_base import MissionControllerBase, Phase


class GridMissionController(MissionControllerBase):
    """Mission controller that uses the existing boustrophedon search path."""

    def __init__(self) -> None:
        super().__init__("mission_controller")
        self.get_logger().info("Grid search policy enabled")

    def _init_search_controller(self) -> None:
        self.grid = GridGenerator(
            width=self.search_width,
            height=self.search_height,
            spacing=self.search_spacing,
            speed=self.search_speed,
            altitude=self.search_altitude,
            origin=(self.search_origin_x, self.search_origin_y),
        )

    def _search_state_detail(self) -> str | None:
        return f"{self.grid.progress:.0%}"

    def _search_step(self) -> None:
        current = self.grid.current_position
        if current is None:
            self.phase = Phase.RTH
            return

        cx, cy, _cz, _cyaw = current
        dist_to_target = math.hypot(
            self.local_pos.x - cx, self.local_pos.y - cy
        )
        if dist_to_target < self.grid.spacing:
            waypoint = self.grid.step(self.dt)
            if waypoint is None:
                self.phase = Phase.RTH
                return
            x, y, _z, yaw = waypoint
        else:
            x, y, _z = cx, cy, _cz
            yaw = self._yaw_from_current_position(x, y)

        if self.tick % 50 == 0:
            self.get_logger().info(
                f"SEARCH grid={self.grid.progress:.0%} "
                f"pos=({self.local_pos.x:.1f},{self.local_pos.y:.1f}) "
                f"target=({x:.1f},{y:.1f}) dist={dist_to_target:.1f}m "
                f"z={self.local_pos.z:.1f}m "
                f"{'advancing' if dist_to_target < self.grid.spacing else 'transiting'}"
            )

        self._publish_setpoint(x, y, self.cruise_altitude, yaw)


MissionController = GridMissionController


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GridMissionController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
