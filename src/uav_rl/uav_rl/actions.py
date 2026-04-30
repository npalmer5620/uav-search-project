"""Discrete macro-actions for V2 search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

from uav_rl.belief_map import MapGeometry
from uav_rl.detection_memory import DetectionTrack
from uav_rl.rl_common import wrap_angle_rad


class SearchAction(IntEnum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    MOVE_NORTHEAST = 4
    MOVE_NORTHWEST = 5
    MOVE_SOUTHEAST = 6
    MOVE_SOUTHWEST = 7
    HOVER_SCAN = 8
    INVESTIGATE_BEST = 9


ACTION_NAMES = {
    SearchAction.MOVE_NORTH: "move_north",
    SearchAction.MOVE_SOUTH: "move_south",
    SearchAction.MOVE_EAST: "move_east",
    SearchAction.MOVE_WEST: "move_west",
    SearchAction.MOVE_NORTHEAST: "move_northeast",
    SearchAction.MOVE_NORTHWEST: "move_northwest",
    SearchAction.MOVE_SOUTHEAST: "move_southeast",
    SearchAction.MOVE_SOUTHWEST: "move_southwest",
    SearchAction.HOVER_SCAN: "hover_scan",
    SearchAction.INVESTIGATE_BEST: "investigate_best_detection",
}

MOVE_DELTAS = {
    SearchAction.MOVE_NORTH: (1.0, 0.0),
    SearchAction.MOVE_SOUTH: (-1.0, 0.0),
    SearchAction.MOVE_EAST: (0.0, 1.0),
    SearchAction.MOVE_WEST: (0.0, -1.0),
    SearchAction.MOVE_NORTHEAST: (1.0, 1.0),
    SearchAction.MOVE_NORTHWEST: (1.0, -1.0),
    SearchAction.MOVE_SOUTHEAST: (-1.0, 1.0),
    SearchAction.MOVE_SOUTHWEST: (-1.0, -1.0),
}


@dataclass(frozen=True)
class ActionConfig:
    cell_size_m: float = 4.0
    fixed_altitude_ned: float = -4.0
    scan_yaw_step_rad: float = math.pi / 4.0
    investigate_standoff_m: float = 8.0


@dataclass(frozen=True)
class ActionGoal:
    x: float
    y: float
    z: float
    yaw: float
    name: str
    overflow_m: float = 0.0
    target_xy: tuple[float, float] | None = None


def coerce_action(action: int | SearchAction) -> SearchAction:
    return SearchAction(int(action))


def move_delta(action: int | SearchAction) -> tuple[float, float] | None:
    try:
        search_action = coerce_action(action)
    except ValueError:
        return None
    return MOVE_DELTAS.get(search_action)


def actions_are_opposed(
    previous_action: int | SearchAction,
    current_action: int | SearchAction,
    *,
    dot_threshold: float = -0.75,
) -> bool:
    previous_delta = move_delta(previous_action)
    current_delta = move_delta(current_action)
    if previous_delta is None or current_delta is None:
        return False

    prev_norm = math.hypot(*previous_delta)
    curr_norm = math.hypot(*current_delta)
    if prev_norm < 1e-6 or curr_norm < 1e-6:
        return False

    dot = (
        previous_delta[0] * current_delta[0]
        + previous_delta[1] * current_delta[1]
    ) / (prev_norm * curr_norm)
    return dot <= float(dot_threshold)


def slew_yaw(current_yaw: float, target_yaw: float, max_step_rad: float) -> float:
    """Move from current yaw toward target yaw by at most max_step_rad."""

    max_step = max(0.0, float(max_step_rad))
    delta = wrap_angle_rad(float(target_yaw) - float(current_yaw))
    if abs(delta) <= max_step:
        return wrap_angle_rad(float(target_yaw))
    return wrap_angle_rad(float(current_yaw) + math.copysign(max_step, delta))


def action_to_goal(
    *,
    action: int | SearchAction,
    x: float,
    y: float,
    yaw: float,
    geometry: MapGeometry,
    config: ActionConfig,
    best_track: DetectionTrack | None = None,
) -> ActionGoal:
    search_action = coerce_action(action)

    if search_action in MOVE_DELTAS:
        dx_unit, dy_unit = MOVE_DELTAS[search_action]
        norm = math.hypot(dx_unit, dy_unit)
        step = config.cell_size_m
        dx = step * dx_unit / max(norm, 1.0)
        dy = step * dy_unit / max(norm, 1.0)
        target_x, target_y, overflow = geometry.clip_xy(x + dx, y + dy)
        target_yaw = math.atan2(dy, dx)
        return ActionGoal(
            x=target_x,
            y=target_y,
            z=config.fixed_altitude_ned,
            yaw=target_yaw,
            name=ACTION_NAMES[search_action],
            overflow_m=overflow,
        )

    if search_action == SearchAction.HOVER_SCAN:
        return ActionGoal(
            x=float(x),
            y=float(y),
            z=config.fixed_altitude_ned,
            yaw=wrap_angle_rad(float(yaw) + config.scan_yaw_step_rad),
            name=ACTION_NAMES[search_action],
        )

    if best_track is None:
        return ActionGoal(
            x=float(x),
            y=float(y),
            z=config.fixed_altitude_ned,
            yaw=wrap_angle_rad(float(yaw) + config.scan_yaw_step_rad),
            name=ACTION_NAMES[search_action],
            target_xy=None,
        )

    target_x, target_y, _target_z = best_track.filtered_position
    from_target_x = float(x) - target_x
    from_target_y = float(y) - target_y
    distance = math.hypot(from_target_x, from_target_y)
    if distance < 1e-6:
        from_target_x = -math.cos(float(yaw))
        from_target_y = -math.sin(float(yaw))
        distance = 1.0

    goal_x = target_x + config.investigate_standoff_m * from_target_x / distance
    goal_y = target_y + config.investigate_standoff_m * from_target_y / distance
    goal_x, goal_y, overflow = geometry.clip_xy(goal_x, goal_y)
    target_yaw = math.atan2(target_y - goal_y, target_x - goal_x)
    return ActionGoal(
        x=goal_x,
        y=goal_y,
        z=config.fixed_altitude_ned,
        yaw=target_yaw,
        name=ACTION_NAMES[search_action],
        overflow_m=overflow,
        target_xy=(target_x, target_y),
    )


def direction_to_action(dx: float, dy: float) -> SearchAction:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return SearchAction.HOVER_SCAN
    angle = math.atan2(dy, dx)
    directions = [
        (SearchAction.MOVE_NORTH, 0.0),
        (SearchAction.MOVE_NORTHEAST, math.pi / 4.0),
        (SearchAction.MOVE_EAST, math.pi / 2.0),
        (SearchAction.MOVE_SOUTHEAST, 3.0 * math.pi / 4.0),
        (SearchAction.MOVE_SOUTH, math.pi),
        (SearchAction.MOVE_SOUTHWEST, -3.0 * math.pi / 4.0),
        (SearchAction.MOVE_WEST, -math.pi / 2.0),
        (SearchAction.MOVE_NORTHWEST, -math.pi / 4.0),
    ]
    return min(directions, key=lambda item: abs(wrap_angle_rad(angle - item[1])))[0]
