"""Candidate-based macro-actions for V2 search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

from uav_rl.belief_map import (
    BeliefMap,
    CHANNEL_OBSERVED,
    CHANNEL_OBSTACLE,
    CHANNEL_UNCERTAINTY,
    CHANNEL_VICTIM_SCORE,
    MapGeometry,
)
from uav_rl.camera_model import ForwardCameraModel
from uav_rl.detection_memory import DetectionMemory, DetectionTrack
from uav_rl.rl_common import wrap_angle_rad


class SearchAction(IntEnum):
    FRONTIER_BEST = 0
    FRONTIER_SECOND = 1
    FRONTIER_THIRD = 2
    DETECTION_CONFIRM_BEST = 3
    DETECTION_CONFIRM_ALT = 4
    HIGH_INFO_BEST = 5
    HIGH_INFO_SECOND = 6
    RETURN_CENTER = 7
    ESCAPE_STUCK = 8
    HOVER_SCAN = 9


ACTION_NAMES = {
    SearchAction.FRONTIER_BEST: "candidate_frontier_best",
    SearchAction.FRONTIER_SECOND: "candidate_frontier_second",
    SearchAction.FRONTIER_THIRD: "candidate_frontier_third",
    SearchAction.DETECTION_CONFIRM_BEST: "candidate_detection_confirm_best",
    SearchAction.DETECTION_CONFIRM_ALT: "candidate_detection_confirm_alt",
    SearchAction.HIGH_INFO_BEST: "candidate_high_info_best",
    SearchAction.HIGH_INFO_SECOND: "candidate_high_info_second",
    SearchAction.RETURN_CENTER: "candidate_return_center",
    SearchAction.ESCAPE_STUCK: "candidate_escape_stuck",
    SearchAction.HOVER_SCAN: "candidate_hover_scan",
}


@dataclass(frozen=True)
class ActionConfig:
    cell_size_m: float = 4.0
    fixed_altitude_ned: float = -4.0
    scan_yaw_step_rad: float = math.pi / 4.0
    investigate_standoff_m: float = 8.0
    max_candidate_count: int = 10


@dataclass(frozen=True)
class ActionGoal:
    x: float
    y: float
    z: float
    yaw: float
    name: str
    overflow_m: float = 0.0
    target_xy: tuple[float, float] | None = None
    valid: bool = True
    score: float = 0.0


def coerce_action(action: int | SearchAction) -> SearchAction:
    return SearchAction(int(action))


def move_delta(_action: int | SearchAction) -> tuple[float, float] | None:
    """Compatibility hook for old primitive actions; V2 actions are candidates."""

    return None


def actions_are_opposed(
    _previous_action: int | SearchAction,
    _current_action: int | SearchAction,
    *,
    dot_threshold: float = -0.75,
) -> bool:
    """Compatibility hook; candidate actions are not directional primitives."""

    _ = dot_threshold
    return False


def slew_yaw(current_yaw: float, target_yaw: float, max_step_rad: float) -> float:
    """Move from current yaw toward target yaw by at most max_step_rad."""

    max_step = max(0.0, float(max_step_rad))
    delta = wrap_angle_rad(float(target_yaw) - float(current_yaw))
    if abs(delta) <= max_step:
        return wrap_angle_rad(float(target_yaw))
    return wrap_angle_rad(float(current_yaw) + math.copysign(max_step, delta))


def _invalid_goal(
    *,
    x: float,
    y: float,
    yaw: float,
    config: ActionConfig,
    name: str,
) -> ActionGoal:
    return ActionGoal(
        x=float(x),
        y=float(y),
        z=config.fixed_altitude_ned,
        yaw=float(yaw),
        name=f"{name}_invalid",
        valid=False,
    )


def _goal_for_viewing_point(
    *,
    target_x: float,
    target_y: float,
    x: float,
    y: float,
    yaw: float,
    geometry: MapGeometry,
    camera: ForwardCameraModel,
    config: ActionConfig,
    name: str,
    standoff_m: float | None = None,
    bearing_from_target_rad: float | None = None,
) -> ActionGoal:
    min_range, max_range = camera.ground_visibility_band(config.fixed_altitude_ned)
    range_m = float(standoff_m) if standoff_m is not None else config.investigate_standoff_m
    range_m = min(max(range_m, min_range + config.cell_size_m), max_range * 0.85)

    if bearing_from_target_rad is None:
        from_target_x = float(x) - float(target_x)
        from_target_y = float(y) - float(target_y)
        if math.hypot(from_target_x, from_target_y) < 1e-6:
            from_target_x = -math.cos(float(yaw))
            from_target_y = -math.sin(float(yaw))
        bearing_from_target_rad = math.atan2(from_target_y, from_target_x)

    goal_x = float(target_x) + range_m * math.cos(float(bearing_from_target_rad))
    goal_y = float(target_y) + range_m * math.sin(float(bearing_from_target_rad))
    goal_x, goal_y, overflow = geometry.clip_xy(goal_x, goal_y)
    target_yaw = math.atan2(float(target_y) - goal_y, float(target_x) - goal_x)
    return ActionGoal(
        x=goal_x,
        y=goal_y,
        z=config.fixed_altitude_ned,
        yaw=target_yaw,
        name=name,
        overflow_m=overflow,
        target_xy=(float(target_x), float(target_y)),
    )


def _new_visible_score(
    *,
    belief: BeliefMap,
    camera: ForwardCameraModel,
    goal: ActionGoal,
) -> float:
    cells = camera.visible_cells(
        geometry=belief.geometry,
        drone_x=goal.x,
        drone_y=goal.y,
        drone_yaw=goal.yaw,
        altitude_ned_z=goal.z,
    )
    score = 0.0
    for row, col in cells:
        cell = belief.grid[row, col]
        if cell[CHANNEL_OBSTACLE] > 0.5:
            continue
        if cell[CHANNEL_OBSERVED] <= 0.0:
            score += 1.0
        score += 0.25 * float(cell[CHANNEL_UNCERTAINTY])
        score += 1.5 * float(cell[CHANNEL_VICTIM_SCORE])
    return score


def _rank_goals(
    *,
    goals: list[ActionGoal],
    x: float,
    y: float,
) -> list[ActionGoal]:
    deduped: list[ActionGoal] = []
    seen: set[tuple[int, int]] = set()
    for goal in sorted(
        goals,
        key=lambda item: (
            -item.score,
            math.hypot(item.x - float(x), item.y - float(y)),
            item.overflow_m,
        ),
    ):
        key = (int(round(goal.x * 10.0)), int(round(goal.y * 10.0)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(goal)
    return deduped


def _frontier_goals(
    *,
    belief: BeliefMap,
    camera: ForwardCameraModel,
    x: float,
    y: float,
    yaw: float,
    config: ActionConfig,
    limit: int,
) -> list[ActionGoal]:
    goals: list[ActionGoal] = []
    for row, col in belief.frontier_cells():
        target_x, target_y = belief.geometry.grid_to_world(row, col)
        goal = _goal_for_viewing_point(
            target_x=target_x,
            target_y=target_y,
            x=x,
            y=y,
            yaw=yaw,
            geometry=belief.geometry,
            camera=camera,
            config=config,
            name="frontier_viewpoint",
        )
        score = _new_visible_score(belief=belief, camera=camera, goal=goal)
        distance_penalty = 0.03 * math.hypot(goal.x - float(x), goal.y - float(y))
        goals.append(
            ActionGoal(
                **{**goal.__dict__, "score": score - distance_penalty}
            )
        )
    return _rank_goals(goals=goals, x=x, y=y)[:limit]


def _detection_goals(
    *,
    memory: DetectionMemory,
    belief: BeliefMap,
    camera: ForwardCameraModel,
    x: float,
    y: float,
    yaw: float,
    config: ActionConfig,
    limit: int,
) -> list[ActionGoal]:
    tracks = sorted(
        (
            track
            for track in memory.tracks
            if not track.confirmed and not track.investigated
        ),
        key=lambda track: (-track.mean_confidence, -track.hits, track.age_steps(10**9)),
    )
    goals: list[ActionGoal] = []
    for track in tracks[: max(limit, 1)]:
        target_x, target_y, _target_z = track.filtered_position
        bearings: list[float] = []
        if track.viewpoints:
            last_x, last_y, _last_yaw = track.viewpoints[-1]
            previous_bearing = math.atan2(last_y - target_y, last_x - target_x)
            bearings.append(wrap_angle_rad(previous_bearing + math.pi))
        bearings.append(math.atan2(float(y) - target_y, float(x) - target_x))

        for idx, bearing in enumerate(bearings):
            goal = _goal_for_viewing_point(
                target_x=target_x,
                target_y=target_y,
                x=x,
                y=y,
                yaw=yaw,
                geometry=belief.geometry,
                camera=camera,
                config=config,
                name="detection_confirmation_viewpoint",
                standoff_m=config.investigate_standoff_m,
                bearing_from_target_rad=bearing,
            )
            diversity = 0.0
            if track.viewpoints:
                diversity = min(
                    1.0,
                    math.hypot(goal.x - track.viewpoints[-1][0], goal.y - track.viewpoints[-1][1])
                    / max(config.investigate_standoff_m, 1.0),
                )
            score = (
                10.0 * track.mean_confidence
                + 1.5 * track.hits
                + 3.0 * diversity
                + 0.2 * _new_visible_score(belief=belief, camera=camera, goal=goal)
                - 0.02 * math.hypot(goal.x - float(x), goal.y - float(y))
                - 0.5 * idx
            )
            goals.append(ActionGoal(**{**goal.__dict__, "score": score}))
    return _rank_goals(goals=goals, x=x, y=y)[:limit]


def _high_info_goals(
    *,
    belief: BeliefMap,
    camera: ForwardCameraModel,
    x: float,
    y: float,
    yaw: float,
    config: ActionConfig,
    recent_cells: list[tuple[int, int]] | None,
    limit: int,
) -> list[ActionGoal]:
    recent = set(recent_cells or [])
    goals: list[ActionGoal] = []
    for row, col, target_x, target_y in belief.geometry.iter_cell_centers():
        if (row, col) in recent:
            continue
        if belief.grid[row, col, CHANNEL_OBSTACLE] > 0.5:
            continue
        if belief.grid[row, col, CHANNEL_OBSERVED] > 0.0 and belief.grid[row, col, CHANNEL_UNCERTAINTY] < 0.25:
            continue
        goal = _goal_for_viewing_point(
            target_x=target_x,
            target_y=target_y,
            x=x,
            y=y,
            yaw=yaw,
            geometry=belief.geometry,
            camera=camera,
            config=config,
            name="high_information_viewpoint",
        )
        score = (
            _new_visible_score(belief=belief, camera=camera, goal=goal)
            + 0.15 * math.hypot(target_x - float(x), target_y - float(y))
            - 0.02 * math.hypot(goal.x - float(x), goal.y - float(y))
        )
        goals.append(ActionGoal(**{**goal.__dict__, "score": score}))
    return _rank_goals(goals=goals, x=x, y=y)[:limit]


def _return_center_goal(
    *,
    belief: BeliefMap,
    camera: ForwardCameraModel,
    x: float,
    y: float,
    config: ActionConfig,
) -> ActionGoal:
    center_x = belief.geometry.origin_x
    center_y = belief.geometry.origin_y
    target_yaw = math.atan2(center_y - float(y), center_x - float(x))
    goal_x, goal_y, overflow = belief.geometry.clip_xy(center_x, center_y)
    goal = ActionGoal(
        x=goal_x,
        y=goal_y,
        z=config.fixed_altitude_ned,
        yaw=target_yaw,
        name="return_to_center",
        overflow_m=overflow,
        target_xy=(center_x, center_y),
    )
    score = _new_visible_score(belief=belief, camera=camera, goal=goal)
    return ActionGoal(**{**goal.__dict__, "score": score})


def generate_candidate_goals(
    *,
    x: float,
    y: float,
    yaw: float,
    belief: BeliefMap,
    memory: DetectionMemory,
    camera: ForwardCameraModel,
    config: ActionConfig,
    scan_allowed: bool = True,
    recent_cells: list[tuple[int, int]] | None = None,
) -> list[ActionGoal]:
    """Return one goal per discrete action slot."""

    max_count = max(len(SearchAction), int(config.max_candidate_count))
    candidates = [
        _invalid_goal(x=x, y=y, yaw=yaw, config=config, name=ACTION_NAMES[SearchAction(idx)])
        for idx in range(len(SearchAction))
    ]

    frontier = _frontier_goals(
        belief=belief,
        camera=camera,
        x=x,
        y=y,
        yaw=yaw,
        config=config,
        limit=3,
    )
    for slot, goal in zip(
        (SearchAction.FRONTIER_BEST, SearchAction.FRONTIER_SECOND, SearchAction.FRONTIER_THIRD),
        frontier,
    ):
        candidates[int(slot)] = ActionGoal(**{**goal.__dict__, "name": ACTION_NAMES[slot]})

    detection = _detection_goals(
        memory=memory,
        belief=belief,
        camera=camera,
        x=x,
        y=y,
        yaw=yaw,
        config=config,
        limit=2,
    )
    for slot, goal in zip(
        (SearchAction.DETECTION_CONFIRM_BEST, SearchAction.DETECTION_CONFIRM_ALT),
        detection,
    ):
        candidates[int(slot)] = ActionGoal(**{**goal.__dict__, "name": ACTION_NAMES[slot]})

    high_info = _high_info_goals(
        belief=belief,
        camera=camera,
        x=x,
        y=y,
        yaw=yaw,
        config=config,
        recent_cells=recent_cells,
        limit=3,
    )
    for slot, goal in zip(
        (SearchAction.HIGH_INFO_BEST, SearchAction.HIGH_INFO_SECOND, SearchAction.ESCAPE_STUCK),
        high_info,
    ):
        candidates[int(slot)] = ActionGoal(**{**goal.__dict__, "name": ACTION_NAMES[slot]})

    candidates[int(SearchAction.RETURN_CENTER)] = ActionGoal(
        **{**_return_center_goal(belief=belief, camera=camera, x=x, y=y, config=config).__dict__, "name": ACTION_NAMES[SearchAction.RETURN_CENTER]}
    )

    if scan_allowed:
        candidates[int(SearchAction.HOVER_SCAN)] = ActionGoal(
            x=float(x),
            y=float(y),
            z=config.fixed_altitude_ned,
            yaw=wrap_angle_rad(float(yaw) + config.scan_yaw_step_rad),
            name=ACTION_NAMES[SearchAction.HOVER_SCAN],
            score=0.0,
        )

    return candidates[:max_count]


def best_valid_candidate(candidates: list[ActionGoal]) -> ActionGoal | None:
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        return None
    return max(valid, key=lambda candidate: candidate.score)


def action_to_goal(
    *,
    action: int | SearchAction,
    x: float,
    y: float,
    yaw: float,
    geometry: MapGeometry,
    config: ActionConfig,
    best_track: DetectionTrack | None = None,
    belief: BeliefMap | None = None,
    memory: DetectionMemory | None = None,
    camera: ForwardCameraModel | None = None,
    scan_allowed: bool = True,
    recent_cells: list[tuple[int, int]] | None = None,
) -> ActionGoal:
    """Map a discrete candidate slot to a goal.

    ``geometry`` and ``best_track`` are kept for older callers/tests; candidate
    selection uses the shared belief map, detection memory, and camera model.
    """

    _ = best_track
    search_action = coerce_action(action)
    if belief is None or memory is None or camera is None:
        return _invalid_goal(
            x=x,
            y=y,
            yaw=yaw,
            config=config,
            name=ACTION_NAMES[search_action],
        )

    candidates = generate_candidate_goals(
        x=x,
        y=y,
        yaw=yaw,
        belief=belief,
        memory=memory,
        camera=camera,
        config=config,
        scan_allowed=scan_allowed,
        recent_cells=recent_cells,
    )
    requested = candidates[int(search_action)]
    if requested.valid:
        return requested
    fallback = best_valid_candidate(candidates)
    if fallback is None:
        fallback_x, fallback_y, overflow = geometry.clip_xy(x, y)
        return ActionGoal(
            x=fallback_x,
            y=fallback_y,
            z=config.fixed_altitude_ned,
            yaw=float(yaw),
            name=f"{ACTION_NAMES[search_action]}_invalid_no_fallback",
            overflow_m=overflow,
            valid=False,
        )
    return ActionGoal(
        **{
            **fallback.__dict__,
            "name": f"{fallback.name}_substituted_for_{ACTION_NAMES[search_action]}",
        }
    )


def direction_to_action(dx: float, dy: float) -> SearchAction:
    """Compatibility helper: map a desired direction to the closest candidate family."""

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return SearchAction.HOVER_SCAN
    return SearchAction.HIGH_INFO_BEST
