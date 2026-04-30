"""Telemetry-based checks for commanded NED paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from uav_planning.control_primitives import NedPosition, NedState


TelemetrySample = tuple[float, NedState]


@dataclass(frozen=True)
class WaypointVisit:
    index: int
    target: NedPosition
    closest: NedPosition
    horizontal_error_m: float
    vertical_error_m: float
    sample_time_s: float
    passed: bool


@dataclass(frozen=True)
class PathTraversalReport:
    visits: tuple[WaypointVisit, ...]
    trace_horizontal_length_m: float
    command_horizontal_length_m: float
    horizontal_tolerance_m: float
    vertical_tolerance_m: float
    path_length_ratio: float
    passed: bool

    def summary_lines(self) -> list[str]:
        lines = [
            (
                "Path verification: "
                f"trace={self.trace_horizontal_length_m:.2f}m, "
                f"commanded={self.command_horizontal_length_m:.2f}m, "
                f"ratio={self.path_length_ratio:.2f}, passed={self.passed}"
            )
        ]
        for visit in self.visits:
            lines.append(
                "  waypoint "
                f"{visit.index}: target=({visit.target.north_m:.2f}, "
                f"{visit.target.east_m:.2f}, {visit.target.down_m:.2f}) "
                f"closest=({visit.closest.north_m:.2f}, "
                f"{visit.closest.east_m:.2f}, {visit.closest.down_m:.2f}) "
                f"h_err={visit.horizontal_error_m:.2f}m "
                f"v_err={visit.vertical_error_m:.2f}m "
                f"passed={visit.passed}"
            )
        return lines


def verify_path_traversal(
    samples: Sequence[TelemetrySample],
    commanded: Sequence[NedPosition],
    *,
    horizontal_tolerance_m: float = 1.5,
    vertical_tolerance_m: float = 1.0,
    min_path_length_ratio: float = 0.55,
) -> PathTraversalReport:
    if not samples:
        raise ValueError("No telemetry samples available for path verification")
    if not commanded:
        raise ValueError("No commanded waypoints supplied for path verification")

    visits: list[WaypointVisit] = []
    cursor = 0
    all_reached_in_order = True

    for index, target in enumerate(commanded):
        best_idx, best_score = cursor, float("inf")
        for sample_idx in range(cursor, len(samples)):
            _sample_time, state = samples[sample_idx]
            horizontal = state.position.horizontal_distance_to(target)
            vertical = abs(state.position.down_m - target.down_m)
            score = math.hypot(horizontal, vertical)
            if score < best_score:
                best_idx = sample_idx
                best_score = score

        sample_time, state = samples[best_idx]
        horizontal_error = state.position.horizontal_distance_to(target)
        vertical_error = abs(state.position.down_m - target.down_m)
        visit_passed = (
            horizontal_error <= horizontal_tolerance_m
            and vertical_error <= vertical_tolerance_m
        )
        all_reached_in_order = all_reached_in_order and visit_passed
        if visit_passed:
            cursor = best_idx

        visits.append(
            WaypointVisit(
                index=index,
                target=target,
                closest=state.position,
                horizontal_error_m=horizontal_error,
                vertical_error_m=vertical_error,
                sample_time_s=sample_time,
                passed=visit_passed,
            )
        )

    trace_horizontal_length = _trace_horizontal_length(samples)
    command_horizontal_length = _command_horizontal_length(samples[0][1].position, commanded)
    if command_horizontal_length <= 1e-6:
        path_length_ratio = 1.0
    else:
        path_length_ratio = trace_horizontal_length / command_horizontal_length

    passed = (
        all_reached_in_order
        and path_length_ratio >= min_path_length_ratio
    )
    return PathTraversalReport(
        visits=tuple(visits),
        trace_horizontal_length_m=trace_horizontal_length,
        command_horizontal_length_m=command_horizontal_length,
        horizontal_tolerance_m=horizontal_tolerance_m,
        vertical_tolerance_m=vertical_tolerance_m,
        path_length_ratio=path_length_ratio,
        passed=passed,
    )


def _trace_horizontal_length(samples: Sequence[TelemetrySample]) -> float:
    total = 0.0
    previous = samples[0][1].position
    for _sample_time, state in samples[1:]:
        total += previous.horizontal_distance_to(state.position)
        previous = state.position
    return total


def _command_horizontal_length(
    start: NedPosition,
    commanded: Sequence[NedPosition],
) -> float:
    total = 0.0
    previous = start
    for target in commanded:
        total += previous.horizontal_distance_to(target)
        previous = target
    return total
