"""Reusable mission-control primitives in local NED coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NedPosition:
    north_m: float
    east_m: float
    down_m: float

    def distance_to(self, other: "NedPosition") -> float:
        return math.sqrt(
            (self.north_m - other.north_m) ** 2
            + (self.east_m - other.east_m) ** 2
            + (self.down_m - other.down_m) ** 2
        )

    def horizontal_distance_to(self, other: "NedPosition") -> float:
        return math.hypot(self.north_m - other.north_m, self.east_m - other.east_m)


@dataclass(frozen=True)
class NedVelocity:
    north_m_s: float
    east_m_s: float
    down_m_s: float

    @property
    def speed(self) -> float:
        return math.sqrt(
            self.north_m_s**2 + self.east_m_s**2 + self.down_m_s**2
        )


@dataclass(frozen=True)
class NedState:
    position: NedPosition
    velocity: NedVelocity
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class PidGains:
    kp: tuple[float, float, float]
    ki: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kd: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VelocityLimits:
    horizontal_m_s: float = 2.0
    up_m_s: float = 1.5
    down_m_s: float = 1.5


@dataclass(frozen=True)
class PositionTolerance:
    horizontal_m: float = 0.75
    vertical_m: float = 0.5
    speed_m_s: float = 0.5

    def reached(self, state: NedState, target: NedPosition) -> bool:
        return (
            state.position.horizontal_distance_to(target) <= self.horizontal_m
            and abs(state.position.down_m - target.down_m) <= self.vertical_m
            and state.velocity.speed <= self.speed_m_s
        )


def _clip(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def _scale_horizontal(north: float, east: float, limit: float) -> tuple[float, float]:
    magnitude = math.hypot(north, east)
    if magnitude <= limit or magnitude <= 1e-9:
        return north, east
    scale = limit / magnitude
    return north * scale, east * scale


class PositionPid:
    """Simple cascaded position-to-velocity controller.

    This ports the assignment ``drone_pid.py`` idea into a reusable component:
    position error becomes a bounded velocity command, then velocity error is
    integrated with PID gains. For MAVSDK position offboard we use the bounded
    velocity command to generate smooth intermediate position setpoints.
    """

    def __init__(
        self,
        *,
        position_gain: tuple[float, float, float] = (0.45, 0.45, 0.45),
        velocity_gains: PidGains = PidGains(
            kp=(1.0, 1.0, 1.2),
            ki=(0.0, 0.0, 0.0),
            kd=(0.2, 0.2, 0.3),
        ),
        velocity_limits: VelocityLimits = VelocityLimits(),
    ) -> None:
        self.position_gain = position_gain
        self.velocity_gains = velocity_gains
        self.velocity_limits = velocity_limits
        self._velocity_integral = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        self._velocity_integral = [0.0, 0.0, 0.0]

    def velocity_setpoint(
        self,
        state: NedState,
        target: NedPosition,
    ) -> NedVelocity:
        north_error = target.north_m - state.position.north_m
        east_error = target.east_m - state.position.east_m
        down_error = target.down_m - state.position.down_m

        north_sp = self.position_gain[0] * north_error
        east_sp = self.position_gain[1] * east_error
        down_sp = self.position_gain[2] * down_error
        north_sp, east_sp = _scale_horizontal(
            north_sp, east_sp, self.velocity_limits.horizontal_m_s
        )
        down_sp = _clip(
            down_sp,
            -self.velocity_limits.up_m_s,
            self.velocity_limits.down_m_s,
        )
        return NedVelocity(north_sp, east_sp, down_sp)

    def next_position_setpoint(
        self,
        state: NedState,
        target: NedPosition,
        dt_s: float,
    ) -> NedPosition:
        velocity = self.velocity_setpoint(state, target)
        next_position = NedPosition(
            north_m=state.position.north_m + velocity.north_m_s * dt_s,
            east_m=state.position.east_m + velocity.east_m_s * dt_s,
            down_m=state.position.down_m + velocity.down_m_s * dt_s,
        )

        return NedPosition(
            north_m=_clamp_between(
                next_position.north_m, state.position.north_m, target.north_m
            ),
            east_m=_clamp_between(
                next_position.east_m, state.position.east_m, target.east_m
            ),
            down_m=_clamp_between(
                next_position.down_m, state.position.down_m, target.down_m
            ),
        )


def _clamp_between(value: float, start: float, end: float) -> float:
    low = min(start, end)
    high = max(start, end)
    return _clip(value, low, high)

