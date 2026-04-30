"""Async MAVSDK mission primitives for SITL and smoke tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import math
import time
from typing import Iterable

from uav_planning.control_primitives import (
    NedPosition,
    NedState,
    NedVelocity,
    PositionPid,
    PositionTolerance,
)


@dataclass(frozen=True)
class MissionPrimitiveConfig:
    command_rate_hz: float = 10.0
    connect_timeout_s: float = 90.0
    health_timeout_s: float = 120.0
    action_timeout_s: float = 20.0
    goto_timeout_s: float = 60.0
    land_timeout_s: float = 90.0
    tolerance: PositionTolerance = PositionTolerance()

    @property
    def command_period_s(self) -> float:
        return 1.0 / max(self.command_rate_hz, 1.0)


class MavsdkMissionPrimitives:
    """Reusable MAVSDK operations in local NED coordinates.

    This is the practical port of the assignment ``offboard_position_ned.py``
    sequence: connect, wait for health, arm, seed an initial setpoint, start
    offboard, command NED positions, and land. ``goto_ned`` sends direct NED
    position targets by default, with optional ``PositionPid`` smoothing for
    callers that want bounded intermediate setpoints.
    """

    def __init__(
        self,
        *,
        system_address: str = "udpin://0.0.0.0:14540",
        config: MissionPrimitiveConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.system_address = system_address
        self.config = config or MissionPrimitiveConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.position_controller = PositionPid()
        self._drone = None
        self._state = NedState(
            position=NedPosition(0.0, 0.0, 0.0),
            velocity=NedVelocity(0.0, 0.0, 0.0),
            yaw_rad=0.0,
        )
        self._state_valid = asyncio.Event()
        self._armed = False
        self._offboard = False
        self._watch_tasks: list[asyncio.Task] = []
        self._history: list[tuple[float, NedState]] = []
        self._history_limit = 20000

    @property
    def state(self) -> NedState:
        return self._state

    def history_mark(self) -> int:
        return len(self._history)

    def history_since(self, mark: int = 0) -> list[tuple[float, NedState]]:
        mark = max(0, min(mark, len(self._history)))
        return list(self._history[mark:])

    async def connect(self) -> None:
        from mavsdk import System

        self._drone = System()
        await self._drone.connect(system_address=self.system_address)
        self.logger.info("MAVSDK connecting to %s", self.system_address)

        deadline = time.monotonic() + self.config.connect_timeout_s
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                self.logger.info("MAVSDK connected")
                self._start_watchers()
                return
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for MAVSDK connection")

    async def wait_ready(self) -> None:
        if self._drone is None:
            raise RuntimeError("connect() must be called before wait_ready()")

        deadline = time.monotonic() + self.config.health_timeout_s
        async for health in self._drone.telemetry.health():
            ready = (
                health.is_armable
                and health.is_local_position_ok
                and health.is_global_position_ok
                and health.is_home_position_ok
            )
            if ready:
                await asyncio.wait_for(
                    self._state_valid.wait(), timeout=self.config.action_timeout_s
                )
                self.logger.info("PX4 health ready")
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for PX4 health "
                    f"(armable={health.is_armable}, "
                    f"local={health.is_local_position_ok}, "
                    f"global={health.is_global_position_ok}, "
                    f"home={health.is_home_position_ok})"
                )

    async def arm(self) -> None:
        if self._drone is None:
            raise RuntimeError("connect() must be called before arm()")
        self.logger.info("Arming")
        await self._drone.action.arm()
        await self._wait_for(lambda: self._armed, "armed")

    async def start_offboard(
        self,
        initial: NedPosition | None = None,
        yaw_deg: float = 0.0,
    ) -> None:
        if self._drone is None:
            raise RuntimeError("connect() must be called before start_offboard()")
        if initial is None:
            initial = self.state.position

        await self.set_position_ned(initial, yaw_deg)
        await asyncio.sleep(self.config.command_period_s)
        self.logger.info("Starting offboard")
        await self._drone.offboard.start()
        await self._wait_for(lambda: self._offboard, "offboard")

    async def takeoff_offboard(self, down_m: float, yaw_deg: float = 0.0) -> None:
        await self.arm()
        await self.start_offboard(self.state.position, yaw_deg=yaw_deg)
        target = NedPosition(
            north_m=self.state.position.north_m,
            east_m=self.state.position.east_m,
            down_m=down_m,
        )
        await self.goto_ned(target, yaw_deg=yaw_deg)

    async def hold(self, duration_s: float, yaw_deg: float = 0.0) -> None:
        target = self.state.position
        self.logger.info("Holding %.1fs at %s", duration_s, target)
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            await self.set_position_ned(target, yaw_deg)
            await asyncio.sleep(self.config.command_period_s)

    async def goto_ned(
        self,
        target: NedPosition,
        *,
        yaw_deg: float = 0.0,
        timeout_s: float | None = None,
        smooth: bool = False,
    ) -> None:
        if timeout_s is None:
            timeout_s = self.config.goto_timeout_s

        self.position_controller.reset()
        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        while True:
            state = self.state
            if self.config.tolerance.reached(state, target):
                await self.set_position_ned(target, yaw_deg)
                self.logger.info(
                    "Reached NED target n=%.2f e=%.2f d=%.2f",
                    target.north_m,
                    target.east_m,
                    target.down_m,
                )
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out going to NED target "
                    f"target={target}, current={state.position}, velocity={state.velocity}"
                )

            if smooth:
                setpoint = self.position_controller.next_position_setpoint(
                    state,
                    target,
                    self.config.command_period_s,
                )
            else:
                setpoint = target
            await self.set_position_ned(setpoint, yaw_deg)

            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                self.logger.info(
                    "Goto NED current=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f)",
                    state.position.north_m,
                    state.position.east_m,
                    state.position.down_m,
                    target.north_m,
                    target.east_m,
                    target.down_m,
                )
            await asyncio.sleep(self.config.command_period_s)

    async def run_search_pattern(
        self,
        waypoints: Iterable[tuple[float, float, float, float]],
        *,
        max_waypoints: int | None = None,
        waypoint_timeout_s: float | None = None,
    ) -> int:
        completed = 0
        for north_m, east_m, down_m, yaw_rad in waypoints:
            if max_waypoints is not None and completed >= max_waypoints:
                return completed
            await self.goto_ned(
                NedPosition(north_m, east_m, down_m),
                yaw_deg=math.degrees(yaw_rad),
                timeout_s=waypoint_timeout_s or self.config.goto_timeout_s,
            )
            completed += 1
        return completed

    async def land(self) -> None:
        if self._drone is None:
            return
        self.logger.info("Landing")
        await self._drone.action.land()
        await self._wait_for(lambda: not self._armed, "disarmed", self.config.land_timeout_s)

    async def set_position_ned(self, position: NedPosition, yaw_deg: float = 0.0) -> None:
        if self._drone is None:
            raise RuntimeError("connect() must be called before set_position_ned()")
        from mavsdk.offboard import PositionNedYaw

        await self._drone.offboard.set_position_ned(
            PositionNedYaw(
                position.north_m,
                position.east_m,
                position.down_m,
                yaw_deg,
            )
        )

    async def close(self) -> None:
        for task in self._watch_tasks:
            task.cancel()
        if self._watch_tasks:
            await asyncio.gather(*self._watch_tasks, return_exceptions=True)
        self._watch_tasks.clear()

    def _start_watchers(self) -> None:
        self._watch_tasks = [
            asyncio.create_task(self._watch_position_velocity()),
            asyncio.create_task(self._watch_armed()),
            asyncio.create_task(self._watch_flight_mode()),
        ]

    async def _watch_position_velocity(self) -> None:
        async for position_velocity in self._drone.telemetry.position_velocity_ned():
            position = position_velocity.position
            velocity = position_velocity.velocity
            self._state = NedState(
                position=NedPosition(
                    float(position.north_m),
                    float(position.east_m),
                    float(position.down_m),
                ),
                velocity=NedVelocity(
                    float(velocity.north_m_s),
                    float(velocity.east_m_s),
                    float(velocity.down_m_s),
                ),
            )
            self._history.append((time.monotonic(), self._state))
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
            self._state_valid.set()

    async def _watch_armed(self) -> None:
        async for armed in self._drone.telemetry.armed():
            self._armed = bool(armed)

    async def _watch_flight_mode(self) -> None:
        async for mode in self._drone.telemetry.flight_mode():
            self._offboard = str(mode).endswith("OFFBOARD")

    async def _wait_for(
        self,
        predicate,
        description: str,
        timeout_s: float | None = None,
    ) -> None:
        if timeout_s is None:
            timeout_s = self.config.action_timeout_s
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for {description}")
            await asyncio.sleep(0.1)
