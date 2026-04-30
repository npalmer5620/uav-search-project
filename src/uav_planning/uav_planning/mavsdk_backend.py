"""Threaded MAVSDK adapter for PX4 mission control."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import math
import threading
from typing import Awaitable


@dataclass
class MavsdkStatus:
    connected: bool = False
    armed: bool = False
    offboard: bool = False
    armable: bool = False
    local_position_ok: bool = False
    global_position_ok: bool = False
    home_position_ok: bool = False
    health_all_ok: bool = False
    north_m: float = 0.0
    east_m: float = 0.0
    down_m: float = 0.0
    velocity_north_m_s: float = 0.0
    velocity_east_m_s: float = 0.0
    velocity_down_m_s: float = 0.0
    position_velocity_valid: bool = False
    yaw_rad: float = 0.0
    attitude_valid: bool = False
    last_error: str = ""


class MavsdkBackend:
    """Run MAVSDK's asyncio API behind a small thread-safe control facade."""

    def __init__(
        self,
        *,
        system_address: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.system_address = system_address
        self.status = MavsdkStatus()
        self._logger = logger or logging.getLogger(__name__)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mavsdk-backend",
            daemon=True,
        )
        self._drone = None
        self._started = False
        self._stop_requested = False
        self._last_position_command: tuple[float, float, float, float] | None = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        self._submit(self._connect())

    def close(self) -> None:
        self._stop_requested = True
        if self._started and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def arm(self) -> None:
        self._submit(self._arm())

    def takeoff(self, altitude_m: float) -> None:
        self._submit(self._takeoff(altitude_m))

    def start_offboard(self) -> None:
        self._submit(self._start_offboard())

    def stop_offboard(self) -> None:
        self._submit(self._stop_offboard())

    def land(self) -> None:
        self._submit(self._land())

    def set_position_ned(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        command = (float(x), float(y), float(z), math.degrees(float(yaw_rad)))
        self._last_position_command = command
        self._submit(self._set_position_ned(*command))

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro: Awaitable[object]) -> Future | None:
        if not self._started or self._stop_requested:
            close = getattr(coro, "close", None)
            if close is not None:
                close()
            return None
        guard_coro = self._guard(coro)
        try:
            return asyncio.run_coroutine_threadsafe(guard_coro, self._loop)
        except RuntimeError:
            guard_coro.close()
            close = getattr(coro, "close", None)
            if close is not None:
                close()
            return None

    async def _guard(self, coro: Awaitable[object]) -> None:
        try:
            await coro
        except Exception as exc:  # pragma: no cover - runtime integration path
            self.status.last_error = str(exc)
            self._logger.warning(f"MAVSDK command failed: {exc}")

    async def _connect(self) -> None:
        from mavsdk import System

        self._drone = System()
        await self._drone.connect(system_address=self.system_address)
        self._logger.info(f"MAVSDK connecting to {self.system_address}")

        async for state in self._drone.core.connection_state():
            if state.is_connected:
                self.status.connected = True
                self._logger.info("MAVSDK connected")
                break

        self._submit(self._watch_armed())
        self._submit(self._watch_flight_mode())
        self._submit(self._watch_health())
        self._submit(self._watch_position_velocity_ned())
        self._submit(self._watch_attitude_euler())

    async def _watch_armed(self) -> None:
        async for armed in self._drone.telemetry.armed():
            self.status.armed = bool(armed)
            if self._stop_requested:
                return

    async def _watch_flight_mode(self) -> None:
        async for mode in self._drone.telemetry.flight_mode():
            self.status.offboard = str(mode).endswith("OFFBOARD")
            if self._stop_requested:
                return

    async def _watch_health(self) -> None:
        async for health in self._drone.telemetry.health():
            self.status.armable = bool(health.is_armable)
            self.status.local_position_ok = bool(health.is_local_position_ok)
            self.status.global_position_ok = bool(health.is_global_position_ok)
            self.status.home_position_ok = bool(health.is_home_position_ok)
            self.status.health_all_ok = (
                bool(health.is_gyrometer_calibration_ok)
                and bool(health.is_accelerometer_calibration_ok)
                and bool(health.is_magnetometer_calibration_ok)
                and self.status.local_position_ok
                and self.status.global_position_ok
                and self.status.home_position_ok
                and self.status.armable
            )
            if self._stop_requested:
                return

    async def _watch_position_velocity_ned(self) -> None:
        async for position_velocity in self._drone.telemetry.position_velocity_ned():
            position = position_velocity.position
            velocity = position_velocity.velocity
            self.status.north_m = float(position.north_m)
            self.status.east_m = float(position.east_m)
            self.status.down_m = float(position.down_m)
            self.status.velocity_north_m_s = float(velocity.north_m_s)
            self.status.velocity_east_m_s = float(velocity.east_m_s)
            self.status.velocity_down_m_s = float(velocity.down_m_s)
            self.status.position_velocity_valid = True
            if self._stop_requested:
                return

    async def _watch_attitude_euler(self) -> None:
        async for attitude in self._drone.telemetry.attitude_euler():
            self.status.yaw_rad = math.atan2(
                math.sin(math.radians(float(attitude.yaw_deg))),
                math.cos(math.radians(float(attitude.yaw_deg))),
            )
            self.status.attitude_valid = True
            if self._stop_requested:
                return

    async def _arm(self) -> None:
        if self._drone is None or not self.status.connected:
            return
        await self._drone.action.arm()

    async def _takeoff(self, altitude_m: float) -> None:
        if self._drone is None or not self.status.connected:
            return
        await self._drone.action.set_takeoff_altitude(float(abs(altitude_m)))
        await self._drone.action.takeoff()

    async def _start_offboard(self) -> None:
        if self._drone is None or not self.status.connected:
            return
        if self._last_position_command is not None:
            await self._set_position_ned(*self._last_position_command)
        await self._drone.offboard.start()

    async def _stop_offboard(self) -> None:
        if self._drone is None or not self.status.connected:
            return
        await self._drone.offboard.stop()

    async def _land(self) -> None:
        if self._drone is None or not self.status.connected:
            return
        await self._drone.action.land()

    async def _set_position_ned(
        self,
        north_m: float,
        east_m: float,
        down_m: float,
        yaw_deg: float,
    ) -> None:
        if self._drone is None or not self.status.connected:
            return
        from mavsdk.offboard import PositionNedYaw

        await self._drone.offboard.set_position_ned(
            PositionNedYaw(north_m, east_m, down_m, yaw_deg)
        )
