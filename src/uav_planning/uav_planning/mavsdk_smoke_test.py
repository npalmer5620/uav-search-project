#!/usr/bin/env python3
"""Focused MAVSDK SITL smoke test."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from uav_planning.control_primitives import NedPosition
from uav_planning.grid_generator import GridGenerator
from uav_planning.mavsdk_mission_primitives import (
    MavsdkMissionPrimitives,
    MissionPrimitiveConfig,
)
from uav_planning.path_verifier import verify_path_traversal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-address",
        default="udpin://0.0.0.0:14540",
        help="MAVSDK system address.",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=-10.0,
        help="Target NED down altitude for takeoff/search.",
    )
    parser.add_argument("--grid-width", type=float, default=20.0)
    parser.add_argument("--grid-height", type=float, default=20.0)
    parser.add_argument("--grid-spacing", type=float, default=10.0)
    parser.add_argument("--grid-speed", type=float, default=2.0)
    parser.add_argument(
        "--search-waypoints",
        type=int,
        default=4,
        help="Number of search waypoints to command before landing.",
    )
    parser.add_argument("--goto-timeout", type=float, default=75.0)
    parser.add_argument("--health-timeout", type=float, default=150.0)
    parser.add_argument("--land-timeout", type=float, default=90.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--verify-horizontal-tolerance", type=float, default=1.5)
    parser.add_argument("--verify-vertical-tolerance", type=float, default=1.0)
    parser.add_argument("--verify-min-path-ratio", type=float, default=0.55)
    return parser


async def _run(args: argparse.Namespace) -> None:
    logger = logging.getLogger("mavsdk_smoke_test")
    config = MissionPrimitiveConfig(
        health_timeout_s=args.health_timeout,
        goto_timeout_s=args.goto_timeout,
        land_timeout_s=args.land_timeout,
    )
    mission = MavsdkMissionPrimitives(
        system_address=args.system_address,
        config=config,
        logger=logger,
    )

    try:
        try:
            logger.info("SMOKE boot/connect: waiting for MAVSDK")
            await mission.connect()
            await mission.wait_ready()

            trace_mark = mission.history_mark()
            takeoff_target = NedPosition(
                mission.state.position.north_m,
                mission.state.position.east_m,
                args.altitude,
            )
            logger.info("SMOKE takeoff: arm and climb to %.1fm NED down", args.altitude)
            logger.info("Phase: BOOT -> ARMING")
            await mission.takeoff_offboard(args.altitude)
            logger.info("Phase: ARMING -> SEARCH")

            grid = GridGenerator(
                width=args.grid_width,
                height=args.grid_height,
                spacing=args.grid_spacing,
                speed=args.grid_speed,
                altitude=args.altitude,
            )
            waypoints = []
            for _ in range(max(1, args.search_waypoints)):
                waypoint = grid.step(1.0)
                if waypoint is None:
                    break
                waypoints.append(waypoint)

            logger.info("SMOKE search: commanding %d waypoint(s)", len(waypoints))
            completed = await mission.run_search_pattern(
                waypoints,
                max_waypoints=len(waypoints),
                waypoint_timeout_s=args.goto_timeout,
            )
            logger.info("SMOKE search: completed %d waypoint(s)", completed)
            await mission.hold(args.hold_seconds)

            commanded = [takeoff_target]
            commanded.extend(
                NedPosition(north_m, east_m, down_m)
                for north_m, east_m, down_m, _yaw_rad in waypoints[:completed]
            )
            report = verify_path_traversal(
                mission.history_since(trace_mark),
                commanded,
                horizontal_tolerance_m=args.verify_horizontal_tolerance,
                vertical_tolerance_m=args.verify_vertical_tolerance,
                min_path_length_ratio=args.verify_min_path_ratio,
            )
            for line in report.summary_lines():
                logger.info(line)
            if not report.passed:
                raise AssertionError("Telemetry path verification failed")

            logger.info("Phase: SEARCH -> LAND")
            await mission.land()
            logger.info("SMOKE PASS")
        except Exception:
            logger.exception("SMOKE FAIL")
            try:
                await mission.land()
            except Exception:
                logger.exception("SMOKE cleanup land failed")
            raise
    finally:
        await mission.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    logging.shutdown()
    os._exit(exit_code)
