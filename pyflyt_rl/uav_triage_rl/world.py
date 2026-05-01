"""Search-world layout and PyFlyt/PyBullet rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


TRIAGE_LEVELS = ("critical", "delayed", "minor")
TRIAGE_COLORS = {
    "critical": (0.95, 0.05, 0.04, 1.0),
    "delayed": (1.0, 0.82, 0.05, 1.0),
    "minor": (0.05, 0.65, 0.20, 1.0),
}


@dataclass(frozen=True)
class VictimSpec:
    id: int
    x: float
    y: float
    yaw: float
    triage: str = "critical"
    height_m: float = 0.25


@dataclass(frozen=True)
class ObstacleSpec:
    id: int
    x: float
    y: float
    yaw: float
    size_x: float
    size_y: float
    height: float


@dataclass(frozen=True)
class WorldConfig:
    width_m: float = 40.0
    height_m: float = 40.0
    cell_size_m: float = 2.0
    search_altitude_m: float = 6.0
    max_speed_m_s: float = 3.0
    decision_period_s: float = 1.5
    max_episode_steps: int = 220
    randomize_start: bool = True
    victim_count: int = 6
    required_victim_count: int = 4
    obstacle_count: int = 10
    min_victim_spacing_m: float = 4.0
    seed: int = 7
    backend: str = "pyflyt"
    physics_hz: int = 240
    control_hz: int = 30
    drone_model: str = "primitive_drone"
    camera_fps: int = 15

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "WorldConfig":
        raw = dict(raw or {})
        pyflyt = dict(raw.pop("pyflyt", {}) or {})
        if pyflyt:
            raw["physics_hz"] = int(pyflyt.get("physics_hz", raw.get("physics_hz", 240)))
            raw["control_hz"] = int(pyflyt.get("control_hz", raw.get("control_hz", 30)))
            raw["drone_model"] = str(pyflyt.get("drone_model", raw.get("drone_model", "primitive_drone")))
            raw["camera_fps"] = int(pyflyt.get("camera_fps", raw.get("camera_fps", 15)))
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    @property
    def x_limits(self) -> tuple[float, float]:
        return (-self.width_m / 2.0, self.width_m / 2.0)

    @property
    def y_limits(self) -> tuple[float, float]:
        return (-self.height_m / 2.0, self.height_m / 2.0)


@dataclass(frozen=True)
class WorldLayout:
    victims: list[VictimSpec]
    obstacles: list[ObstacleSpec]


def _far_enough(x: float, y: float, points: list[tuple[float, float]], min_distance: float) -> bool:
    return all(math.hypot(x - px, y - py) >= min_distance for px, py in points)


def sample_layout(
    config: WorldConfig,
    rng: np.random.Generator,
    *,
    start_xy: tuple[float, float] = (0.0, 0.0),
) -> WorldLayout:
    """Sample victim and obstacle positions inside the search box."""

    x_min, x_max = config.x_limits
    y_min, y_max = config.y_limits
    victims: list[VictimSpec] = []
    used: list[tuple[float, float]] = [start_xy]
    min_spacing = max(float(config.min_victim_spacing_m), float(config.cell_size_m))

    for idx in range(max(0, int(config.victim_count))):
        chosen: tuple[float, float] | None = None
        for _ in range(500):
            x = float(rng.uniform(x_min + 1.0, x_max - 1.0))
            y = float(rng.uniform(y_min + 1.0, y_max - 1.0))
            if _far_enough(x, y, used, min_spacing):
                chosen = (x, y)
                break
        if chosen is None:
            chosen = (
                float(rng.uniform(x_min + 1.0, x_max - 1.0)),
                float(rng.uniform(y_min + 1.0, y_max - 1.0)),
            )
        used.append(chosen)
        victims.append(
            VictimSpec(
                id=idx,
                x=chosen[0],
                y=chosen[1],
                yaw=float(rng.uniform(-math.pi, math.pi)),
                triage=TRIAGE_LEVELS[idx % len(TRIAGE_LEVELS)],
            )
        )

    obstacles: list[ObstacleSpec] = []
    for idx in range(max(0, int(config.obstacle_count))):
        for _ in range(300):
            x = float(rng.uniform(x_min + 2.0, x_max - 2.0))
            y = float(rng.uniform(y_min + 2.0, y_max - 2.0))
            if _far_enough(x, y, used, min_spacing * 0.65):
                break
        obstacles.append(
            ObstacleSpec(
                id=idx,
                x=x,
                y=y,
                yaw=float(rng.uniform(-math.pi, math.pi)),
                size_x=float(rng.uniform(1.2, 3.0)),
                size_y=float(rng.uniform(0.8, 2.2)),
                height=float(rng.uniform(0.4, 1.5)),
            )
        )

    return WorldLayout(victims=victims, obstacles=obstacles)


class KinematicRuntime:
    """Small non-rendering fallback used by tests and explicit kinematic configs."""

    def __init__(self, config: WorldConfig, layout: WorldLayout, *, render: bool = False) -> None:
        self.config = config
        self.layout = layout
        self.render_enabled = render
        self.x = 0.0
        self.y = 0.0
        self.z = config.search_altitude_m
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self._goal = np.array([0.0, 0.0, 0.0, config.search_altitude_m], dtype=float)

    def start(self, start_state: tuple[float, float, float, float]) -> None:
        self.x, self.y, self.yaw, self.z = map(float, start_state)
        self._goal = np.array([self.x, self.y, self.yaw, self.z], dtype=float)

    def set_goal(self, x: float, y: float, yaw: float, z: float) -> None:
        self._goal = np.array([x, y, yaw, z], dtype=float)

    def step_many(self, count: int) -> None:
        dt = max(self.config.decision_period_s, 1e-6)
        old_x, old_y = self.x, self.y
        dx = float(self._goal[0] - self.x)
        dy = float(self._goal[1] - self.y)
        max_move = self.config.max_speed_m_s * dt
        distance = math.hypot(dx, dy)
        if distance > max_move > 0:
            scale = max_move / distance
            dx *= scale
            dy *= scale
        self.x += dx
        self.y += dy
        self.yaw = float(math.atan2(math.sin(self._goal[2]), math.cos(self._goal[2])))
        self.z = float(self._goal[3])
        self.vx = (self.x - old_x) / dt
        self.vy = (self.y - old_y) / dt
        _ = count

    def state(self) -> tuple[float, float, float, float, float, float]:
        return self.x, self.y, self.z, self.yaw, self.vx, self.vy

    def camera_image(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        return


class PyFlytRuntime:
    """Thin wrapper around PyFlyt Aviary using QuadX position mode."""

    def __init__(
        self,
        config: WorldConfig,
        layout: WorldLayout,
        *,
        camera_config: Any,
        render: bool = False,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.layout = layout
        self.camera_config = camera_config
        self.render_enabled = render
        self.seed = seed
        self.aviary = None

    def start(self, start_state: tuple[float, float, float, float]) -> None:
        try:
            from PyFlyt.core import Aviary
        except ImportError as exc:
            raise RuntimeError(
                "PyFlyt is required for backend='pyflyt'. Run: bash scripts/setup_macos_cpu.sh"
            ) from exc

        start_x, start_y, yaw, z = map(float, start_state)
        start_pos = np.array([[start_x, start_y, z]], dtype=float)
        start_orn = np.array([[0.0, 0.0, yaw]], dtype=float)
        drone_options = {
            "use_camera": True,
            "drone_model": self.config.drone_model,
            "control_hz": self.config.control_hz,
            "camera_fps": self.config.camera_fps,
            "camera_angle_degrees": int(round(float(self.camera_config.camera_pitch_deg))),
            "camera_FOV_degrees": int(round(float(self.camera_config.horizontal_fov_deg))),
            "camera_resolution": (
                int(self.camera_config.image_height_px),
                int(self.camera_config.image_width_px),
            ),
        }
        self.aviary = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=self.render_enabled,
            drone_type="quadx",
            drone_options=drone_options,
            physics_hz=self.config.physics_hz,
            world_scale=max(self.config.width_m, self.config.height_m) / 10.0,
            seed=self.seed,
        )
        self.aviary.set_mode(7)
        self._spawn_world_bodies()
        self.aviary.register_all_new_bodies()

    def _box(self, *, half_extents: tuple[float, float, float], color: tuple[float, float, float, float],
             position: tuple[float, float, float], yaw: float, collision: bool = True) -> int:
        import pybullet as pb

        assert self.aviary is not None
        orn = self.aviary.getQuaternionFromEuler([0.0, 0.0, yaw])
        visual = self.aviary.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )
        collision_shape = -1
        if collision:
            collision_shape = self.aviary.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=half_extents,
            )
        return int(
            self.aviary.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual,
                basePosition=position,
                baseOrientation=orn,
            )
        )

    def _sphere(self, *, radius: float, color: tuple[float, float, float, float],
                position: tuple[float, float, float]) -> int:
        import pybullet as pb

        assert self.aviary is not None
        visual = self.aviary.createVisualShape(
            pb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        collision = self.aviary.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
        return int(
            self.aviary.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position,
            )
        )

    def _offset(self, x: float, y: float, yaw: float, fwd: float, side: float) -> tuple[float, float]:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return x + c * fwd - s * side, y + s * fwd + c * side

    def _spawn_world_bodies(self) -> None:
        for victim in self.layout.victims:
            color = TRIAGE_COLORS.get(victim.triage, TRIAGE_COLORS["critical"])
            z = 0.08
            self._box(
                half_extents=(0.55, 0.18, 0.08),
                color=color,
                position=(victim.x, victim.y, z),
                yaw=victim.yaw,
            )
            hx, hy = self._offset(victim.x, victim.y, victim.yaw, 0.68, 0.0)
            self._sphere(radius=0.16, color=(0.82, 0.62, 0.48, 1.0), position=(hx, hy, 0.14))
            for fwd, side in [(-0.35, -0.28), (-0.35, 0.28), (0.28, -0.28), (0.28, 0.28)]:
                lx, ly = self._offset(victim.x, victim.y, victim.yaw, fwd, side)
                self._box(
                    half_extents=(0.28, 0.055, 0.055),
                    color=(0.08, 0.10, 0.12, 1.0),
                    position=(lx, ly, 0.07),
                    yaw=victim.yaw + (0.35 if side > 0 else -0.35),
                )

        for obstacle in self.layout.obstacles:
            self._box(
                half_extents=(obstacle.size_x / 2.0, obstacle.size_y / 2.0, obstacle.height / 2.0),
                color=(0.30, 0.33, 0.36, 1.0),
                position=(obstacle.x, obstacle.y, obstacle.height / 2.0),
                yaw=obstacle.yaw,
            )

    def set_goal(self, x: float, y: float, yaw: float, z: float) -> None:
        assert self.aviary is not None
        self.aviary.set_setpoint(0, np.array([x, y, yaw, z], dtype=float))

    def step_many(self, count: int) -> None:
        assert self.aviary is not None
        for _ in range(max(1, int(count))):
            self.aviary.step()

    def state(self) -> tuple[float, float, float, float, float, float]:
        assert self.aviary is not None
        state = self.aviary.state(0)
        pos = state[3]
        yaw = float(state[1, 2])
        vel = state[2]
        return float(pos[0]), float(pos[1]), float(pos[2]), yaw, float(vel[0]), float(vel[1])

    def camera_image(self) -> np.ndarray | None:
        if self.aviary is None or not self.aviary.drones:
            return None
        drone = self.aviary.drones[0]
        rgba = getattr(drone, "rgbaImg", None)
        if rgba is None:
            return None
        arr = np.asarray(rgba)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[:, :, :3].astype(np.uint8, copy=False)
        return None

    def close(self) -> None:
        if self.aviary is not None:
            self.aviary.disconnect()
            self.aviary = None
