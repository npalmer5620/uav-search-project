"""Shared RL environment and inference utilities."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_WORLD_TARGET_CLASSES = ["person", "person", "person", "car", "car"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path | None, *, default: Path | None = None) -> Path:
    if path is None or str(path).strip() == "":
        if default is None:
            raise ValueError("A path is required")
        return default

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class SearchTaskConfig:
    width: float = 40.0
    height: float = 40.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    coverage_grid_side: int = 16
    max_step_xy_m: float = 4.0
    max_yaw_step_rad: float = math.pi / 4.0
    decision_period_s: float = 0.5
    search_speed: float = 2.0
    target_detection_radius_m: float = 4.0
    coverage_reward_scale: float = 10.0
    first_target_found_bonus: float = 5.0
    success_bonus: float = 50.0
    step_penalty: float = 0.01
    boundary_penalty: float = 0.25
    timeout_penalty: float = 10.0
    stagnation_penalty: float = 0.002
    required_target_count: int | None = None
    max_episode_steps: int | None = None
    world_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SearchTaskConfig":
        if not raw:
            return cls()
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def x_limits(self) -> tuple[float, float]:
        return (
            self.origin_x - self.height / 2.0,
            self.origin_x + self.height / 2.0,
        )

    @property
    def y_limits(self) -> tuple[float, float]:
        return (
            self.origin_y - self.width / 2.0,
            self.origin_y + self.width / 2.0,
        )

    def clip_xy(self, x: float, y: float) -> tuple[float, float, float]:
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits
        clipped_x = min(max(float(x), x_min), x_max)
        clipped_y = min(max(float(y), y_min), y_max)
        overflow = math.hypot(clipped_x - x, clipped_y - y)
        return clipped_x, clipped_y, overflow

    def velocity_scale(self, decision_period_s: float) -> float:
        derived = self.max_step_xy_m / max(decision_period_s, 1e-6)
        return max(1.0, self.search_speed, derived)

    def effective_max_episode_steps(self) -> int:
        if self.max_episode_steps is not None:
            return max(1, int(self.max_episode_steps))

        coverage_budget = self.coverage_grid_side * self.coverage_grid_side * 2
        transit_budget = int(
            math.ceil((self.width + self.height) / max(self.max_step_xy_m, 0.1)) * 8
        )
        return max(256, coverage_budget + transit_budget)


@dataclass(frozen=True)
class TargetInstance:
    class_name: str
    x: float
    y: float


def infer_target_class(*values: str) -> str | None:
    text = " ".join(values).lower()
    if any(token in text for token in ("person", "female", "male", "visitor", "walking")):
        return "person"
    if "truck" in text:
        return "truck"
    if "bus" in text:
        return "bus"
    if "motorcycle" in text:
        return "motorcycle"
    if "bicycle" in text:
        return "bicycle"
    if any(token in text for token in ("car", "suv", "hatchback", "vehicle")):
        return "car"
    return None


def load_world_target_classes(world_path: str | Path | None) -> list[str]:
    resolved = resolve_repo_path(
        world_path,
        default=repo_root() / "worlds" / "search_area.sdf",
    )
    try:
        tree = ET.parse(resolved)
    except (ET.ParseError, FileNotFoundError):
        return list(DEFAULT_WORLD_TARGET_CLASSES)

    classes: list[str] = []
    for include in tree.findall(".//include"):
        uri = (include.findtext("uri") or "").strip()
        name = (include.findtext("name") or "").strip()
        class_name = infer_target_class(uri, name)
        if class_name is not None:
            classes.append(class_name)

    return classes or list(DEFAULT_WORLD_TARGET_CLASSES)


class CoverageMap:
    """Occupancy-style coverage accumulator over the configured search area."""

    def __init__(self, side: int, config: SearchTaskConfig) -> None:
        self.side = max(1, int(side))
        self.config = config
        self._grid = np.zeros((self.side, self.side), dtype=np.float32)
        self._cell_dx = config.height / self.side if self.side > 0 else config.height
        self._cell_dy = config.width / self.side if self.side > 0 else config.width
        self._sampling_step = max(
            min(self._cell_dx, self._cell_dy) / 2.0,
            0.25,
        )

    @property
    def num_cells(self) -> int:
        return int(self._grid.size)

    @property
    def coverage_fraction(self) -> float:
        return float(self._grid.mean())

    def reset(self) -> None:
        self._grid.fill(0.0)

    def flatten(self) -> np.ndarray:
        return self._grid.reshape(-1).astype(np.float32, copy=False)

    def _indices(self, x: float, y: float) -> tuple[int, int]:
        x_min, x_max = self.config.x_limits
        y_min, y_max = self.config.y_limits
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        row = int(np.clip(((x - x_min) / x_span) * self.side, 0, self.side - 1))
        col = int(np.clip(((y - y_min) / y_span) * self.side, 0, self.side - 1))
        return row, col

    def mark_point(self, x: float, y: float) -> float:
        row, col = self._indices(x, y)
        if self._grid[row, col] >= 1.0:
            return 0.0
        self._grid[row, col] = 1.0
        return 1.0 / float(self.num_cells)

    def update_segment(self, x0: float, y0: float, x1: float, y1: float) -> float:
        distance = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(distance / self._sampling_step)))
        delta = 0.0
        for step_idx in range(steps + 1):
            t = step_idx / steps
            sample_x = x0 + (x1 - x0) * t
            sample_y = y0 + (y1 - y0) * t
            delta += self.mark_point(sample_x, sample_y)
        return delta


class ObservationEncoder:
    """Builds the shared flat observation vector used in training and inference."""

    def __init__(
        self,
        config: SearchTaskConfig,
        coverage_map: CoverageMap,
        *,
        decision_period_s: float,
    ) -> None:
        self.config = config
        self.coverage_map = coverage_map
        self.velocity_scale = config.velocity_scale(decision_period_s)
        self.max_episode_steps = config.effective_max_episode_steps()
        self.max_episode_duration_s = self.max_episode_steps * max(decision_period_s, 1e-6)

    def encode(
        self,
        *,
        x: float,
        y: float,
        vx: float,
        vy: float,
        yaw: float,
        elapsed_s: float,
    ) -> np.ndarray:
        x_half = max(self.config.height / 2.0, 1e-6)
        y_half = max(self.config.width / 2.0, 1e-6)
        state = np.array(
            [
                np.clip((x - self.config.origin_x) / x_half, -1.0, 1.0),
                np.clip((y - self.config.origin_y) / y_half, -1.0, 1.0),
                np.clip(vx / self.velocity_scale, -1.0, 1.0),
                np.clip(vy / self.velocity_scale, -1.0, 1.0),
                np.clip(wrap_angle_rad(yaw) / math.pi, -1.0, 1.0),
                np.clip(elapsed_s / max(float(self.max_episode_duration_s), 1.0), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate((state, self.coverage_map.flatten()), dtype=np.float32)


def clip_action_to_deltas(
    action: np.ndarray | list[float] | tuple[float, ...],
    *,
    max_step_xy_m: float,
    max_yaw_step_rad: float,
) -> tuple[float, float, float]:
    action_array = np.asarray(action, dtype=np.float32).reshape(3)
    clipped = np.clip(action_array, -1.0, 1.0)
    return (
        float(clipped[0] * max_step_xy_m),
        float(clipped[1] * max_step_xy_m),
        float(clipped[2] * max_yaw_step_rad),
    )


def apply_relative_action(
    *,
    x: float,
    y: float,
    yaw: float,
    action: np.ndarray | list[float] | tuple[float, ...],
    config: SearchTaskConfig,
) -> tuple[float, float, float, float]:
    dx, dy, dyaw = clip_action_to_deltas(
        action,
        max_step_xy_m=config.max_step_xy_m,
        max_yaw_step_rad=config.max_yaw_step_rad,
    )
    target_x, target_y, overflow = config.clip_xy(x + dx, y + dy)
    target_yaw = wrap_angle_rad(yaw + dyaw)
    return target_x, target_y, target_yaw, overflow


def load_vecnormalize_for_inference(
    path: str | Path | None,
    config: SearchTaskConfig,
):
    """Load saved observation normalization stats for PPO inference."""

    if path is None or str(path).strip() == "":
        return None

    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return None

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from uav_rl.search_task_env import SearchTaskEnv

    vec_env = DummyVecEnv([lambda: SearchTaskEnv(config=config)])
    vec_normalize = VecNormalize.load(str(resolved), vec_env)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return vec_normalize
