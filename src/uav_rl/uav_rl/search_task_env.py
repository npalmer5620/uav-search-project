"""Task-level Gymnasium environment for search-policy training."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from uav_rl.rl_common import (
    CoverageMap,
    ObservationEncoder,
    SearchTaskConfig,
    TargetInstance,
    apply_relative_action,
    load_world_target_classes,
)


class SearchTaskEnv(gym.Env):
    """Fast search task used to train the high-level PPO policy."""

    metadata = {"render_modes": []}

    def __init__(self, config: SearchTaskConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config if isinstance(config, SearchTaskConfig) else SearchTaskConfig.from_mapping(config)
        )
        self.coverage = CoverageMap(self.config.coverage_grid_side, self.config)
        self.encoder = ObservationEncoder(
            self.config,
            self.coverage,
            decision_period_s=self.config.decision_period_s,
        )
        self.max_episode_steps = self.config.effective_max_episode_steps()
        self.target_class_mix = load_world_target_classes(self.config.world_path)
        self.required_target_count = (
            self.config.required_target_count
            if self.config.required_target_count is not None
            else len(self.target_class_mix)
        )

        obs_size = 6 + self.config.coverage_grid_side * self.config.coverage_grid_side
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.elapsed_steps = 0
        self.targets: list[TargetInstance] = []
        self.found_target_indices: set[int] = set()

    def _sample_start_state(self) -> tuple[float, float, float]:
        x_min, x_max = self.config.x_limits
        y_min, y_max = self.config.y_limits
        x = float(self.np_random.uniform(x_min, x_max))
        y = float(self.np_random.uniform(y_min, y_max))
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        return x, y, yaw

    def _sample_targets(
        self,
        *,
        start_x: float,
        start_y: float,
        provided_targets: list[TargetInstance] | None,
    ) -> list[TargetInstance]:
        if provided_targets is not None:
            return list(provided_targets)

        x_min, x_max = self.config.x_limits
        y_min, y_max = self.config.y_limits
        min_distance = max(self.config.target_detection_radius_m * 1.5, 2.0)
        targets: list[TargetInstance] = []
        for class_name in self.target_class_mix:
            chosen = None
            for _ in range(200):
                x = float(self.np_random.uniform(x_min, x_max))
                y = float(self.np_random.uniform(y_min, y_max))
                if math.hypot(x - start_x, y - start_y) < min_distance:
                    continue
                if any(math.hypot(x - target.x, y - target.y) < min_distance for target in targets):
                    continue
                chosen = TargetInstance(class_name=class_name, x=x, y=y)
                break
            if chosen is None:
                chosen = TargetInstance(
                    class_name=class_name,
                    x=float(self.np_random.uniform(x_min, x_max)),
                    y=float(self.np_random.uniform(y_min, y_max)),
                )
            targets.append(chosen)
        return targets

    def _make_obs(self) -> np.ndarray:
        return self.encoder.encode(
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            yaw=self.yaw,
            elapsed_s=float(self.elapsed_steps) * self.config.decision_period_s,
        )

    def _make_info(
        self,
        *,
        reward_new_coverage: float = 0.0,
        reward_target_bonus: float = 0.0,
        reward_success_bonus: float = 0.0,
        reward_step_penalty: float = 0.0,
        reward_boundary_penalty: float = 0.0,
        reward_timeout_penalty: float = 0.0,
        reward_stagnation_penalty: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "coverage_fraction": self.coverage.coverage_fraction,
            "found_targets": len(self.found_target_indices),
            "required_targets": self.required_target_count,
            "reward_new_coverage": reward_new_coverage,
            "reward_target_bonus": reward_target_bonus,
            "reward_success_bonus": reward_success_bonus,
            "reward_step_penalty": reward_step_penalty,
            "reward_boundary_penalty": reward_boundary_penalty,
            "reward_timeout_penalty": reward_timeout_penalty,
            "reward_stagnation_penalty": reward_stagnation_penalty,
            "targets": [
                {"class_name": target.class_name, "x": target.x, "y": target.y}
                for target in self.targets
            ],
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.coverage.reset()
        self.elapsed_steps = 0
        self.found_target_indices.clear()

        provided_start = options.get("start_state")
        if provided_start is None:
            self.x, self.y, self.yaw = self._sample_start_state()
        else:
            self.x = float(provided_start["x"])
            self.y = float(provided_start["y"])
            self.yaw = float(provided_start.get("yaw", 0.0))

        self.vx = 0.0
        self.vy = 0.0

        provided_targets = options.get("targets")
        parsed_targets = None
        if provided_targets is not None:
            parsed_targets = [
                TargetInstance(
                    class_name=str(target["class_name"]),
                    x=float(target["x"]),
                    y=float(target["y"]),
                )
                for target in provided_targets
            ]

        self.targets = self._sample_targets(
            start_x=self.x,
            start_y=self.y,
            provided_targets=parsed_targets,
        )
        self.coverage.mark_point(self.x, self.y)

        return self._make_obs(), self._make_info()

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_x = self.x
        prev_y = self.y

        next_x, next_y, next_yaw, overflow = apply_relative_action(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            action=action,
            config=self.config,
        )
        self.x = next_x
        self.y = next_y
        self.yaw = next_yaw
        self.vx = (self.x - prev_x) / max(self.config.decision_period_s, 1e-6)
        self.vy = (self.y - prev_y) / max(self.config.decision_period_s, 1e-6)
        self.elapsed_steps += 1

        coverage_delta = self.coverage.update_segment(prev_x, prev_y, self.x, self.y)
        reward_new_coverage = self.config.coverage_reward_scale * coverage_delta

        new_target_count = 0
        for index, target in enumerate(self.targets):
            if index in self.found_target_indices:
                continue
            if math.hypot(self.x - target.x, self.y - target.y) <= self.config.target_detection_radius_m:
                self.found_target_indices.add(index)
                new_target_count += 1

        terminated = (
            len(self.found_target_indices) >= self.required_target_count
            or self.coverage.coverage_fraction >= 1.0
        )
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated

        reward_target_bonus = new_target_count * self.config.first_target_found_bonus
        reward_success_bonus = self.config.success_bonus if terminated else 0.0
        reward_step_penalty = self.config.step_penalty
        reward_boundary_penalty = overflow * self.config.boundary_penalty
        reward_timeout_penalty = self.config.timeout_penalty if truncated else 0.0
        reward_stagnation_penalty = (
            self.config.stagnation_penalty
            if coverage_delta <= 0.0 and new_target_count == 0
            else 0.0
        )
        reward = (
            reward_new_coverage
            + reward_target_bonus
            + reward_success_bonus
            - reward_step_penalty
            - reward_boundary_penalty
            - reward_timeout_penalty
            - reward_stagnation_penalty
        )

        info = self._make_info(
            reward_new_coverage=reward_new_coverage,
            reward_target_bonus=reward_target_bonus,
            reward_success_bonus=reward_success_bonus,
            reward_step_penalty=reward_step_penalty,
            reward_boundary_penalty=reward_boundary_penalty,
            reward_timeout_penalty=reward_timeout_penalty,
            reward_stagnation_penalty=reward_stagnation_penalty,
        )
        info["success"] = terminated

        return self._make_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
