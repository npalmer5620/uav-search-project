"""Gymnasium environment for lean PyFlyt UAV triage search."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from uav_triage_rl.bbox import BBoxPerception, BBoxSimConfig
from uav_triage_rl.perception import (
    CameraConfig,
    DetectionMemory,
    PerceptionConfig,
    SyntheticPerception,
)
from uav_triage_rl.policy import (
    ACTION_NAMES,
    BELIEF_CHANNELS,
    CHANNEL_OBSERVED,
    CHANNEL_OBSTACLE,
    CHANNEL_UNCERTAINTY,
    CHANNEL_VICTIM_SCORE,
    ActionConfig,
    MapGeometry,
    SearchAction,
    lawnmower_action,
    select_goal,
)
from uav_triage_rl.rewards import RewardConfig, compute_reward
from uav_triage_rl.world import (
    KinematicRuntime,
    PyFlytRuntime,
    WorldConfig,
    WorldLayout,
    sample_layout,
)


class TriageSearchEnv(gym.Env):
    """Discrete macro-action UAV search environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        raw = dict(config or {})
        env_raw = dict(raw.get("env", raw) or {})
        self.world_config = WorldConfig.from_mapping(env_raw)
        self.camera_config = CameraConfig.from_mapping(raw.get("camera"))
        self.perception_config = PerceptionConfig.from_mapping(raw.get("perception"))
        bbox_raw = dict(raw.get("perception", {}) or {}).get("bbox")
        self.bbox_config = BBoxSimConfig.from_mapping(bbox_raw)
        self.action_config = ActionConfig.from_mapping(raw.get("actions"))
        self.reward_config = RewardConfig.from_mapping(raw.get("rewards"))
        self.render_mode = render_mode

        self.geometry = MapGeometry(
            width_m=self.world_config.width_m,
            height_m=self.world_config.height_m,
            cell_size_m=self.world_config.cell_size_m,
        )
        if self.perception_config.mode == "bbox":
            self.perception = BBoxPerception(
                camera=self.camera_config,
                perception=self.perception_config,
                config=self.bbox_config,
            )
        else:
            self.perception = SyntheticPerception(self.camera_config, self.perception_config)
        self.memory = DetectionMemory(self.perception_config)

        self.action_space = spaces.Discrete(len(SearchAction))
        self.patch_side = int(env_raw.get("patch_side", 11))
        self.top_k_tracks = int(env_raw.get("top_k_tracks", 3))
        self.state_size = 9
        self.observation_size = (
            self.patch_side * self.patch_side * BELIEF_CHANNELS
            + self.state_size
            + len(SearchAction)
            + self.top_k_tracks * 5
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

        self.runtime: KinematicRuntime | PyFlytRuntime | None = None
        self.layout: WorldLayout | None = None
        self.belief = np.zeros((self.geometry.rows, self.geometry.cols, BELIEF_CHANNELS), dtype=np.float32)
        self.step_index = 0
        self.last_action = -1
        self.x = 0.0
        self.y = 0.0
        self.z = self.world_config.search_altitude_m
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.recent_cells: list[tuple[int, int]] = []
        self.last_bboxes: list[dict[str, Any]] = []

    def _reset_belief(self) -> None:
        self.belief.fill(0.0)
        self.belief[:, :, CHANNEL_UNCERTAINTY] = 1.0

    def _sample_start(self) -> tuple[float, float, float, float]:
        if not self.world_config.randomize_start:
            return 0.0, 0.0, 0.0, self.world_config.search_altitude_m
        margin = max(1.0, self.world_config.cell_size_m)
        x = float(self.np_random.uniform(self.geometry.x_limits[0] + margin, self.geometry.x_limits[1] - margin))
        y = float(self.np_random.uniform(self.geometry.y_limits[0] + margin, self.geometry.y_limits[1] - margin))
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        return x, y, yaw, self.world_config.search_altitude_m

    def _make_runtime(self, *, render: bool, seed: int | None) -> KinematicRuntime | PyFlytRuntime:
        assert self.layout is not None
        if self.world_config.backend == "kinematic":
            return KinematicRuntime(self.world_config, self.layout, render=render)
        return PyFlytRuntime(
            self.world_config,
            self.layout,
            camera_config=self.camera_config,
            render=render,
            seed=seed,
        )

    def _mark_obstacles(self) -> None:
        assert self.layout is not None
        for obstacle in self.layout.obstacles:
            row, col = self.geometry.world_to_grid(obstacle.x, obstacle.y)
            self.belief[row, col, CHANNEL_OBSTACLE] = 1.0

    def _observe(self, *, investigating: bool) -> dict[str, Any]:
        assert self.layout is not None
        visible = self.perception.visible_cells(
            geometry=self.geometry,
            drone_x=self.x,
            drone_y=self.y,
            drone_z=self.z,
            drone_yaw=self.yaw,
        )
        new_cells = 0
        uncertainty_reduction = 0.0
        for row, col in visible:
            old_observed = self.belief[row, col, CHANNEL_OBSERVED]
            old_uncertainty = self.belief[row, col, CHANNEL_UNCERTAINTY]
            if old_observed <= 0.0:
                new_cells += 1
            self.belief[row, col, CHANNEL_OBSERVED] = 1.0
            self.belief[row, col, CHANNEL_UNCERTAINTY] = max(0.0, old_uncertainty - 0.35)
            uncertainty_reduction += max(0.0, float(old_uncertainty - self.belief[row, col, CHANNEL_UNCERTAINTY]))

        observations = self.perception.detect(
            self.layout.victims,
            drone_x=self.x,
            drone_y=self.y,
            drone_z=self.z,
            drone_yaw=self.yaw,
            rng=self.np_random,
            investigating=investigating,
            visible_cells=visible,
            geometry=self.geometry,
        )
        self.last_bboxes = [
            bbox.as_dict()
            for bbox in getattr(self.perception, "last_bboxes", [])
        ]
        for obs in observations:
            row, col = self.geometry.world_to_grid(obs.x, obs.y)
            self.belief[row, col, CHANNEL_VICTIM_SCORE] = min(
                1.0,
                self.belief[row, col, CHANNEL_VICTIM_SCORE] + obs.confidence * 0.25,
            )
        detection_updates, new_confirmed = self.memory.update(observations, step=self.step_index)
        return {
            "new_observed_cells": new_cells,
            "uncertainty_reduction": uncertainty_reduction,
            "detections": len(observations),
            "bbox_detections": len(self.last_bboxes),
            "bboxes": list(self.last_bboxes),
            "detection_updates": detection_updates,
            "new_confirmed_victims": new_confirmed,
        }

    def _local_patch(self) -> np.ndarray:
        row, col = self.geometry.world_to_grid(self.x, self.y)
        radius = self.patch_side // 2
        padded = np.pad(
            self.belief,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        row += radius
        col += radius
        patch = padded[row - radius: row + radius + 1, col - radius: col + radius + 1, :]
        return patch.reshape(-1).astype(np.float32, copy=False)

    def _make_obs(self) -> np.ndarray:
        half_w = max(self.world_config.width_m / 2.0, 1e-6)
        half_h = max(self.world_config.height_m / 2.0, 1e-6)
        speed_ref = max(self.world_config.max_speed_m_s, 1.0)
        coverage = float(np.mean(self.belief[:, :, CHANNEL_OBSERVED]))
        state = np.array(
            [
                self.x / half_w,
                self.y / half_h,
                self.z / max(self.world_config.search_altitude_m, 1.0),
                np.cos(self.yaw),
                np.sin(self.yaw),
                self.vx / speed_ref,
                self.vy / speed_ref,
                coverage,
                self.step_index / max(1, self.world_config.max_episode_steps),
            ],
            dtype=np.float32,
        )
        last_action = np.zeros(len(SearchAction), dtype=np.float32)
        if 0 <= self.last_action < len(SearchAction):
            last_action[self.last_action] = 1.0

        track_features: list[float] = []
        for track in self.memory.top_tracks(self.top_k_tracks):
            track_features.extend(
                [
                    (track.x - self.x) / max(self.world_config.width_m, 1.0),
                    (track.y - self.y) / max(self.world_config.height_m, 1.0),
                    track.mean_confidence,
                    min(1.0, (self.step_index - track.last_seen_step) / max(1, self.perception_config.track_timeout_steps)),
                    1.0 if track.confirmed else 0.0,
                ]
            )
        while len(track_features) < self.top_k_tracks * 5:
            track_features.append(0.0)

        return np.concatenate(
            [
                self._local_patch(),
                state,
                last_action,
                np.asarray(track_features, dtype=np.float32),
            ]
        ).astype(np.float32)

    def _make_info(self, events: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.layout is not None
        required = min(self.world_config.required_victim_count, len(self.layout.victims))
        confirmed = len(self.memory.confirmed_truth_ids)
        return {
            "step": self.step_index,
            "coverage_fraction": float(np.mean(self.belief[:, :, CHANNEL_OBSERVED])),
            "confirmed_victims": confirmed,
            "required_victims": required,
            "success": confirmed >= required,
            "last_action": ACTION_NAMES.get(SearchAction(self.last_action), "") if self.last_action >= 0 else "",
            "events": dict(events or {}),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.runtime is not None:
            self.runtime.close()
        self._reset_belief()
        self.memory.reset()
        if hasattr(self.perception, "reset"):
            self.perception.reset()
        self.step_index = 0
        self.last_action = -1
        self.last_bboxes = []
        options = options or {}

        if "start_state" in options:
            start = options["start_state"]
            start_state = (
                float(start.get("x", 0.0)),
                float(start.get("y", 0.0)),
                float(start.get("yaw", 0.0)),
                float(start.get("z", self.world_config.search_altitude_m)),
            )
        else:
            start_state = self._sample_start()
        self.x, self.y, self.yaw, self.z = start_state
        self.vx = 0.0
        self.vy = 0.0

        self.layout = sample_layout(
            self.world_config,
            self.np_random,
            start_xy=(self.x, self.y),
        )
        self._mark_obstacles()
        self.runtime = self._make_runtime(render=self.render_mode == "human", seed=seed)
        self.runtime.start(start_state)
        self.recent_cells = [self.geometry.world_to_grid(self.x, self.y)]
        self._observe(investigating=False)
        return self._make_obs(), self._make_info()

    def step(self, action: int):
        action_int = int(np.asarray(action).item())
        self.last_action = action_int
        prev_cell = self.geometry.world_to_grid(self.x, self.y)
        tracks = self.memory.top_tracks(limit=5)
        goal = select_goal(
            action=action_int,
            geometry=self.geometry,
            belief=self.belief,
            tracks=tracks,
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            config=self.action_config,
        )
        out_of_bounds = goal.overflow_m > 1e-6
        assert self.runtime is not None
        macro_steps = max(1, int(round(self.world_config.decision_period_s * self.world_config.control_hz)))
        self.runtime.set_goal(goal.x, goal.y, goal.yaw, self.world_config.search_altitude_m)
        self.runtime.step_many(macro_steps)
        self.x, self.y, self.z, self.yaw, self.vx, self.vy = self.runtime.state()
        self.step_index += 1

        current_cell = self.geometry.world_to_grid(self.x, self.y)
        self.recent_cells.append(current_cell)
        self.recent_cells = self.recent_cells[-8:]
        collision = bool(self.belief[current_cell[0], current_cell[1], CHANNEL_OBSTACLE] > 0.5)
        revisit = current_cell == prev_cell and action_int != int(SearchAction.HOVER_SCAN)
        investigating = action_int in (int(SearchAction.INVESTIGATE_BEST), int(SearchAction.INVESTIGATE_OFFSET))
        events = self._observe(investigating=investigating)
        required = min(self.world_config.required_victim_count, len(self.layout.victims) if self.layout else 0)
        success = len(self.memory.confirmed_truth_ids) >= required
        truncated = self.step_index >= self.world_config.max_episode_steps
        events.update(
            {
                "collision": collision,
                "out_of_bounds": out_of_bounds,
                "revisit": revisit,
                "success": success,
                "truncated": truncated,
                "confirmed_victims": len(self.memory.confirmed_truth_ids),
                "required_victims": required,
                "goal_name": goal.name,
            }
        )
        reward = compute_reward(events, self.reward_config)
        terminated = bool(success)
        return self._make_obs(), reward, terminated, bool(truncated and not terminated), self._make_info(events)

    def render(self):
        if self.runtime is not None:
            frame = self.runtime.camera_image()
            if frame is not None:
                return frame
        return self._render_topdown()

    def _render_topdown(self) -> np.ndarray:
        assert self.layout is not None
        size = 640
        image = np.full((size, size, 3), 235, dtype=np.uint8)

        def xy_to_px(x: float, y: float) -> tuple[int, int]:
            px = int((x - self.geometry.x_limits[0]) / self.world_config.width_m * (size - 1))
            py = int((self.geometry.y_limits[1] - y) / self.world_config.height_m * (size - 1))
            return int(np.clip(px, 0, size - 1)), int(np.clip(py, 0, size - 1))

        for obstacle in self.layout.obstacles:
            px, py = xy_to_px(obstacle.x, obstacle.y)
            image[max(0, py - 8): min(size, py + 8), max(0, px - 8): min(size, px + 8)] = (80, 80, 80)
        for victim in self.layout.victims:
            px, py = xy_to_px(victim.x, victim.y)
            color = (220, 20, 20) if victim.triage == "critical" else ((220, 180, 20) if victim.triage == "delayed" else (40, 160, 60))
            image[max(0, py - 5): min(size, py + 5), max(0, px - 12): min(size, px + 12)] = color
        px, py = xy_to_px(self.x, self.y)
        image[max(0, py - 6): min(size, py + 6), max(0, px - 6): min(size, px + 6)] = (20, 80, 220)
        return image

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

    def lawnmower_action(self) -> int:
        return lawnmower_action(self.step_index)
