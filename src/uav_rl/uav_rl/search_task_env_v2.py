"""DQN-compatible search task with belief-map observations and discrete actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from uav_rl.actions import (
    ActionConfig,
    ACTION_NAMES,
    SearchAction,
    action_to_goal,
    generate_candidate_goals,
    slew_yaw,
)
from uav_rl.belief_map import BeliefMap, MapGeometry
from uav_rl.camera_model import ForwardCameraModel
from uav_rl.detection_memory import (
    ConfirmationConfig,
    DetectionMemory,
    DetectionObservation,
)
from uav_rl.observation_v2 import ObservationConfig, encode_observation
from uav_rl.reward import RewardConfig, SearchEvents, compute_search_reward
from uav_rl.rl_common import load_world_target_classes


@dataclass(frozen=True)
class TargetInstanceV2:
    class_name: str
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class SearchTaskConfigV2:
    width: float = 40.0
    height: float = 40.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    cell_size_m: float = 4.0
    patch_side: int = 11
    fixed_altitude_ned: float = -4.0
    scan_yaw_step_rad: float = math.radians(25.0)
    max_yaw_step_rad: float = math.radians(25.0)
    decision_period_s: float = 2.0
    search_speed_m_s: float = 2.0
    max_episode_steps: int = 256
    max_unproductive_scan_streak: int = 2
    required_target_count: int | None = None
    target_count: int | None = None
    world_path: str | None = None
    randomize_start: bool = True
    altitude_jitter_m: float = 0.0
    obstacle_count: int = 0
    detection_probability: float = 0.85
    investigate_detection_probability: float = 0.98
    false_positive_rate: float = 0.01
    detection_noise_m: float = 0.45
    min_target_spacing_m: float = 4.0
    camera: ForwardCameraModel = field(default_factory=ForwardCameraModel)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SearchTaskConfigV2":
        raw = dict(raw or {})
        camera = raw.pop("camera", None)
        confirmation = raw.pop("confirmation", None)
        reward = raw.pop("reward", None)
        if camera is not None:
            raw["camera"] = ForwardCameraModel(**dict(camera))
        if confirmation is not None:
            raw["confirmation"] = ConfirmationConfig(**dict(confirmation))
        if reward is not None:
            raw["reward"] = RewardConfig(**dict(reward))
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @property
    def geometry(self) -> MapGeometry:
        return MapGeometry(
            width=self.width,
            height=self.height,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            cell_size_m=self.cell_size_m,
        )

    @property
    def action_config(self) -> ActionConfig:
        return ActionConfig(
            cell_size_m=self.cell_size_m,
            fixed_altitude_ned=self.fixed_altitude_ned,
            scan_yaw_step_rad=self.scan_yaw_step_rad,
        )

    @property
    def observation_config(self) -> ObservationConfig:
        return ObservationConfig(
            patch_side=self.patch_side,
            top_k_detections=3,
            action_count=len(SearchAction),
            altitude_reference_m=max(abs(self.fixed_altitude_ned), 10.0),
            speed_reference_m_s=max(self.search_speed_m_s * 2.0, 1.0),
            max_episode_steps=self.max_episode_steps,
        )

    def required_targets(self, sampled_target_count: int) -> int:
        if self.required_target_count is not None:
            return max(1, int(self.required_target_count))
        return max(1, sampled_target_count)


class SearchTaskEnvV2(gym.Env):
    """Fast abstract environment for high-level UAV search decisions."""

    metadata = {"render_modes": []}

    def __init__(self, config: SearchTaskConfigV2 | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, SearchTaskConfigV2)
            else SearchTaskConfigV2.from_mapping(config)
        )
        self.geometry = self.config.geometry
        self.belief = BeliefMap(self.geometry)
        self.memory = DetectionMemory(self.config.confirmation)
        self.camera = self.config.camera
        self.obs_config = self.config.observation_config

        self.action_space = spaces.Discrete(len(SearchAction))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self.obs_config.observation_size,),
            dtype=np.float32,
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.z = self.config.fixed_altitude_ned
        self.vx = 0.0
        self.vy = 0.0
        self.step_index = 0
        self.targets: list[TargetInstanceV2] = []
        self.confirmed_truth_ids: set[int] = set()
        self.obstacle_cells: set[tuple[int, int]] = set()
        self.last_action_name = ""
        self.last_action_int = -1
        self.last_move_action_int = -1
        self.previous_cell: tuple[int, int] | None = None
        self.recent_cells: list[tuple[int, int]] = []
        self.backtrack_count = 0
        self.shielded_action_count = 0
        self.unproductive_scan_streak = 0
        self.scan_disabled_count = 0

    def _sample_start_state(self) -> tuple[float, float, float, float]:
        if not self.config.randomize_start:
            return (
                self.config.origin_x,
                self.config.origin_y,
                0.0,
                self.config.fixed_altitude_ned,
            )
        x_min, x_max = self.geometry.x_limits
        y_min, y_max = self.geometry.y_limits
        margin = min(self.config.cell_size_m, self.config.width / 4.0, self.config.height / 4.0)
        x = float(self.np_random.uniform(x_min + margin, x_max - margin))
        y = float(self.np_random.uniform(y_min + margin, y_max - margin))
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        z = self.config.fixed_altitude_ned + float(
            self.np_random.uniform(-self.config.altitude_jitter_m, self.config.altitude_jitter_m)
        )
        return x, y, yaw, z

    def _target_classes(self) -> list[str]:
        classes = load_world_target_classes(self.config.world_path)
        if self.config.target_count is not None:
            desired = max(1, int(self.config.target_count))
            if len(classes) >= desired:
                return classes[:desired]
            classes = classes + ["person"] * (desired - len(classes))
        return classes

    def _sample_targets(
        self,
        *,
        start_x: float,
        start_y: float,
        provided_targets: list[TargetInstanceV2] | None,
    ) -> list[TargetInstanceV2]:
        if provided_targets is not None:
            return list(provided_targets)

        x_min, x_max = self.geometry.x_limits
        y_min, y_max = self.geometry.y_limits
        min_distance = max(float(self.config.min_target_spacing_m), self.config.cell_size_m)
        targets: list[TargetInstanceV2] = []
        for class_name in self._target_classes():
            chosen = None
            for _ in range(300):
                x = float(self.np_random.uniform(x_min, x_max))
                y = float(self.np_random.uniform(y_min, y_max))
                if math.hypot(x - start_x, y - start_y) < min_distance:
                    continue
                if any(math.hypot(x - target.x, y - target.y) < min_distance for target in targets):
                    continue
                chosen = TargetInstanceV2(class_name=class_name, x=x, y=y)
                break
            if chosen is None:
                chosen = TargetInstanceV2(
                    class_name=class_name,
                    x=float(self.np_random.uniform(x_min, x_max)),
                    y=float(self.np_random.uniform(y_min, y_max)),
                )
            targets.append(chosen)
        return targets

    def _sample_obstacles(self) -> set[tuple[int, int]]:
        obstacle_count = max(0, int(self.config.obstacle_count))
        obstacles: set[tuple[int, int]] = set()
        start_cell = self.geometry.world_to_grid(self.x, self.y)
        target_cells = {self.geometry.world_to_grid(target.x, target.y) for target in self.targets}
        attempts = 0
        while len(obstacles) < obstacle_count and attempts < obstacle_count * 50 + 100:
            attempts += 1
            row = int(self.np_random.integers(0, self.geometry.rows))
            col = int(self.np_random.integers(0, self.geometry.cols))
            cell = (row, col)
            if cell == start_cell or cell in target_cells:
                continue
            obstacles.add(cell)
            ox, oy = self.geometry.grid_to_world(row, col)
            self.belief.mark_obstacle(ox, oy, confidence=1.0)
        return obstacles

    def _make_obs(self) -> np.ndarray:
        return encode_observation(
            belief=self.belief,
            memory=self.memory,
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            altitude_ned_z=self.z,
            vx=self.vx,
            vy=self.vy,
            step=self.step_index,
            config=self.obs_config,
            last_action=self.last_action_int,
        )

    def _visible_cells(self) -> list[tuple[int, int]]:
        return self.camera.visible_cells(
            geometry=self.geometry,
            drone_x=self.x,
            drone_y=self.y,
            drone_yaw=self.yaw,
            altitude_ned_z=self.z,
        )

    def _make_detection_observations(self, *, investigating: bool) -> list[DetectionObservation]:
        observations: list[DetectionObservation] = []
        detection_probability = (
            self.config.investigate_detection_probability
            if investigating
            else self.config.detection_probability
        )
        for truth_id, target in enumerate(self.targets):
            if truth_id in self.confirmed_truth_ids:
                continue
            visible = self.camera.is_ground_point_visible(
                drone_x=self.x,
                drone_y=self.y,
                drone_yaw=self.yaw,
                altitude_ned_z=self.z,
                point_x=target.x,
                point_y=target.y,
            )
            if not visible or self.np_random.random() > detection_probability:
                continue
            noise = max(0.0, float(self.config.detection_noise_m))
            obs_x = target.x + float(self.np_random.normal(0.0, noise))
            obs_y = target.y + float(self.np_random.normal(0.0, noise))
            confidence = float(self.np_random.uniform(0.62, 0.95))
            observations.append(
                DetectionObservation(
                    class_name=target.class_name,
                    confidence=confidence,
                    x=obs_x,
                    y=obs_y,
                    z=target.z,
                    truth_id=truth_id,
                )
            )

        if self.config.false_positive_rate > 0.0 and self.np_random.random() < self.config.false_positive_rate:
            visible_cells = self._visible_cells()
            if visible_cells:
                row, col = visible_cells[int(self.np_random.integers(0, len(visible_cells)))]
                fp_x, fp_y = self.geometry.grid_to_world(row, col)
                observations.append(
                    DetectionObservation(
                        class_name="person",
                        confidence=float(self.np_random.uniform(0.35, 0.75)),
                        x=fp_x,
                        y=fp_y,
                        z=0.0,
                        truth_id=None,
                    )
                )
        return observations

    def _observe(self, *, investigating: bool = False) -> tuple[int, float, int, int]:
        visible_update = self.belief.mark_visible(self._visible_cells())
        observations = self._make_detection_observations(investigating=investigating)
        _new_tracks, updated_existing = self.memory.update(
            observations,
            step=self.step_index,
            drone_x=self.x,
            drone_y=self.y,
            drone_yaw=self.yaw,
        )
        for obs in observations:
            self.belief.update_detection(obs.x, obs.y, obs.confidence)

        newly_confirmed = 0
        false_confirmations = 0
        for track in self.memory.confirm_ready_tracks():
            tx, ty, _tz = track.filtered_position
            self.belief.confirm_cell(tx, ty)
            truth_id = track.dominant_truth_id
            if truth_id is None or truth_id in self.confirmed_truth_ids:
                false_confirmations += 1
            else:
                self.confirmed_truth_ids.add(truth_id)
                newly_confirmed += 1

        return (
            visible_update.new_observed_cells,
            visible_update.uncertainty_reduction,
            updated_existing,
            newly_confirmed - false_confirmations,
        )

    def _parse_targets_option(self, raw_targets: Any) -> list[TargetInstanceV2] | None:
        if raw_targets is None:
            return None
        return [
            TargetInstanceV2(
                class_name=str(target.get("class_name", "person")),
                x=float(target["x"]),
                y=float(target["y"]),
                z=float(target.get("z", 0.0)),
            )
            for target in raw_targets
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.belief.reset()
        self.memory.reset()
        self.step_index = 0
        self.confirmed_truth_ids.clear()
        self.last_action_name = ""
        self.last_action_int = -1
        self.last_move_action_int = -1
        self.previous_cell = None
        self.recent_cells = []
        self.backtrack_count = 0
        self.shielded_action_count = 0
        self.unproductive_scan_streak = 0
        self.scan_disabled_count = 0
        self.vx = 0.0
        self.vy = 0.0

        provided_start = options.get("start_state")
        if provided_start is None:
            self.x, self.y, self.yaw, self.z = self._sample_start_state()
        else:
            self.x = float(provided_start["x"])
            self.y = float(provided_start["y"])
            self.yaw = float(provided_start.get("yaw", 0.0))
            self.z = float(provided_start.get("z", self.config.fixed_altitude_ned))

        self.targets = self._sample_targets(
            start_x=self.x,
            start_y=self.y,
            provided_targets=self._parse_targets_option(options.get("targets")),
        )
        self.obstacle_cells = self._sample_obstacles()
        self.recent_cells = [self.geometry.world_to_grid(self.x, self.y)]
        self._observe(investigating=False)
        return self._make_obs(), self._make_info()

    def step(self, action: int):
        requested_action_int = int(np.asarray(action).item())
        action_int = requested_action_int
        prev_x = self.x
        prev_y = self.y
        start_cell = self.geometry.world_to_grid(prev_x, prev_y)
        scan_allowed = self._scan_allowed()
        best_track = self.memory.best_track()
        goal = action_to_goal(
            action=action_int,
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            geometry=self.geometry,
            config=self.config.action_config,
            best_track=best_track,
            belief=self.belief,
            memory=self.memory,
            camera=self.camera,
            scan_allowed=scan_allowed,
            recent_cells=self.recent_cells,
        )
        substituted_candidate = "_substituted_for_" in goal.name or not goal.valid
        if substituted_candidate:
            self.shielded_action_count += 1
            if requested_action_int == int(SearchAction.HOVER_SCAN) and not scan_allowed:
                self.scan_disabled_count += 1
        requested_goal_cell = self.geometry.world_to_grid(goal.x, goal.y)
        shielded_backtrack = (
            self.previous_cell is not None
            and requested_goal_cell == self.previous_cell
            and requested_goal_cell != start_cell
        )
        if shielded_backtrack:
            self.shielded_action_count += 1
            action_int = int(SearchAction.ESCAPE_STUCK)
            goal = action_to_goal(
                action=action_int,
                x=self.x,
                y=self.y,
                yaw=self.yaw,
                geometry=self.geometry,
                config=self.config.action_config,
                best_track=best_track,
                belief=self.belief,
                memory=self.memory,
                camera=self.camera,
                scan_allowed=scan_allowed,
                recent_cells=self.recent_cells,
            )
        shielded_action = substituted_candidate or shielded_backtrack
        selected_scan = (
            requested_action_int == int(SearchAction.HOVER_SCAN)
            and goal.name == ACTION_NAMES[SearchAction.HOVER_SCAN]
        )
        self.x = goal.x
        self.y = goal.y
        if self.config.max_yaw_step_rad > 0.0:
            self.yaw = slew_yaw(self.yaw, goal.yaw, self.config.max_yaw_step_rad)
        else:
            self.yaw = goal.yaw
        self.z = goal.z
        self.vx = (self.x - prev_x) / max(self.config.decision_period_s, 1e-6)
        self.vy = (self.y - prev_y) / max(self.config.decision_period_s, 1e-6)
        self.step_index += 1
        self.last_action_name = goal.name
        self.belief.age(self.config.decision_period_s)

        current_cell = self.geometry.world_to_grid(self.x, self.y)
        action_is_move = current_cell != start_cell
        immediate_backtrack = (
            action_is_move
            and self.previous_cell is not None
            and current_cell == self.previous_cell
        )
        if immediate_backtrack:
            self.backtrack_count += 1
        collision = current_cell in self.obstacle_cells
        out_of_bounds = goal.overflow_m > 1e-6
        min_ground, max_ground = self.camera.ground_visibility_band(self.z)
        unsafe_altitude = min_ground >= max_ground

        investigating = "detection_confirm" in goal.name
        had_track_for_investigation = investigating and best_track is not None
        prev_unconfirmed_tracks = self.memory.unconfirmed_count()
        new_cells, uncertainty_reduction, useful_reobs, confirmation_delta = self._observe(
            investigating=investigating
        )
        new_confirmed = max(0, confirmation_delta)
        false_confirmation = confirmation_delta < 0
        false_investigation = investigating and (
            not had_track_for_investigation
            or false_confirmation
            or (best_track is not None and best_track.dominant_truth_id is None)
        )
        if investigating and best_track is not None:
            best_track.investigated = True

        new_track_created = self.memory.unconfirmed_count() > prev_unconfirmed_tracks
        productive_scan = selected_scan and (
            new_cells > 0 or useful_reobs > 0 or new_confirmed > 0 or new_track_created
        )
        if selected_scan:
            if productive_scan:
                self.unproductive_scan_streak = 0
            else:
                self.unproductive_scan_streak += 1
        elif action_is_move:
            self.unproductive_scan_streak = 0

        useless_revisit = (
            new_cells == 0
            and useful_reobs == 0
            and new_confirmed == 0
            and not investigating
        )
        required_targets = self.config.required_targets(len(self.targets))
        mission_success = len(self.confirmed_truth_ids) >= required_targets
        terminated = mission_success or collision or unsafe_altitude
        truncated = self.step_index >= self.config.max_episode_steps and not terminated
        missed_required_targets = (
            max(0, required_targets - len(self.confirmed_truth_ids))
            if truncated
            else 0
        )

        events = SearchEvents(
            new_observed_cells=new_cells,
            uncertainty_reduction=uncertainty_reduction,
            useful_reobservations=useful_reobs,
            new_confirmed_victims=new_confirmed,
            useless_revisit=useless_revisit,
            false_investigation=false_investigation,
            immediate_backtrack=immediate_backtrack,
            reverse_move=False,
            shielded_action=shielded_action,
            collision=collision,
            unsafe_altitude=unsafe_altitude,
            out_of_bounds=out_of_bounds,
            mission_success=mission_success,
            missed_required_targets=missed_required_targets,
        )
        reward = compute_search_reward(events, self.config.reward)

        if current_cell != start_cell:
            self.previous_cell = start_cell
            if action_is_move:
                self.last_move_action_int = action_int
        self.last_action_int = action_int
        self.recent_cells.append(current_cell)
        self.recent_cells = self.recent_cells[-6:]

        info = self._make_info(reward_breakdown=reward.as_dict(), events=events)
        info["success"] = mission_success
        return self._make_obs(), reward.total, terminated, truncated, info

    def _make_info(
        self,
        *,
        reward_breakdown: dict[str, float] | None = None,
        events: SearchEvents | None = None,
    ) -> dict[str, Any]:
        return {
            "coverage_fraction": self.belief.coverage_fraction,
            "mean_uncertainty": self.belief.mean_uncertainty,
            "confirmed_targets": len(self.confirmed_truth_ids),
            "required_targets": self.config.required_targets(len(self.targets)),
            "unconfirmed_tracks": self.memory.unconfirmed_count(),
            "last_action": self.last_action_name,
            "last_action_int": self.last_action_int,
            "backtrack_count": self.backtrack_count,
            "shielded_action_count": self.shielded_action_count,
            "unproductive_scan_streak": self.unproductive_scan_streak,
            "scan_disabled_count": self.scan_disabled_count,
            "reward": reward_breakdown or {},
            "events": asdict(events) if events is not None else {},
            "targets": [
                {"class_name": target.class_name, "x": target.x, "y": target.y, "z": target.z}
                for target in self.targets
            ],
        }

    def action_toward(self, target: tuple[float, float] | None) -> int:
        if target is None:
            return int(SearchAction.HIGH_INFO_BEST)
        candidates = generate_candidate_goals(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            belief=self.belief,
            memory=self.memory,
            camera=self.camera,
            config=self.config.action_config,
            scan_allowed=self._scan_allowed(),
            recent_cells=self.recent_cells,
        )
        ranked = [
            (idx, math.hypot(goal.x - target[0], goal.y - target[1]))
            for idx, goal in enumerate(candidates)
            if goal.valid and idx != int(SearchAction.HOVER_SCAN)
        ]
        if not ranked:
            return int(SearchAction.HIGH_INFO_BEST)
        return min(ranked, key=lambda item: item[1])[0]

    def greedy_frontier_action(self) -> int:
        return int(SearchAction.FRONTIER_BEST)

    def greedy_hybrid_action(self) -> int:
        best_track = self.memory.best_track()
        if best_track is not None and best_track.mean_confidence >= 0.55:
            return int(SearchAction.DETECTION_CONFIRM_BEST)
        best_cell = self.belief.best_victim_cell(self.x, self.y)
        if best_cell is not None:
            return self.action_toward(best_cell)
        return self.greedy_frontier_action()

    def lawnmower_action(self) -> int:
        row, _col = self.geometry.world_to_grid(self.x, self.y)
        x_min, x_max = self.geometry.x_limits
        y_min, y_max = self.geometry.y_limits
        eastbound = row % 2 == 0
        if eastbound and self.y < y_max - self.config.cell_size_m:
            return self.action_toward((self.x, self.y + self.config.cell_size_m))
        if not eastbound and self.y > y_min + self.config.cell_size_m:
            return self.action_toward((self.x, self.y - self.config.cell_size_m))
        if self.x < x_max - self.config.cell_size_m:
            return self.action_toward((self.x + self.config.cell_size_m, self.y))
        return int(SearchAction.RETURN_CENTER)

    def _scan_allowed(self) -> bool:
        return self.unproductive_scan_streak < max(
            0,
            int(self.config.max_unproductive_scan_streak),
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
