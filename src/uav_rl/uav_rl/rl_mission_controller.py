#!/usr/bin/env python3
"""ROS 2 mission controller that uses a PPO policy for SEARCH behavior."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException

from uav_planning.mission_base import MissionControllerBase, Phase
from uav_rl.rl_common import (
    CoverageMap,
    ObservationEncoder,
    SearchTaskConfig,
    apply_relative_action,
    load_vecnormalize_for_inference,
    resolve_repo_path,
)


class RLMissionController(MissionControllerBase):
    """Mission controller that replaces the grid sweep with PPO search steps."""

    def __init__(self) -> None:
        self._model = None
        self._vec_normalize = None
        self._coverage_map: CoverageMap | None = None
        self._observation_encoder: ObservationEncoder | None = None
        self._policy_config: SearchTaskConfig | None = None
        self._target_xyyaw: tuple[float, float, float] | None = None
        self._last_pose_xy: tuple[float, float] | None = None
        self._coverage_initialized = False
        self._decision_interval_ticks = 1
        self._last_decision_tick = -1
        self._last_action = np.zeros(3, dtype=np.float32)
        self._last_overflow = 0.0

        super().__init__(
            "rl_mission_controller",
            extra_parameter_declarations=self._declare_rl_parameters,
        )
        self.get_logger().info(
            "RL search policy enabled "
            f"(model={self.rl_model_path}, vecnormalize={self.rl_vecnormalize_path})"
        )

    def _declare_rl_parameters(self) -> None:
        self.declare_parameter("rl.model_path", "artifacts/rl/search_policy/model.zip")
        self.declare_parameter(
            "rl.vecnormalize_path",
            "artifacts/rl/search_policy/vecnormalize.pkl",
        )
        self.declare_parameter("rl.decision_period_s", 0.5)
        self.declare_parameter("rl.max_step_xy_m", 4.0)
        self.declare_parameter("rl.coverage_grid_side", 16)

    def _init_search_controller(self) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "stable_baselines3 is required for rl_mission_controller"
            ) from exc

        self.rl_model_path = str(self.get_parameter("rl.model_path").value).strip()
        self.rl_vecnormalize_path = str(
            self.get_parameter("rl.vecnormalize_path").value
        ).strip()
        self.rl_decision_period_s = max(
            0.1, float(self.get_parameter("rl.decision_period_s").value)
        )
        self.rl_max_step_xy_m = max(
            0.1, float(self.get_parameter("rl.max_step_xy_m").value)
        )
        self.rl_coverage_grid_side = max(
            1, int(self.get_parameter("rl.coverage_grid_side").value)
        )

        self._policy_config = SearchTaskConfig(
            width=self.search_width,
            height=self.search_height,
            origin_x=self.search_origin_x,
            origin_y=self.search_origin_y,
            coverage_grid_side=self.rl_coverage_grid_side,
            max_step_xy_m=self.rl_max_step_xy_m,
            decision_period_s=self.rl_decision_period_s,
            search_speed=self.search_speed,
        )
        self._coverage_map = CoverageMap(
            self.rl_coverage_grid_side,
            self._policy_config,
        )
        self._observation_encoder = ObservationEncoder(
            self._policy_config,
            self._coverage_map,
            decision_period_s=self.rl_decision_period_s,
        )
        self._decision_interval_ticks = max(
            1,
            int(round(self.rl_decision_period_s / max(self.dt, 1e-6))),
        )

        model_path = resolve_repo_path(self.rl_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"RL model not found: {model_path}"
            )
        self._model = PPO.load(str(model_path))
        self._vec_normalize = load_vecnormalize_for_inference(
            self.rl_vecnormalize_path,
            self._policy_config,
        )
        if self._vec_normalize is None:
            self.get_logger().warn(
                "VecNormalize stats not found; using raw observations for inference"
            )

    def _on_search_enter(self) -> None:
        current_xy = (float(self.local_pos.x), float(self.local_pos.y))
        if self._coverage_map is not None and not self._coverage_initialized:
            self._coverage_map.reset()
            self._coverage_map.mark_point(current_xy[0], current_xy[1])
            self._coverage_initialized = True
        self._last_pose_xy = current_xy
        self._target_xyyaw = None
        self._last_decision_tick = self.tick - self._decision_interval_ticks

    def _on_search_exit(self) -> None:
        self._last_pose_xy = None
        self._target_xyyaw = None

    def _search_state_detail(self) -> str | None:
        if self._coverage_map is None:
            return None
        return f"{self._coverage_map.coverage_fraction:.0%}"

    def _predict_action(self, obs: np.ndarray) -> np.ndarray:
        inference_obs = obs
        if self._vec_normalize is not None:
            inference_obs = self._vec_normalize.normalize_obs(obs.reshape(1, -1)).squeeze(0)
        action, _ = self._model.predict(inference_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def _search_step(self) -> None:
        if self._coverage_map is None or self._observation_encoder is None:
            raise RuntimeError("RL search policy was not initialized")

        current_xy = (float(self.local_pos.x), float(self.local_pos.y))
        if self._last_pose_xy is None:
            self._last_pose_xy = current_xy
        reward_coverage = self._coverage_map.update_segment(
            self._last_pose_xy[0],
            self._last_pose_xy[1],
            current_xy[0],
            current_xy[1],
        )
        self._last_pose_xy = current_xy

        if self._coverage_map.coverage_fraction >= 1.0:
            self._publish_event("RL search coverage complete. Returning to launch.")
            self.phase = Phase.RTH
            return

        need_decision = (
            self._target_xyyaw is None
            or (self.tick - self._last_decision_tick) >= self._decision_interval_ticks
        )
        if need_decision:
            obs = self._observation_encoder.encode(
                x=self.local_pos.x,
                y=self.local_pos.y,
                vx=self.local_pos.vx if self.local_pos.v_xy_valid else 0.0,
                vy=self.local_pos.vy if self.local_pos.v_xy_valid else 0.0,
                yaw=self._current_yaw(),
                elapsed_s=self.search_elapsed_s,
            )
            self._last_action = self._predict_action(obs)
            target_x, target_y, target_yaw, self._last_overflow = apply_relative_action(
                x=self.local_pos.x,
                y=self.local_pos.y,
                yaw=self._current_yaw(),
                action=self._last_action,
                config=self._policy_config,
            )
            self._target_xyyaw = (target_x, target_y, target_yaw)
            self._last_decision_tick = self.tick

        if self._target_xyyaw is None:
            target_x = self.local_pos.x
            target_y = self.local_pos.y
            target_yaw = self._current_yaw()
        else:
            target_x, target_y, target_yaw = self._target_xyyaw

        if self.tick % 50 == 0 or need_decision:
            distance = math.hypot(target_x - self.local_pos.x, target_y - self.local_pos.y)
            self.get_logger().info(
                "SEARCH rl="
                f"{self._coverage_map.coverage_fraction:.0%} "
                f"pos=({self.local_pos.x:.1f},{self.local_pos.y:.1f}) "
                f"target=({target_x:.1f},{target_y:.1f}) dist={distance:.1f}m "
                f"action=({self._last_action[0]:.2f},{self._last_action[1]:.2f},{self._last_action[2]:.2f}) "
                f"overflow={self._last_overflow:.2f}m "
                f"new_coverage={reward_coverage:.3f}"
            )

        self._publish_setpoint(target_x, target_y, self.cruise_altitude, target_yaw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RLMissionController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
