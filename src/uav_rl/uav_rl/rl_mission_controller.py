#!/usr/bin/env python3
"""ROS 2 mission controller that uses a learned search policy."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3DArray

from uav_planning.mission_base import MissionControllerBase, Phase
from uav_rl.actions import ActionConfig, ACTION_NAMES, SearchAction, action_to_goal, slew_yaw
from uav_rl.belief_map import BeliefMap
from uav_rl.camera_model import ForwardCameraModel
from uav_rl.detection_memory import (
    ConfirmationConfig,
    DetectionMemory,
    DetectionObservation as V2DetectionObservation,
)
from uav_rl.observation_v2 import ObservationConfig, encode_observation
from uav_rl.pyflyt_runtime_adapter import (
    PyFlytActionConfig,
    PyFlytAdapterConfig,
    PyFlytRuntimeAdapter,
    PyFlytSearchAction,
    load_pyflyt_adapter_config,
)
from uav_rl.rl_common import (
    CoverageMap,
    ObservationEncoder,
    SearchTaskConfig,
    apply_relative_action,
    load_vecnormalize_for_inference,
    resolve_repo_path,
)
from uav_rl.search_task_env_v2 import SearchTaskConfigV2


class RLMissionController(MissionControllerBase):
    """Mission controller that replaces the grid sweep with RL search steps."""

    def __init__(self) -> None:
        self.rl_policy_version = "v2"
        self.rl_algorithm = "dqn"
        self.rl_artifact_dir = "artifacts/rl/search_policy_v2"
        self.rl_model_path = "artifacts/rl/search_policy_v2/model.zip"
        self.rl_vecnormalize_path = ""
        self.rl_pyflyt_config_path = ""
        self.rl_scan_yaw_step_deg = 25.0
        self.rl_max_yaw_step_deg = 25.0
        self.rl_max_unproductive_scan_streak = 2

        self._model = None
        self._vec_normalize = None

        # V1 PPO state.
        self._coverage_map: CoverageMap | None = None
        self._observation_encoder: ObservationEncoder | None = None
        self._policy_config: SearchTaskConfig | None = None

        # V2 DQN state.
        self._v2_config: SearchTaskConfigV2 | None = None
        self._v2_belief: BeliefMap | None = None
        self._v2_memory: DetectionMemory | None = None
        self._v2_camera: ForwardCameraModel | None = None
        self._v2_obs_config: ObservationConfig | None = None
        self._v2_action_config: ActionConfig | None = None
        self._v2_last_action = -1
        self._v2_last_action_name = ""
        self._v2_previous_cell: tuple[int, int] | None = None
        self._v2_recent_cells: list[tuple[int, int]] = []
        self._v2_shielded_actions = 0
        self._v2_unproductive_scan_streak = 0
        self._v2_scan_disabled_count = 0
        self._v2_last_action_was_scan = False
        self._v2_detection_update_counter = 0
        self._v2_scan_detection_counter_at_start = 0

        # PyFlyt DQN state.
        self._pyflyt_adapter: PyFlytRuntimeAdapter | None = None
        self._pyflyt_memory: DetectionMemory | None = None
        self._pyflyt_config_source = ""
        self._pyflyt_detection_update_counter = 0

        self._target_xyyaw: tuple[float, float, float] | None = None
        self._last_pose_xy: tuple[float, float] | None = None
        self._coverage_initialized = False
        self._decision_interval_ticks = 1
        self._last_decision_tick = -1
        self._last_action = np.zeros(3, dtype=np.float32)
        self._last_overflow = 0.0
        self._last_visible_new_cells = 0

        self._camera_info: CameraInfo | None = None
        self._depth_image: np.ndarray | None = None

        super().__init__(
            "rl_mission_controller",
            extra_parameter_declarations=self._declare_rl_parameters,
        )

        if self.rl_policy_version in {"v2", "pyflyt", "pyflyt_v0"}:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(CameraInfo, "/camera/camera_info", self._camera_info_cb, qos)
            self.create_subscription(Image, "/camera/depth/image_raw", self._depth_cb, qos)

        self.get_logger().info(
            "RL search policy enabled "
            f"(version={self.rl_policy_version}, algorithm={self.rl_algorithm}, "
            f"model={self.rl_model_path})"
        )

    def _declare_rl_parameters(self) -> None:
        self.declare_parameter("rl.policy_version", "v2")
        self.declare_parameter("rl.algorithm", "dqn")
        self.declare_parameter("rl.artifact_dir", "artifacts/rl/search_policy_v2")
        self.declare_parameter("rl.model_path", "artifacts/rl/search_policy_v2/model.zip")
        self.declare_parameter(
            "rl.vecnormalize_path",
            "artifacts/rl/search_policy/vecnormalize.pkl",
        )
        self.declare_parameter("rl.pyflyt_config_path", "")
        self.declare_parameter("rl.decision_period_s", 2.0)
        self.declare_parameter("rl.max_step_xy_m", 4.0)
        self.declare_parameter("rl.scan_yaw_step_deg", 25.0)
        self.declare_parameter("rl.max_yaw_step_deg", 25.0)
        self.declare_parameter("rl.max_unproductive_scan_streak", 2)
        self.declare_parameter("rl.coverage_grid_side", 16)
        self.declare_parameter("rl.cell_size_m", 4.0)
        self.declare_parameter("rl.patch_side", 11)

    def _init_search_controller(self) -> None:
        self.rl_policy_version = (
            str(self.get_parameter("rl.policy_version").value).strip().lower() or "v2"
        )
        self.rl_algorithm = str(self.get_parameter("rl.algorithm").value).strip() or "dqn"
        self.rl_artifact_dir = str(self.get_parameter("rl.artifact_dir").value).strip()
        configured_model = str(self.get_parameter("rl.model_path").value).strip()
        self.rl_model_path = configured_model or f"{self.rl_artifact_dir}/model.zip"
        self.rl_vecnormalize_path = str(self.get_parameter("rl.vecnormalize_path").value).strip()
        self.rl_pyflyt_config_path = str(
            self.get_parameter("rl.pyflyt_config_path").value
        ).strip()
        self.rl_decision_period_s = max(
            0.1, float(self.get_parameter("rl.decision_period_s").value)
        )
        self.rl_max_step_xy_m = max(
            0.1, float(self.get_parameter("rl.max_step_xy_m").value)
        )
        self.rl_scan_yaw_step_deg = max(
            0.0, float(self.get_parameter("rl.scan_yaw_step_deg").value)
        )
        self.rl_max_yaw_step_deg = max(
            0.0, float(self.get_parameter("rl.max_yaw_step_deg").value)
        )
        self.rl_max_unproductive_scan_streak = max(
            0,
            int(self.get_parameter("rl.max_unproductive_scan_streak").value),
        )
        self.rl_coverage_grid_side = max(
            1, int(self.get_parameter("rl.coverage_grid_side").value)
        )
        self.rl_cell_size_m = max(0.5, float(self.get_parameter("rl.cell_size_m").value))
        self.rl_patch_side = max(3, int(self.get_parameter("rl.patch_side").value))

        if self.rl_policy_version == "v1":
            self._init_v1_controller()
        elif self.rl_policy_version == "v2":
            self._init_v2_controller()
        elif self.rl_policy_version in {"pyflyt", "pyflyt_v0"}:
            self._init_pyflyt_controller()
        else:
            raise RuntimeError(
                "Unsupported rl.policy_version="
                f"'{self.rl_policy_version}' (expected v1, v2, or pyflyt)"
            )

        self._decision_interval_ticks = max(
            1,
            int(round(self.rl_decision_period_s / max(self.dt, 1e-6))),
        )

    def _init_v1_controller(self) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("stable_baselines3 is required for RL mission control") from exc

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
        self._coverage_map = CoverageMap(self.rl_coverage_grid_side, self._policy_config)
        self._observation_encoder = ObservationEncoder(
            self._policy_config,
            self._coverage_map,
            decision_period_s=self.rl_decision_period_s,
        )
        model_path = resolve_repo_path(self.rl_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RL model not found: {model_path}")
        self._model = PPO.load(str(model_path))
        self._vec_normalize = load_vecnormalize_for_inference(
            self.rl_vecnormalize_path,
            self._policy_config,
        )
        if self._vec_normalize is None:
            self.get_logger().warn("VecNormalize stats not found; using raw observations")

    def _init_v2_controller(self) -> None:
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("stable_baselines3 is required for RL mission control") from exc

        self._v2_config = SearchTaskConfigV2(
            width=self.search_width,
            height=self.search_height,
            origin_x=self.search_origin_x,
            origin_y=self.search_origin_y,
            cell_size_m=self.rl_cell_size_m,
            patch_side=self.rl_patch_side,
            fixed_altitude_ned=self.search_altitude,
            decision_period_s=self.rl_decision_period_s,
            search_speed_m_s=self.search_speed,
            max_episode_steps=max(1, int(self.search_width * self.search_height / max(self.rl_cell_size_m, 1.0))),
            max_unproductive_scan_streak=self.rl_max_unproductive_scan_streak,
        )
        self._v2_belief = BeliefMap(self._v2_config.geometry)
        self._v2_memory = DetectionMemory(self._v2_config.confirmation)
        self._v2_camera = self._v2_config.camera
        self._v2_obs_config = self._v2_config.observation_config
        self._v2_action_config = ActionConfig(
            cell_size_m=self.rl_cell_size_m,
            fixed_altitude_ned=self.search_altitude,
            scan_yaw_step_rad=math.radians(self.rl_scan_yaw_step_deg),
        )

        model_path = resolve_repo_path(self.rl_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"V2 DQN model not found: {model_path}")
        self._model = DQN.load(str(model_path))
        expected_shape = (self._v2_obs_config.observation_size,)
        model_shape = getattr(self._model.observation_space, "shape", None)
        if model_shape != expected_shape:
            raise RuntimeError(
                "V2 DQN observation shape mismatch: "
                f"model expects {model_shape}, runtime builds {expected_shape}. "
                "Retrain the V2 policy with the current Gym environment."
            )

    def _init_pyflyt_controller(self) -> None:
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("stable_baselines3 is required for RL mission control") from exc

        model_path = resolve_repo_path(self.rl_model_path)
        artifact_dir = resolve_repo_path(self.rl_artifact_dir)
        config_path = (
            resolve_repo_path(self.rl_pyflyt_config_path)
            if self.rl_pyflyt_config_path
            else None
        )
        fallback_config = PyFlytAdapterConfig(
            width_m=self.search_width,
            height_m=self.search_height,
            origin_x=self.search_origin_x,
            origin_y=self.search_origin_y,
            cell_size_m=self.rl_cell_size_m,
            search_altitude_m=max(abs(self.search_altitude), 1.0),
            max_speed_m_s=max(self.search_speed, 1.0),
            decision_period_s=self.rl_decision_period_s,
            max_episode_steps=max(
                1,
                int(self.search_width * self.search_height / max(self.rl_cell_size_m, 1.0)),
            ),
            patch_side=self.rl_patch_side,
            action=PyFlytActionConfig(scan_yaw_step_deg=self.rl_scan_yaw_step_deg),
        )
        pyflyt_config, loaded_config_path = load_pyflyt_adapter_config(
            model_path=model_path,
            artifact_dir=artifact_dir,
            config_path=config_path,
            fallback=fallback_config,
        )
        self._pyflyt_config_source = str(loaded_config_path) if loaded_config_path else ""
        self.rl_decision_period_s = max(0.1, pyflyt_config.decision_period_s)
        self.rl_cell_size_m = max(0.5, pyflyt_config.cell_size_m)
        self.rl_patch_side = max(3, pyflyt_config.patch_side)
        self.rl_scan_yaw_step_deg = max(0.0, pyflyt_config.action.scan_yaw_step_deg)
        self._pyflyt_adapter = PyFlytRuntimeAdapter(pyflyt_config)
        self._pyflyt_memory = DetectionMemory(
            ConfirmationConfig(
                match_distance_m=pyflyt_config.match_distance_m,
                max_history=8,
                track_timeout_steps=pyflyt_config.track_timeout_steps,
                min_hits=pyflyt_config.min_hits,
                min_mean_confidence=pyflyt_config.min_mean_confidence,
                min_viewpoint_separation_m=0.0,
                min_yaw_span_rad=0.0,
                max_position_spread_m=max(2.0, pyflyt_config.match_distance_m * 2.0),
            )
        )

        if not model_path.exists():
            raise FileNotFoundError(f"PyFlyt DQN model not found: {model_path}")
        self._model = DQN.load(str(model_path), device="cpu")
        expected_shape = (self._pyflyt_adapter.observation_size,)
        model_shape = getattr(self._model.observation_space, "shape", None)
        if model_shape != expected_shape:
            raise RuntimeError(
                "PyFlyt DQN observation shape mismatch: "
                f"model expects {model_shape}, runtime builds {expected_shape}. "
                "Use a pyflyt_rl model/config pair with the same patch_side/top_k_tracks."
            )
        action_n = getattr(getattr(self._model, "action_space", None), "n", None)
        if action_n != len(PyFlytSearchAction):
            raise RuntimeError(
                "PyFlyt DQN action-space mismatch: "
                f"model has {action_n}, runtime expects {len(PyFlytSearchAction)}"
            )
        self.get_logger().info(
            "Loaded PyFlyt runtime adapter "
            f"(obs={expected_shape[0]}, cell={pyflyt_config.cell_size_m:.1f}m, "
            f"decision={pyflyt_config.decision_period_s:.1f}s, "
            f"config={self._pyflyt_config_source or 'parameters'})"
        )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _depth_cb(self, msg: Image) -> None:
        try:
            if msg.encoding == "32FC1":
                self._depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
                self._depth_image = depth * 0.001
        except ValueError as exc:
            self.get_logger().warning(f"Depth image conversion failed: {exc}")

    def _detection_cb(self, msg: Detection3DArray) -> None:
        if self.rl_policy_version == "v1":
            super()._detection_cb(msg)
            return
        if self.rl_policy_version in {"pyflyt", "pyflyt_v0"}:
            self._pyflyt_detection_cb(msg)
            return
        if self.phase != Phase.SEARCH or self._v2_memory is None or self._v2_belief is None:
            return

        observations: list[V2DetectionObservation] = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            confidence = float(hyp.hypothesis.score)
            if confidence < self.confidence_threshold:
                continue
            enu_x = float(hyp.pose.pose.position.x)
            enu_y = float(hyp.pose.pose.position.y)
            enu_z = float(hyp.pose.pose.position.z)
            ned_x = enu_y
            ned_y = enu_x
            ned_z = -enu_z
            if self._near_investigated(ned_x, ned_y):
                continue
            observations.append(
                V2DetectionObservation(
                    class_name=hyp.hypothesis.class_id,
                    confidence=confidence,
                    x=ned_x,
                    y=ned_y,
                    z=ned_z,
                )
            )
            self._v2_belief.update_detection(ned_x, ned_y, confidence)

        if not observations:
            return

        new_tracks, updated_existing = self._v2_memory.update(
            observations,
            step=self.tick,
            drone_x=float(self.local_pos.x),
            drone_y=float(self.local_pos.y),
            drone_yaw=self._current_yaw(),
        )
        if new_tracks > 0 or updated_existing > 0:
            self._v2_detection_update_counter += new_tracks + updated_existing
        for track in self._v2_memory.confirm_ready_tracks():
            tx, ty, tz = track.filtered_position
            self._v2_belief.confirm_cell(tx, ty)
            self._v2_detection_update_counter += 1
            ex, ey, ez = self._ned_to_enu(tx, ty, tz)
            self.investigated_locations.append(
                (tx, ty, ex, ey, ez, track.class_name, track.mean_confidence)
            )
            self._publish_event(
                f"RL confirmed {track.class_name} track #{track.track_id} "
                f"(hits={track.hits}, conf={track.mean_confidence:.2f})"
            )

    def _pyflyt_detection_cb(self, msg: Detection3DArray) -> None:
        if (
            self.phase != Phase.SEARCH
            or self._pyflyt_adapter is None
            or self._pyflyt_memory is None
        ):
            return

        observations: list[V2DetectionObservation] = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            confidence = float(hyp.hypothesis.score)
            if confidence < self.confidence_threshold:
                continue
            enu_x = float(hyp.pose.pose.position.x)
            enu_y = float(hyp.pose.pose.position.y)
            enu_z = float(hyp.pose.pose.position.z)
            ned_x = enu_y
            ned_y = enu_x
            ned_z = -enu_z
            if self._near_investigated(ned_x, ned_y):
                continue
            observations.append(
                V2DetectionObservation(
                    class_name=hyp.hypothesis.class_id,
                    confidence=confidence,
                    x=ned_x,
                    y=ned_y,
                    z=ned_z,
                )
            )
            self._pyflyt_adapter.mark_detection(ned_x, ned_y, confidence)

        if not observations:
            return

        step = self._pyflyt_decision_step()
        new_tracks, updated_existing = self._pyflyt_memory.update(
            observations,
            step=step,
            drone_x=float(self.local_pos.x),
            drone_y=float(self.local_pos.y),
            drone_yaw=self._current_yaw(),
        )
        if new_tracks > 0 or updated_existing > 0:
            self._pyflyt_detection_update_counter += new_tracks + updated_existing
        for track in self._pyflyt_memory.confirm_ready_tracks():
            tx, ty, tz = track.filtered_position
            self._pyflyt_adapter.mark_confirmed(tx, ty)
            self._pyflyt_detection_update_counter += 1
            ex, ey, ez = self._ned_to_enu(tx, ty, tz)
            self.investigated_locations.append(
                (tx, ty, ex, ey, ez, track.class_name, track.mean_confidence)
            )
            self._publish_event(
                f"PyFlyt RL confirmed {track.class_name} track #{track.track_id} "
                f"(hits={track.hits}, conf={track.mean_confidence:.2f})"
            )

    def _on_search_enter(self) -> None:
        current_xy = (float(self.local_pos.x), float(self.local_pos.y))
        if self.rl_policy_version == "v1":
            if self._coverage_map is not None and not self._coverage_initialized:
                self._coverage_map.reset()
                self._coverage_map.mark_point(current_xy[0], current_xy[1])
                self._coverage_initialized = True
        elif self.rl_policy_version == "v2":
            if self._v2_belief is not None:
                self._v2_belief.reset()
            if self._v2_memory is not None:
                self._v2_memory.reset()
        elif self._pyflyt_adapter is not None:
            self._pyflyt_adapter.reset()
            if self._pyflyt_memory is not None:
                self._pyflyt_memory.reset()
            self._pyflyt_detection_update_counter = 0
        self._last_pose_xy = current_xy
        self._target_xyyaw = None
        self._last_decision_tick = self.tick - self._decision_interval_ticks
        self._v2_previous_cell = None
        self._v2_recent_cells = [self._v2_belief.geometry.world_to_grid(*current_xy)] if self._v2_belief else []
        self._v2_shielded_actions = 0
        self._v2_unproductive_scan_streak = 0
        self._v2_scan_disabled_count = 0
        self._v2_last_action_was_scan = False
        self._v2_detection_update_counter = 0
        self._v2_scan_detection_counter_at_start = 0

    def _on_search_exit(self) -> None:
        self._last_pose_xy = None
        self._target_xyyaw = None
        self._v2_previous_cell = None
        self._v2_recent_cells = []

    def _search_state_detail(self) -> str | None:
        if self.rl_policy_version == "v1":
            if self._coverage_map is None:
                return None
            return f"{self._coverage_map.coverage_fraction:.0%}"
        if self.rl_policy_version in {"pyflyt", "pyflyt_v0"}:
            if self._pyflyt_adapter is None:
                return None
            return f"{self._pyflyt_adapter.coverage_fraction:.0%}"
        if self._v2_belief is None:
            return None
        return f"{self._v2_belief.coverage_fraction:.0%}"

    def _predict_v1_action(self, obs: np.ndarray) -> np.ndarray:
        inference_obs = obs
        if self._vec_normalize is not None:
            inference_obs = self._vec_normalize.normalize_obs(obs.reshape(1, -1)).squeeze(0)
        action, _ = self._model.predict(inference_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def _predict_v2_action(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    def _pyflyt_decision_step(self) -> int:
        return int(self.search_elapsed_s / max(self.rl_decision_period_s, 1e-6))

    def _limit_v2_yaw(self, desired_yaw: float) -> float:
        max_step_rad = math.radians(self.rl_max_yaw_step_deg)
        if max_step_rad <= 0.0:
            return self._current_yaw()
        return slew_yaw(self._current_yaw(), desired_yaw, max_step_rad)

    def _update_v2_belief_from_current_view(self) -> None:
        if self._v2_belief is None or self._v2_camera is None:
            return
        self._v2_belief.age(self.rl_decision_period_s)
        cells = self._v2_camera.visible_cells(
            geometry=self._v2_belief.geometry,
            drone_x=float(self.local_pos.x),
            drone_y=float(self.local_pos.y),
            drone_yaw=self._current_yaw(),
            altitude_ned_z=float(self.cruise_altitude or self.search_altitude),
        )
        update = self._v2_belief.mark_visible(cells)
        self._last_visible_new_cells = update.new_observed_cells
        self._update_v2_obstacles_from_depth()

    def _update_v2_obstacles_from_depth(self) -> None:
        if self._v2_belief is None or self._camera_info is None or self._depth_image is None:
            return
        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return
        height, width = self._depth_image.shape
        for v in range(height // 4, height, max(1, height // 6)):
            for u in range(0, width, max(1, width // 8)):
                depth = float(self._depth_image[v, u])
                if not math.isfinite(depth) or depth <= 0.2 or depth >= 18.5:
                    continue
                bearing = math.atan2((u - cx), fx)
                vertical = math.atan2((v - cy), fy)
                horizontal_range = max(0.0, depth * math.cos(bearing) * math.cos(vertical))
                world_bearing = self._current_yaw() + bearing
                obs_x = float(self.local_pos.x) + horizontal_range * math.cos(world_bearing)
                obs_y = float(self.local_pos.y) + horizontal_range * math.sin(world_bearing)
                self._v2_belief.mark_obstacle(obs_x, obs_y, confidence=0.5)

    def _update_pyflyt_belief_from_current_view(self) -> None:
        if self._pyflyt_adapter is None:
            return
        self._pyflyt_adapter.mark_view(
            drone_x=float(self.local_pos.x),
            drone_y=float(self.local_pos.y),
            drone_yaw=self._current_yaw(),
            altitude_ned_z=float(self.cruise_altitude or self.search_altitude),
        )
        self._update_pyflyt_obstacles_from_depth()

    def _update_pyflyt_obstacles_from_depth(self) -> None:
        if self._pyflyt_adapter is None or self._camera_info is None or self._depth_image is None:
            return
        fx = float(self._camera_info.k[0])
        fy = float(self._camera_info.k[4])
        cx = float(self._camera_info.k[2])
        cy = float(self._camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return
        height, width = self._depth_image.shape
        for v in range(height // 4, height, max(1, height // 6)):
            for u in range(0, width, max(1, width // 8)):
                depth = float(self._depth_image[v, u])
                if not math.isfinite(depth) or depth <= 0.2 or depth >= 18.5:
                    continue
                bearing = math.atan2((u - cx), fx)
                vertical = math.atan2((v - cy), fy)
                horizontal_range = max(0.0, depth * math.cos(bearing) * math.cos(vertical))
                world_bearing = self._current_yaw() + bearing
                obs_x = float(self.local_pos.x) + horizontal_range * math.cos(world_bearing)
                obs_y = float(self.local_pos.y) + horizontal_range * math.sin(world_bearing)
                self._pyflyt_adapter.mark_obstacle(obs_x, obs_y, confidence=0.5)

    def _v2_scan_allowed(self) -> bool:
        if self._v2_config is None:
            return True
        return self._v2_unproductive_scan_streak < max(
            0,
            int(self._v2_config.max_unproductive_scan_streak),
        )

    def _update_v2_scan_productivity(self) -> None:
        if not self._v2_last_action_was_scan:
            return
        productive = (
            self._last_visible_new_cells > 0
            or self._v2_detection_update_counter > self._v2_scan_detection_counter_at_start
        )
        if productive:
            self._v2_unproductive_scan_streak = 0
        else:
            self._v2_unproductive_scan_streak += 1
        self._v2_last_action_was_scan = False

    def _search_step(self) -> None:
        if self.rl_policy_version == "v1":
            self._search_step_v1()
        elif self.rl_policy_version == "v2":
            self._search_step_v2()
        else:
            self._search_step_pyflyt()

    def _search_step_v1(self) -> None:
        if self._coverage_map is None or self._observation_encoder is None:
            raise RuntimeError("V1 RL search policy was not initialized")

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
            self._last_action = self._predict_v1_action(obs)
            target_x, target_y, target_yaw, self._last_overflow = apply_relative_action(
                x=self.local_pos.x,
                y=self.local_pos.y,
                yaw=self._current_yaw(),
                action=self._last_action,
                config=self._policy_config,
            )
            self._target_xyyaw = (target_x, target_y, target_yaw)
            self._last_decision_tick = self.tick

        target_x, target_y, target_yaw = self._target_xyyaw or (
            self.local_pos.x,
            self.local_pos.y,
            self._current_yaw(),
        )
        if self.tick % 50 == 0 or need_decision:
            distance = math.hypot(target_x - self.local_pos.x, target_y - self.local_pos.y)
            self.get_logger().info(
                "SEARCH rl_v1="
                f"{self._coverage_map.coverage_fraction:.0%} "
                f"pos=({self.local_pos.x:.1f},{self.local_pos.y:.1f}) "
                f"target=({target_x:.1f},{target_y:.1f}) dist={distance:.1f}m "
                f"action=({self._last_action[0]:.2f},{self._last_action[1]:.2f},{self._last_action[2]:.2f}) "
                f"overflow={self._last_overflow:.2f}m new_coverage={reward_coverage:.3f}"
            )
        self._publish_setpoint(target_x, target_y, self.cruise_altitude, target_yaw)

    def _search_step_v2(self) -> None:
        if (
            self._v2_belief is None
            or self._v2_memory is None
            or self._v2_camera is None
            or self._v2_obs_config is None
            or self._v2_action_config is None
        ):
            raise RuntimeError("V2 RL search policy was not initialized")

        if self._v2_belief.coverage_fraction >= 1.0:
            self._publish_event("RL V2 search coverage complete. Returning to launch.")
            self.phase = Phase.RTH
            return

        need_decision = (
            self._target_xyyaw is None
            or (self.tick - self._last_decision_tick) >= self._decision_interval_ticks
        )
        if need_decision:
            self._update_v2_belief_from_current_view()
            self._update_v2_scan_productivity()
            obs = encode_observation(
                belief=self._v2_belief,
                memory=self._v2_memory,
                x=float(self.local_pos.x),
                y=float(self.local_pos.y),
                yaw=self._current_yaw(),
                altitude_ned_z=float(self.cruise_altitude or self.search_altitude),
                vx=self.local_pos.vx if self.local_pos.v_xy_valid else 0.0,
                vy=self.local_pos.vy if self.local_pos.v_xy_valid else 0.0,
                step=int(self.search_elapsed_s / max(self.rl_decision_period_s, 1e-6)),
                config=self._v2_obs_config,
                last_action=self._v2_last_action,
            )
            requested_action = self._predict_v2_action(obs)
            self._v2_last_action = requested_action
            best_track = self._v2_memory.best_track()
            scan_allowed = self._v2_scan_allowed()
            goal = action_to_goal(
                action=self._v2_last_action,
                x=float(self.local_pos.x),
                y=float(self.local_pos.y),
                yaw=self._current_yaw(),
                geometry=self._v2_belief.geometry,
                config=self._v2_action_config,
                best_track=best_track,
                belief=self._v2_belief,
                memory=self._v2_memory,
                camera=self._v2_camera,
                scan_allowed=scan_allowed,
                recent_cells=self._v2_recent_cells,
            )
            current_cell = self._v2_belief.geometry.world_to_grid(
                float(self.local_pos.x),
                float(self.local_pos.y),
            )
            substituted_candidate = "_substituted_for_" in goal.name or not goal.valid
            if substituted_candidate:
                self._v2_shielded_actions += 1
                if requested_action == int(SearchAction.HOVER_SCAN) and not scan_allowed:
                    self._v2_scan_disabled_count += 1
            goal_cell = self._v2_belief.geometry.world_to_grid(goal.x, goal.y)
            if (
                self._v2_previous_cell is not None
                and goal_cell == self._v2_previous_cell
                and goal_cell != current_cell
            ):
                self._v2_shielded_actions += 1
                self._v2_last_action = int(SearchAction.ESCAPE_STUCK)
                goal = action_to_goal(
                    action=self._v2_last_action,
                    x=float(self.local_pos.x),
                    y=float(self.local_pos.y),
                    yaw=self._current_yaw(),
                    geometry=self._v2_belief.geometry,
                    config=self._v2_action_config,
                    best_track=best_track,
                    belief=self._v2_belief,
                    memory=self._v2_memory,
                    camera=self._v2_camera,
                    scan_allowed=scan_allowed,
                    recent_cells=self._v2_recent_cells,
                )
                goal_cell = self._v2_belief.geometry.world_to_grid(goal.x, goal.y)
            if goal_cell != current_cell:
                self._v2_previous_cell = current_cell
                self._v2_unproductive_scan_streak = 0
            target_yaw = self._limit_v2_yaw(goal.yaw)
            self._target_xyyaw = (goal.x, goal.y, target_yaw)
            self._v2_last_action_name = goal.name
            self._last_overflow = goal.overflow_m
            if "detection_confirm" in goal.name and best_track is not None:
                best_track.investigated = True
            self._v2_last_action_was_scan = (
                self._v2_last_action == int(SearchAction.HOVER_SCAN)
                and goal.name == ACTION_NAMES[SearchAction.HOVER_SCAN]
            )
            if self._v2_last_action_was_scan:
                self._v2_scan_detection_counter_at_start = self._v2_detection_update_counter
            self._v2_recent_cells.append(goal_cell)
            self._v2_recent_cells = self._v2_recent_cells[-6:]
            self._last_decision_tick = self.tick

        target_x, target_y, target_yaw = self._target_xyyaw or (
            self.local_pos.x,
            self.local_pos.y,
            self._current_yaw(),
        )
        if self.tick % 50 == 0 or need_decision:
            distance = math.hypot(target_x - self.local_pos.x, target_y - self.local_pos.y)
            self.get_logger().info(
                "SEARCH rl_v2="
                f"{self._v2_belief.coverage_fraction:.0%} "
                f"pos=({self.local_pos.x:.1f},{self.local_pos.y:.1f}) "
                f"target=({target_x:.1f},{target_y:.1f}) dist={distance:.1f}m "
                f"yaw=({math.degrees(self._current_yaw()):.0f}->{math.degrees(target_yaw):.0f}deg) "
                f"action={self._v2_last_action}:{self._v2_last_action_name} "
                f"tracks={self._v2_memory.unconfirmed_count()} "
                f"overflow={self._last_overflow:.2f}m "
                f"shielded={self._v2_shielded_actions} "
                f"new_visible={self._last_visible_new_cells} "
                f"scan_streak={self._v2_unproductive_scan_streak} "
                f"scan_disabled={self._v2_scan_disabled_count}"
            )
        self._publish_setpoint(target_x, target_y, self.cruise_altitude, target_yaw)

    def _search_step_pyflyt(self) -> None:
        if self._pyflyt_adapter is None or self._pyflyt_memory is None:
            raise RuntimeError("PyFlyt RL search policy was not initialized")

        if self._pyflyt_adapter.coverage_fraction >= 1.0:
            self._publish_event("PyFlyt RL search coverage complete. Returning to launch.")
            self.phase = Phase.RTH
            return

        need_decision = (
            self._target_xyyaw is None
            or (self.tick - self._last_decision_tick) >= self._decision_interval_ticks
        )
        if need_decision:
            self._update_pyflyt_belief_from_current_view()
            step = self._pyflyt_decision_step()
            obs = self._pyflyt_adapter.encode_observation(
                x=float(self.local_pos.x),
                y=float(self.local_pos.y),
                yaw=self._current_yaw(),
                altitude_ned_z=float(self.cruise_altitude or self.search_altitude),
                vx=self.local_pos.vx if self.local_pos.v_xy_valid else 0.0,
                vy=self.local_pos.vy if self.local_pos.v_xy_valid else 0.0,
                step=step,
                memory=self._pyflyt_memory,
            )
            requested_action = self._predict_v2_action(obs)
            goal = self._pyflyt_adapter.goal_for_action(
                action=requested_action,
                x=float(self.local_pos.x),
                y=float(self.local_pos.y),
                yaw=self._current_yaw(),
                memory=self._pyflyt_memory,
            )
            self._target_xyyaw = (goal.x, goal.y, goal.yaw)
            self._last_overflow = goal.overflow_m
            self._last_decision_tick = self.tick

        target_x, target_y, target_yaw = self._target_xyyaw or (
            self.local_pos.x,
            self.local_pos.y,
            self._current_yaw(),
        )
        if self.tick % 50 == 0 or need_decision:
            distance = math.hypot(target_x - self.local_pos.x, target_y - self.local_pos.y)
            confirmed = sum(1 for track in self._pyflyt_memory.tracks if track.confirmed)
            self.get_logger().info(
                "SEARCH rl_pyflyt="
                f"{self._pyflyt_adapter.coverage_fraction:.0%} "
                f"pos=({self.local_pos.x:.1f},{self.local_pos.y:.1f}) "
                f"target=({target_x:.1f},{target_y:.1f}) dist={distance:.1f}m "
                f"yaw=({math.degrees(self._current_yaw()):.0f}->{math.degrees(target_yaw):.0f}deg) "
                f"action={self._pyflyt_adapter.last_action}:"
                f"{self._pyflyt_adapter.last_action_name} "
                f"tracks={self._pyflyt_memory.unconfirmed_count()} "
                f"confirmed={confirmed} "
                f"overflow={self._last_overflow:.2f}m "
                f"new_visible={self._pyflyt_adapter.last_visible_new_cells} "
                f"detection_updates={self._pyflyt_detection_update_counter}"
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
