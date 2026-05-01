import math

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from uav_rl.actions import (
    ACTION_NAMES,
    ActionConfig,
    SearchAction,
    action_to_goal,
    actions_are_opposed,
    generate_candidate_goals,
    slew_yaw,
)
from uav_rl.belief_map import BeliefMap, MapGeometry, CHANNEL_CONFIRMED, CHANNEL_VICTIM_SCORE
from uav_rl.camera_model import ForwardCameraModel
from uav_rl.detection_memory import DetectionMemory, DetectionObservation
from uav_rl.reward import RewardConfig, SearchEvents, compute_search_reward
from uav_rl.search_task_env_v2 import SearchTaskConfigV2, SearchTaskEnvV2
from uav_rl.train_search_policy_v2 import train_search_policy_v2


def test_forward_camera_height_visibility_prefers_lower_altitude():
    camera = ForwardCameraModel()

    assert camera.min_ground_range_m_at_4m < camera.min_ground_range_m_at_6m
    assert camera.min_ground_range_m_at_6m < camera.depth_far_m
    assert camera.min_ground_range_m_at_10m > camera.depth_far_m * 0.9
    assert camera.is_ground_point_visible(
        drone_x=0.0,
        drone_y=0.0,
        drone_yaw=0.0,
        altitude_ned_z=-4.0,
        point_x=8.0,
        point_y=0.0,
    )
    assert not camera.is_ground_point_visible(
        drone_x=0.0,
        drone_y=0.0,
        drone_yaw=0.0,
        altitude_ned_z=-10.0,
        point_x=15.0,
        point_y=0.0,
    )


def test_candidate_yaw_rules_and_investigate_standoff():
    geometry = MapGeometry(width=40.0, height=40.0, cell_size_m=4.0)
    config = ActionConfig(cell_size_m=4.0, fixed_altitude_ned=-4.0, investigate_standoff_m=8.0)
    belief = BeliefMap(geometry)
    camera = ForwardCameraModel()
    memory = DetectionMemory()
    belief.mark_visible([geometry.world_to_grid(0.0, 0.0)])

    scan = action_to_goal(
        action=SearchAction.HOVER_SCAN,
        x=0.0,
        y=0.0,
        yaw=0.0,
        geometry=geometry,
        config=config,
        belief=belief,
        memory=memory,
        camera=camera,
    )
    assert scan.x == pytest.approx(0.0)
    assert scan.y == pytest.approx(0.0)
    assert scan.yaw == pytest.approx(math.pi / 4.0)

    memory = DetectionMemory()
    memory.update(
        [DetectionObservation(class_name="person", confidence=0.9, x=10.0, y=0.0)],
        step=1,
        drone_x=0.0,
        drone_y=0.0,
        drone_yaw=0.0,
    )
    goal = action_to_goal(
        action=SearchAction.DETECTION_CONFIRM_BEST,
        x=0.0,
        y=0.0,
        yaw=0.0,
        geometry=geometry,
        config=config,
        best_track=memory.best_track(),
        belief=belief,
        memory=memory,
        camera=camera,
    )
    assert goal.name == ACTION_NAMES[SearchAction.DETECTION_CONFIRM_BEST]
    assert goal.target_xy == pytest.approx((10.0, 0.0))
    assert math.hypot(goal.x - 10.0, goal.y) >= config.investigate_standoff_m
    assert goal.yaw == pytest.approx(math.atan2(-goal.y, 10.0 - goal.x))

    candidates = generate_candidate_goals(
        x=0.0,
        y=0.0,
        yaw=0.0,
        belief=belief,
        memory=memory,
        camera=camera,
        config=config,
        scan_allowed=False,
    )
    assert not candidates[int(SearchAction.HOVER_SCAN)].valid


def test_slew_yaw_limits_heading_changes_across_wraparound():
    assert slew_yaw(0.0, math.pi, math.pi / 4.0) == pytest.approx(math.pi / 4.0)
    assert slew_yaw(0.0, math.pi / 8.0, math.pi / 4.0) == pytest.approx(math.pi / 8.0)

    current = math.radians(170.0)
    target = math.radians(-170.0)
    limited = slew_yaw(current, target, math.radians(15.0))

    assert limited == pytest.approx(math.radians(-175.0))
    assert not actions_are_opposed(SearchAction.FRONTIER_BEST, SearchAction.ESCAPE_STUCK)
    assert not actions_are_opposed(-1, SearchAction.ESCAPE_STUCK)


def test_belief_map_detection_and_confirmation_updates():
    geometry = MapGeometry(width=16.0, height=16.0, cell_size_m=4.0)
    belief = BeliefMap(geometry)
    cells = [(0, 0), (0, 1)]

    update = belief.mark_visible(cells)
    assert update.new_observed_cells == 2
    assert update.uncertainty_reduction > 0.0

    row, col = belief.update_detection(-5.0, -5.0, 0.8)
    assert belief.grid[row, col, CHANNEL_VICTIM_SCORE] > 0.0
    belief.confirm_cell(-5.0, -5.0)
    assert belief.grid[row, col, CHANNEL_CONFIRMED] == pytest.approx(1.0)


def test_reward_terms_match_total():
    cfg = RewardConfig(
        new_observed_cell_reward=1.0,
        uncertainty_reduction_reward=0.5,
        useful_reobserve_reward=5.0,
        confirmed_victim_reward=25.0,
        mission_success_bonus=0.0,
        decision_penalty=0.05,
        useless_revisit_penalty=2.0,
        out_of_bounds_penalty=10.0,
        immediate_backtrack_penalty=3.0,
        reverse_move_penalty=0.5,
        shielded_action_penalty=1.0,
        missed_required_target_penalty=0.0,
    )
    events = SearchEvents(
        new_observed_cells=2,
        uncertainty_reduction=1.0,
        useful_reobservations=1,
        new_confirmed_victims=1,
        useless_revisit=True,
        immediate_backtrack=True,
        reverse_move=True,
        shielded_action=True,
        out_of_bounds=True,
    )
    reward = compute_search_reward(events, cfg)
    expected = 2.0 + 0.5 + 5.0 + 25.0 - 0.05 - 2.0 - 3.0 - 0.5 - 1.0 - 10.0
    assert reward.total == pytest.approx(expected)


def test_env_hard_caps_unproductive_hover_scan():
    config = SearchTaskConfigV2(
        width=20.0,
        height=20.0,
        cell_size_m=4.0,
        target_count=1,
        required_target_count=1,
        detection_probability=0.0,
        investigate_detection_probability=0.0,
        false_positive_rate=0.0,
        detection_noise_m=0.0,
        max_episode_steps=16,
        max_unproductive_scan_streak=2,
        randomize_start=False,
    )
    env = SearchTaskEnvV2(config=config)
    obs, _info = env.reset(
        seed=11,
        options={"targets": [{"class_name": "person", "x": -8.0, "y": -8.0}]},
    )
    assert obs.shape == env.observation_space.shape

    all_cells = [
        (row, col)
        for row in range(env.geometry.rows)
        for col in range(env.geometry.cols)
    ]
    env.belief.mark_visible(all_cells, uncertainty_drop=1.0)

    _obs, _reward, terminated, truncated, info = env.step(int(SearchAction.HOVER_SCAN))
    assert not terminated
    assert not truncated
    assert info["last_action_int"] == int(SearchAction.HOVER_SCAN)
    assert info["unproductive_scan_streak"] == 1
    assert info["scan_disabled_count"] == 0

    _obs, _reward, _terminated, _truncated, info = env.step(int(SearchAction.HOVER_SCAN))
    assert info["unproductive_scan_streak"] == 2
    assert info["scan_disabled_count"] == 0

    _obs, _reward, _terminated, _truncated, info = env.step(int(SearchAction.HOVER_SCAN))
    assert info["events"]["shielded_action"]
    assert "substituted_for_candidate_hover_scan" in info["last_action"]
    assert info["scan_disabled_count"] == 1
    assert info["shielded_action_count"] == 1
    assert info["reward"]["penalty_shielded_action"] == pytest.approx(1.0)


def test_search_task_env_v2_runs_gym_and_sb3_checkers():
    sb3 = pytest.importorskip("stable_baselines3")
    from gymnasium.utils.env_checker import check_env as gym_check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    config = SearchTaskConfigV2(
        width=20.0,
        height=20.0,
        cell_size_m=4.0,
        target_count=1,
        required_target_count=1,
        false_positive_rate=0.0,
        detection_noise_m=0.1,
        max_episode_steps=32,
        randomize_start=False,
    )
    env = SearchTaskEnvV2(config=config)
    gym_check_env(env)
    sb3_check_env(env, warn=True)


def test_train_search_policy_v2_smoke(tmp_path):
    pytest.importorskip("stable_baselines3")
    yaml = pytest.importorskip("yaml")

    config_path = tmp_path / "search_policy_v2.yaml"
    artifact_dir = tmp_path / "artifacts"
    config_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "width": 20.0,
                    "height": 20.0,
                    "cell_size_m": 4.0,
                    "patch_side": 11,
                    "target_count": 1,
                    "required_target_count": 1,
                    "false_positive_rate": 0.0,
                    "detection_noise_m": 0.1,
                    "max_episode_steps": 32,
                    "randomize_start": False,
                },
                "dqn": {
                    "policy": "MlpPolicy",
                    "seed": 3,
                    "learning_rate": 0.0001,
                    "buffer_size": 1000,
                    "learning_starts": 0,
                    "batch_size": 16,
                    "gamma": 0.99,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "target_update_interval": 64,
                    "exploration_fraction": 0.5,
                    "exploration_initial_eps": 1.0,
                    "exploration_final_eps": 0.2,
                },
                "train": {
                    "total_timesteps": 32,
                    "eval_episodes": 1,
                    "baseline_episodes": 1,
                    "progress_eval_episodes": 1,
                    "progress_eval_interval_timesteps": 0,
                    "artifact_dir": str(artifact_dir),
                    "tensorboard_log": str(tmp_path / "tb"),
                },
            }
        ),
        encoding="utf-8",
    )

    summary = train_search_policy_v2(config_path=config_path)

    assert (artifact_dir / "model.zip").exists()
    assert (artifact_dir / "eval_metrics.json").exists()
    assert (artifact_dir / "baseline_comparison.json").exists()
    assert (artifact_dir / "training_summary.json").exists()
    assert summary["algorithm"] == "DQN"
    assert np.isfinite(summary["evaluation"]["mean_reward"])
