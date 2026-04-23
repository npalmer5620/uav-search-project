import math

import pytest

gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")
yaml = pytest.importorskip("yaml")

from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from uav_rl.rl_common import SearchTaskConfig
from uav_rl.search_task_env import SearchTaskEnv
from uav_rl.train_search_policy import train_search_policy


def test_search_task_env_reward_terms_match_total():
    config = SearchTaskConfig(
        width=2.0,
        height=2.0,
        coverage_grid_side=2,
        max_step_xy_m=2.0,
        target_detection_radius_m=0.1,
        coverage_reward_scale=10.0,
        first_target_found_bonus=3.0,
        success_bonus=7.0,
        step_penalty=0.5,
        boundary_penalty=1.0,
        timeout_penalty=11.0,
        stagnation_penalty=0.2,
        required_target_count=1,
        max_episode_steps=10,
    )
    env = SearchTaskEnv(config=config)
    env.reset(
        seed=7,
        options={
            "start_state": {"x": 0.75, "y": 0.75, "yaw": 0.0},
            "targets": [{"class_name": "person", "x": 1.0, "y": 1.0}],
        },
    )

    _obs, reward, terminated, truncated, info = env.step([1.0, 1.0, 0.0])

    assert terminated is True
    assert truncated is False
    expected = (
        info["reward_new_coverage"]
        + info["reward_target_bonus"]
        + info["reward_success_bonus"]
        - info["reward_step_penalty"]
        - info["reward_boundary_penalty"]
        - info["reward_timeout_penalty"]
        - info["reward_stagnation_penalty"]
    )
    assert reward == pytest.approx(expected)
    assert info["reward_target_bonus"] == pytest.approx(3.0)
    assert info["reward_success_bonus"] == pytest.approx(7.0)
    assert info["reward_timeout_penalty"] == pytest.approx(0.0)
    assert info["reward_stagnation_penalty"] == pytest.approx(0.0)
    assert info["reward_boundary_penalty"] > 0.0


def test_search_task_env_applies_timeout_and_stagnation_penalties():
    config = SearchTaskConfig(
        width=2.0,
        height=2.0,
        coverage_grid_side=2,
        max_step_xy_m=0.0,
        target_detection_radius_m=0.1,
        coverage_reward_scale=10.0,
        first_target_found_bonus=3.0,
        success_bonus=7.0,
        step_penalty=0.5,
        boundary_penalty=1.0,
        timeout_penalty=11.0,
        stagnation_penalty=0.2,
        required_target_count=1,
        max_episode_steps=1,
    )
    env = SearchTaskEnv(config=config)
    env.reset(
        seed=7,
        options={
            "start_state": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "targets": [{"class_name": "person", "x": 0.9, "y": 0.9}],
        },
    )

    _obs, reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0])

    assert terminated is False
    assert truncated is True
    expected = (
        info["reward_new_coverage"]
        + info["reward_target_bonus"]
        + info["reward_success_bonus"]
        - info["reward_step_penalty"]
        - info["reward_boundary_penalty"]
        - info["reward_timeout_penalty"]
        - info["reward_stagnation_penalty"]
    )
    assert reward == pytest.approx(expected)
    assert info["reward_success_bonus"] == pytest.approx(0.0)
    assert info["reward_timeout_penalty"] == pytest.approx(11.0)
    assert info["reward_stagnation_penalty"] == pytest.approx(0.2)


def test_search_task_env_runs_gym_and_sb3_checkers():
    config = SearchTaskConfig(
        width=6.0,
        height=6.0,
        coverage_grid_side=6,
        max_step_xy_m=1.0,
        max_episode_steps=32,
    )
    env = SearchTaskEnv(config=config)
    gym_check_env(env)
    sb3_check_env(env, warn=True)


def test_train_search_policy_smoke(tmp_path):
    config_path = tmp_path / "search_policy.yaml"
    artifact_dir = tmp_path / "artifacts"
    config_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "width": 8.0,
                    "height": 8.0,
                    "coverage_grid_side": 8,
                    "max_step_xy_m": 1.5,
                    "search_speed": 1.5,
                    "required_target_count": 1,
                    "max_episode_steps": 64,
                },
                "ppo": {
                    "policy": "MlpPolicy",
                    "seed": 3,
                    "learning_rate": 0.0003,
                    "n_steps": 32,
                    "batch_size": 32,
                    "n_epochs": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "clip_range": 0.2,
                },
                "train": {
                    "total_timesteps": 64,
                    "num_envs": 1,
                    "eval_episodes": 1,
                    "progress_eval_episodes": 1,
                    "progress_eval_interval_timesteps": 16,
                    "curriculum": [
                        {
                            "name": "bootstrap",
                            "total_timesteps": 1,
                            "env": {
                                "width": 6.0,
                                "height": 6.0,
                                "required_target_count": 1,
                                "max_episode_steps": 32,
                            },
                        },
                        {
                            "name": "full_task",
                            "total_timesteps": 1,
                            "env": {
                                "width": 8.0,
                                "height": 8.0,
                                "required_target_count": 1,
                                "max_episode_steps": 64,
                            },
                        },
                    ],
                    "artifact_dir": str(artifact_dir),
                    "tensorboard_log": str(tmp_path / "tb"),
                },
            }
        ),
        encoding="utf-8",
    )

    summary = train_search_policy(config_path=config_path)

    assert (artifact_dir / "model.zip").exists()
    assert (artifact_dir / "vecnormalize.pkl").exists()
    assert (artifact_dir / "eval_metrics.json").exists()
    assert (artifact_dir / "training_summary.json").exists()
    assert (artifact_dir / config_path.name).exists()
    assert summary["total_timesteps"] == 64
    assert len(summary["stages"]) == 2
    assert summary["stages"][0]["name"] == "bootstrap"
    assert summary["stages"][1]["name"] == "full_task"
    assert sum(stage["total_timesteps"] for stage in summary["stages"]) == 64
    assert math.isfinite(summary["evaluation"]["mean_reward"])
