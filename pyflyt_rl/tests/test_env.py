from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("gymnasium") is None, reason="Gymnasium not installed")


def test_env_reset_step_shapes():
    from uav_triage_rl.env import TriageSearchEnv

    config = {
        "env": {
            "backend": "kinematic",
            "victim_count": 2,
            "required_victim_count": 1,
            "obstacle_count": 1,
            "max_episode_steps": 8,
            "randomize_start": False,
        },
        "perception": {
            "detection_probability": 1.0,
            "false_positive_rate": 0.0,
            "min_hits": 1,
        },
    }
    env = TriageSearchEnv(config=config)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert info["required_victims"] == 1
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "coverage_fraction" in info
    env.close()

