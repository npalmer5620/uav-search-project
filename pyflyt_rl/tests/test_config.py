from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("yaml") is None, reason="PyYAML not installed")
def test_default_config_loads():
    from uav_triage_rl.config import load_config

    path, config = load_config()
    assert path.name == "default.yaml"
    assert config["env"]["backend"] == "pyflyt"
    assert config["env"]["victim_count"] > 0


def test_yolo_default_model_points_to_repo_root():
    from uav_triage_rl.yolo import YoloDetector

    path = YoloDetector.resolve_model_path(None)
    assert path.name == "yolo11n.pt"
    assert path.parent.name == "uav-search-project"

