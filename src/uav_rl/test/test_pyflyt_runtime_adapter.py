import math

import numpy as np
import pytest

from uav_rl.detection_memory import (
    ConfirmationConfig,
    DetectionMemory,
    DetectionObservation,
)
from uav_rl.pyflyt_runtime_adapter import (
    PyFlytAdapterConfig,
    PyFlytRuntimeAdapter,
    PyFlytSearchAction,
    load_pyflyt_adapter_config,
)


def test_pyflyt_adapter_observation_contract_matches_default_model_shape():
    config = PyFlytAdapterConfig()
    adapter = PyFlytRuntimeAdapter(config)
    memory = DetectionMemory(
        ConfirmationConfig(
            min_hits=2,
            min_mean_confidence=0.55,
            min_viewpoint_separation_m=0.0,
            min_yaw_span_rad=0.0,
        )
    )
    memory.update(
        [DetectionObservation(class_name="person", confidence=0.8, x=8.0, y=1.0)],
        step=1,
        drone_x=0.0,
        drone_y=0.0,
        drone_yaw=0.0,
    )
    adapter.mark_view(drone_x=0.0, drone_y=0.0, drone_yaw=0.0, altitude_ned_z=-6.0)
    adapter.goal_for_action(
        action=int(PyFlytSearchAction.HOVER_SCAN),
        x=0.0,
        y=0.0,
        yaw=0.0,
        memory=memory,
    )

    obs = adapter.encode_observation(
        x=0.0,
        y=0.0,
        yaw=0.0,
        altitude_ned_z=-6.0,
        vx=0.0,
        vy=0.0,
        step=1,
        memory=memory,
    )

    assert obs.shape == (518,)
    assert obs.dtype == np.float32
    assert adapter.observation_size == config.observation_size == 518
    assert obs[-15 + 2] == pytest.approx(0.8)


def test_pyflyt_adapter_maps_investigate_action_to_standoff_goal():
    adapter = PyFlytRuntimeAdapter(PyFlytAdapterConfig())
    memory = DetectionMemory()
    memory.update(
        [DetectionObservation(class_name="person", confidence=0.9, x=8.0, y=0.0)],
        step=1,
        drone_x=0.0,
        drone_y=0.0,
        drone_yaw=0.0,
    )

    goal = adapter.goal_for_action(
        action=int(PyFlytSearchAction.INVESTIGATE_BEST),
        x=0.0,
        y=0.0,
        yaw=0.0,
        memory=memory,
    )

    assert goal.name == "investigate_best"
    assert goal.target_xy == pytest.approx((8.0, 0.0))
    assert math.hypot(goal.x - 8.0, goal.y) == pytest.approx(7.0)
    assert goal.yaw == pytest.approx(0.0)


def test_load_pyflyt_adapter_config_prefers_artifact_yaml(tmp_path):
    pytest.importorskip("yaml")
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    model_path = artifact_dir / "model.zip"
    model_path.write_bytes(b"not a real model")
    (artifact_dir / "config.yaml").write_text(
        """
env:
  width_m: 32.0
  height_m: 24.0
  cell_size_m: 2.0
  decision_period_s: 1.25
  patch_side: 9
camera:
  horizontal_fov_deg: 80.0
actions:
  scan_yaw_step_deg: 40.0
perception:
  min_hits: 3
  min_mean_confidence: 0.6
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config, source = load_pyflyt_adapter_config(
        model_path=model_path,
        artifact_dir=artifact_dir,
        config_path=None,
        fallback=PyFlytAdapterConfig(),
    )

    assert source == artifact_dir / "config.yaml"
    assert config.width_m == pytest.approx(32.0)
    assert config.height_m == pytest.approx(24.0)
    assert config.decision_period_s == pytest.approx(1.25)
    assert config.patch_side == 9
    assert config.action.scan_yaw_step_deg == pytest.approx(40.0)
    assert config.min_hits == 3
    assert config.min_mean_confidence == pytest.approx(0.6)
