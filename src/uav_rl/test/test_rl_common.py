import math

import numpy as np

from uav_rl.rl_common import (
    CoverageMap,
    ObservationEncoder,
    SearchTaskConfig,
    clip_action_to_deltas,
    load_world_target_classes,
)


def test_load_world_target_classes_matches_search_area_mix():
    classes = load_world_target_classes(None)
    assert len(classes) == 13
    assert classes.count("person") == 12
    assert classes.count("car") == 1


def test_coverage_map_marks_new_cells_once():
    config = SearchTaskConfig(width=4.0, height=4.0, coverage_grid_side=4)
    coverage = CoverageMap(config.coverage_grid_side, config)

    first_delta = coverage.mark_point(-1.5, -1.5)
    second_delta = coverage.mark_point(-1.5, -1.5)
    segment_delta = coverage.update_segment(-1.5, -1.5, 1.5, -1.5)

    assert first_delta == 1.0 / 16.0
    assert second_delta == 0.0
    assert segment_delta > 0.0
    assert coverage.coverage_fraction > first_delta


def test_observation_encoder_normalizes_state_and_appends_coverage():
    config = SearchTaskConfig(
        width=40.0,
        height=20.0,
        origin_x=5.0,
        origin_y=-2.0,
        coverage_grid_side=4,
        max_step_xy_m=4.0,
        search_speed=2.0,
        max_episode_steps=100,
    )
    coverage = CoverageMap(config.coverage_grid_side, config)
    coverage.mark_point(5.0, -2.0)
    encoder = ObservationEncoder(config, coverage, decision_period_s=0.5)

    obs = encoder.encode(
        x=10.0,
        y=3.0,
        vx=8.0,
        vy=-8.0,
        yaw=math.pi / 2.0,
        elapsed_s=25.0,
    )

    assert obs.shape == (6 + 16,)
    assert np.isclose(obs[0], 0.5)
    assert np.isclose(obs[1], 0.25)
    assert obs[2] == 1.0
    assert obs[3] == -1.0
    assert np.isclose(obs[4], 0.5)
    assert np.isclose(obs[5], 0.5)
    assert obs[6:].sum() == 1.0


def test_clip_action_to_deltas_respects_limits():
    dx, dy, dyaw = clip_action_to_deltas(
        np.array([2.0, -0.5, -3.0], dtype=np.float32),
        max_step_xy_m=4.0,
        max_yaw_step_rad=math.pi / 4.0,
    )

    assert dx == 4.0
    assert dy == -2.0
    assert np.isclose(dyaw, -(math.pi / 4.0))
