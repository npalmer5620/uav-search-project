from __future__ import annotations

import math

import numpy as np

from uav_triage_rl.world import WorldConfig, sample_layout


def test_sample_layout_places_prone_victims_with_spacing():
    config = WorldConfig(victim_count=5, obstacle_count=3, min_victim_spacing_m=3.0)
    layout = sample_layout(config, np.random.default_rng(10), start_xy=(0.0, 0.0))
    assert len(layout.victims) == 5
    assert len(layout.obstacles) == 3
    for idx, left in enumerate(layout.victims):
        assert config.x_limits[0] <= left.x <= config.x_limits[1]
        assert config.y_limits[0] <= left.y <= config.y_limits[1]
        for right in layout.victims[idx + 1:]:
            assert math.hypot(left.x - right.x, left.y - right.y) >= 3.0

