from __future__ import annotations

import numpy as np

from uav_triage_rl.bbox import BBoxPerception, BBoxSimConfig
from uav_triage_rl.perception import CameraConfig, PerceptionConfig
from uav_triage_rl.world import VictimSpec


def test_bbox_perception_projects_noisy_boxes():
    camera = CameraConfig()
    perception = PerceptionConfig(
        mode="bbox",
        detection_probability=1.0,
        false_positive_rate=0.0,
    )
    bbox = BBoxPerception(
        camera=camera,
        perception=perception,
        config=BBoxSimConfig(dropout_rate=0.0, center_jitter_px=2.0),
    )
    victims = [VictimSpec(id=0, x=4.2, y=0.0, yaw=0.0)]
    observations = bbox.detect(
        victims,
        drone_x=0.0,
        drone_y=0.0,
        drone_z=6.0,
        drone_yaw=0.0,
        rng=np.random.default_rng(3),
        investigating=False,
        visible_cells=[],
        geometry=None,
    )

    assert len(observations) == 1
    assert observations[0].source == "bbox"
    assert observations[0].bbox_xyxy is not None
    x1, y1, x2, y2 = observations[0].bbox_xyxy
    assert 0.0 <= x1 < x2 <= camera.image_width_px - 1
    assert 0.0 <= y1 < y2 <= camera.image_height_px - 1
    assert bbox.last_bboxes[0].truth_id == 0

