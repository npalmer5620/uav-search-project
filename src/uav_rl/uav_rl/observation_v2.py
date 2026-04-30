"""Flat observation encoder shared by V2 training and runtime inference."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from uav_rl.belief_map import BeliefMap
from uav_rl.detection_memory import DetectionMemory


@dataclass(frozen=True)
class ObservationConfig:
    patch_side: int = 11
    top_k_detections: int = 3
    action_count: int = 10
    altitude_reference_m: float = 10.0
    speed_reference_m_s: float = 4.0
    max_episode_steps: int = 256

    @property
    def observation_size(self) -> int:
        patch_side = self.patch_side + (1 if self.patch_side % 2 == 0 else 0)
        return patch_side * patch_side * 6 + 10 + self.action_count + self.top_k_detections * 7 + 4


def encode_observation(
    *,
    belief: BeliefMap,
    memory: DetectionMemory,
    x: float,
    y: float,
    yaw: float,
    altitude_ned_z: float,
    vx: float,
    vy: float,
    step: int,
    config: ObservationConfig,
    last_action: int = -1,
) -> np.ndarray:
    patch = belief.local_patch(x, y, side=config.patch_side).reshape(-1)
    geom = belief.geometry
    x_half = max(geom.height / 2.0, 1e-6)
    y_half = max(geom.width / 2.0, 1e-6)
    speed = math.hypot(vx, vy)
    drone_state = np.array(
        [
            np.clip((x - geom.origin_x) / x_half, -1.0, 1.0),
            np.clip((y - geom.origin_y) / y_half, -1.0, 1.0),
            math.sin(yaw),
            math.cos(yaw),
            np.clip(abs(altitude_ned_z) / max(config.altitude_reference_m, 1.0), 0.0, 2.0),
            1.0 if altitude_ned_z < 0.0 else 0.0,
            np.clip(vx / max(config.speed_reference_m_s, 1.0), -1.0, 1.0),
            np.clip(vy / max(config.speed_reference_m_s, 1.0), -1.0, 1.0),
            np.clip(speed / max(config.speed_reference_m_s, 1.0), 0.0, 2.0),
            np.clip(step / max(float(config.max_episode_steps), 1.0), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    last_action_features = np.zeros(max(1, int(config.action_count)), dtype=np.float32)
    if 0 <= int(last_action) < last_action_features.shape[0]:
        last_action_features[int(last_action)] = 1.0
    det_features = memory.top_k_features(
        drone_x=x,
        drone_y=y,
        drone_yaw=yaw,
        step=step,
        k=config.top_k_detections,
        max_range_m=max(geom.diagonal_m, 1.0),
        age_norm_steps=max(config.max_episode_steps // 2, 1),
    ).reshape(-1)

    frontier = belief.nearest_frontier(x, y)
    if frontier is None:
        frontier_dist = 1.0
    else:
        frontier_dist = np.clip(math.hypot(frontier[0] - x, frontier[1] - y) / max(geom.diagonal_m, 1.0), 0.0, 1.0)

    best_detection = belief.best_victim_cell(x, y)
    if best_detection is None:
        detection_dist = 1.0
    else:
        detection_dist = np.clip(math.hypot(best_detection[0] - x, best_detection[1] - y) / max(geom.diagonal_m, 1.0), 0.0, 1.0)

    summary = np.array(
        [
            np.clip(belief.coverage_fraction, 0.0, 1.0),
            np.clip(memory.unconfirmed_count() / max(float(config.top_k_detections), 1.0), 0.0, 1.0),
            float(frontier_dist),
            float(detection_dist),
        ],
        dtype=np.float32,
    )
    return np.concatenate((patch, drone_state, last_action_features, det_features, summary)).astype(np.float32)
