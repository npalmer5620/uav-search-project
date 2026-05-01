"""Reward shaping for triage-search macro-actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    new_observed_cell_reward: float = 0.10
    uncertainty_reduction_reward: float = 0.04
    detection_update_reward: float = 1.0
    confirmed_victim_reward: float = 45.0
    mission_success_bonus: float = 90.0
    decision_penalty: float = 0.08
    revisit_penalty: float = 0.30
    collision_penalty: float = 15.0
    out_of_bounds_penalty: float = 10.0
    missed_required_victim_penalty: float = 10.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RewardConfig":
        return cls(**dict(raw or {}))


def compute_reward(events: dict[str, Any], config: RewardConfig) -> float:
    reward = -float(config.decision_penalty)
    reward += float(events.get("new_observed_cells", 0)) * config.new_observed_cell_reward
    reward += float(events.get("uncertainty_reduction", 0.0)) * config.uncertainty_reduction_reward
    reward += float(events.get("detection_updates", 0)) * config.detection_update_reward
    reward += float(events.get("new_confirmed_victims", 0)) * config.confirmed_victim_reward

    if events.get("revisit"):
        reward -= config.revisit_penalty
    if events.get("collision"):
        reward -= config.collision_penalty
    if events.get("out_of_bounds"):
        reward -= config.out_of_bounds_penalty
    if events.get("success"):
        reward += config.mission_success_bonus
    if events.get("truncated") and not events.get("success"):
        missed = max(0, int(events.get("required_victims", 0)) - int(events.get("confirmed_victims", 0)))
        reward -= missed * config.missed_required_victim_penalty
    return float(reward)

