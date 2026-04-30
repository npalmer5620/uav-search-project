"""Reward accounting for V2 search training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    new_observed_cell_reward: float = 0.2
    uncertainty_reduction_reward: float = 0.05
    useful_reobserve_reward: float = 3.0
    confirmed_victim_reward: float = 60.0
    mission_success_bonus: float = 120.0
    decision_penalty: float = 0.1
    useless_revisit_penalty: float = 2.0
    false_investigation_penalty: float = 8.0
    immediate_backtrack_penalty: float = 3.0
    reverse_move_penalty: float = 0.5
    shielded_action_penalty: float = 1.0
    collision_penalty: float = 20.0
    unsafe_altitude_penalty: float = 20.0
    out_of_bounds_penalty: float = 20.0
    missed_required_target_penalty: float = 15.0


@dataclass(frozen=True)
class SearchEvents:
    new_observed_cells: int = 0
    uncertainty_reduction: float = 0.0
    useful_reobservations: int = 0
    new_confirmed_victims: int = 0
    useless_revisit: bool = False
    false_investigation: bool = False
    immediate_backtrack: bool = False
    reverse_move: bool = False
    shielded_action: bool = False
    collision: bool = False
    unsafe_altitude: bool = False
    out_of_bounds: bool = False
    mission_success: bool = False
    missed_required_targets: int = 0


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    reward_new_observed: float = 0.0
    reward_uncertainty_reduction: float = 0.0
    reward_useful_reobserve: float = 0.0
    reward_confirmed_victim: float = 0.0
    reward_mission_success: float = 0.0
    penalty_decision: float = 0.0
    penalty_useless_revisit: float = 0.0
    penalty_false_investigation: float = 0.0
    penalty_immediate_backtrack: float = 0.0
    penalty_reverse_move: float = 0.0
    penalty_shielded_action: float = 0.0
    penalty_collision: float = 0.0
    penalty_unsafe_altitude: float = 0.0
    penalty_out_of_bounds: float = 0.0
    penalty_missed_required_targets: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_search_reward(
    events: SearchEvents,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    cfg = config or RewardConfig()
    reward_new_observed = cfg.new_observed_cell_reward * float(events.new_observed_cells)
    reward_uncertainty = cfg.uncertainty_reduction_reward * float(events.uncertainty_reduction)
    reward_useful_reobserve = cfg.useful_reobserve_reward * float(events.useful_reobservations)
    reward_confirmed = cfg.confirmed_victim_reward * float(events.new_confirmed_victims)
    reward_success = cfg.mission_success_bonus if events.mission_success else 0.0

    penalty_decision = cfg.decision_penalty
    penalty_useless = cfg.useless_revisit_penalty if events.useless_revisit else 0.0
    penalty_false = cfg.false_investigation_penalty if events.false_investigation else 0.0
    penalty_backtrack = cfg.immediate_backtrack_penalty if events.immediate_backtrack else 0.0
    penalty_reverse = cfg.reverse_move_penalty if events.reverse_move else 0.0
    penalty_shielded = cfg.shielded_action_penalty if events.shielded_action else 0.0
    penalty_collision = cfg.collision_penalty if events.collision else 0.0
    penalty_unsafe = cfg.unsafe_altitude_penalty if events.unsafe_altitude else 0.0
    penalty_oob = cfg.out_of_bounds_penalty if events.out_of_bounds else 0.0
    penalty_missed = cfg.missed_required_target_penalty * float(
        max(0, int(events.missed_required_targets))
    )

    total = (
        reward_new_observed
        + reward_uncertainty
        + reward_useful_reobserve
        + reward_confirmed
        + reward_success
        - penalty_decision
        - penalty_useless
        - penalty_false
        - penalty_backtrack
        - penalty_reverse
        - penalty_shielded
        - penalty_collision
        - penalty_unsafe
        - penalty_oob
        - penalty_missed
    )
    return RewardBreakdown(
        total=float(total),
        reward_new_observed=float(reward_new_observed),
        reward_uncertainty_reduction=float(reward_uncertainty),
        reward_useful_reobserve=float(reward_useful_reobserve),
        reward_confirmed_victim=float(reward_confirmed),
        reward_mission_success=float(reward_success),
        penalty_decision=float(penalty_decision),
        penalty_useless_revisit=float(penalty_useless),
        penalty_false_investigation=float(penalty_false),
        penalty_immediate_backtrack=float(penalty_backtrack),
        penalty_reverse_move=float(penalty_reverse),
        penalty_shielded_action=float(penalty_shielded),
        penalty_collision=float(penalty_collision),
        penalty_unsafe_altitude=float(penalty_unsafe),
        penalty_out_of_bounds=float(penalty_oob),
        penalty_missed_required_targets=float(penalty_missed),
    )
