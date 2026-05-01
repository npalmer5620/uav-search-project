"""Lean PyFlyt triage-search RL package."""

from __future__ import annotations

ENV_ID = "UAVTriage/PyFlytSearch-v0"


def register_env() -> None:
    """Register the Gymnasium environment when Gymnasium is installed."""

    try:
        from gymnasium.envs.registration import register
        from gymnasium.error import Error as GymnasiumError
    except Exception:
        return

    try:
        register(
            id=ENV_ID,
            entry_point="uav_triage_rl.env:TriageSearchEnv",
        )
    except GymnasiumError as exc:
        if "Cannot re-register id" not in str(exc):
            raise


register_env()

__all__ = ["ENV_ID", "register_env"]

