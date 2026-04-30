"""RL task environments and mission controller."""

try:
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover - optional until RL deps are installed
    register = None


if register is not None:  # pragma: no branch
    try:
        register(
            id="UavSearchTask-v0",
            entry_point="uav_rl.search_task_env:SearchTaskEnv",
        )
        register(
            id="UavSearchTask-v2",
            entry_point="uav_rl.search_task_env_v2:SearchTaskEnvV2",
        )
    except Exception:
        pass
