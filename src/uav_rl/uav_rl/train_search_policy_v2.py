"""Train the V2 DQN search policy in the belief-map task environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import yaml

from uav_rl.rl_common import repo_root, resolve_repo_path
from uav_rl.search_task_env_v2 import SearchTaskConfigV2, SearchTaskEnvV2


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "search_policy_v2.yaml"


def load_config(path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_repo_path(path, default=default_config_path())
    with config_path.open("r", encoding="utf-8") as handle:
        return config_path, yaml.safe_load(handle) or {}


def tensorboard_is_available() -> bool:
    return importlib.util.find_spec("tensorboard") is not None


def merge_config(base: SearchTaskConfigV2, overrides: dict[str, Any] | None) -> SearchTaskConfigV2:
    merged = base.as_dict()
    merged.update(dict(overrides or {}))
    return SearchTaskConfigV2.from_mapping(merged)


def scale_stage_timesteps(weights: list[int], total_timesteps: int) -> list[int]:
    if not weights:
        return []
    weights = [max(1, int(weight)) for weight in weights]
    total_timesteps = max(len(weights), int(total_timesteps))
    raw = [total_timesteps * weight / sum(weights) for weight in weights]
    scaled = [max(1, int(value)) for value in raw]
    diff = total_timesteps - sum(scaled)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - scaled[idx], reverse=diff > 0)
    while diff != 0:
        changed = False
        for idx in order:
            if diff > 0:
                scaled[idx] += 1
                diff -= 1
                changed = True
            elif scaled[idx] > 1:
                scaled[idx] -= 1
                diff += 1
                changed = True
            if diff == 0:
                break
        if not changed:
            break
    return scaled


def build_training_stages(
    *,
    base_env_config: SearchTaskConfigV2,
    train_config: dict[str, Any],
    total_timesteps_override: int | None,
) -> list[dict[str, Any]]:
    curriculum = list(train_config.get("curriculum") or [])
    if not curriculum:
        return [
            {
                "name": "default",
                "config": base_env_config,
                "total_timesteps": int(total_timesteps_override or train_config.get("total_timesteps", 1_000_000)),
            }
        ]

    weights = [max(1, int(stage.get("total_timesteps", 1))) for stage in curriculum]
    total = int(total_timesteps_override or train_config.get("total_timesteps", sum(weights)))
    scaled = scale_stage_timesteps(weights, total)
    stages: list[dict[str, Any]] = []
    for idx, stage in enumerate(curriculum):
        stages.append(
            {
                "name": str(stage.get("name") or f"stage_{idx + 1}"),
                "config": merge_config(base_env_config, stage.get("env")),
                "total_timesteps": scaled[idx],
            }
        )
    return stages


def make_env(config: SearchTaskConfigV2, seed: int | None = None):
    from stable_baselines3.common.monitor import Monitor

    env = SearchTaskEnvV2(config=config)
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def run_episodes(
    *,
    config: SearchTaskConfigV2,
    episodes: int,
    policy: Callable[[SearchTaskEnvV2, Any], int],
) -> dict[str, Any]:
    import numpy as np

    rewards: list[float] = []
    coverages: list[float] = []
    confirmed: list[int] = []
    false_investigations = 0
    immediate_backtracks = 0
    reverse_moves = 0
    shielded_actions = 0
    successes = 0
    steps: list[int] = []
    backtracks: list[int] = []
    shielded_counts: list[int] = []
    first_confirm_steps: list[int] = []

    for seed in range(max(1, int(episodes))):
        env = SearchTaskEnvV2(config=config)
        obs, _info = env.reset(seed=seed)
        terminated = False
        truncated = False
        episode_reward = 0.0
        first_confirm = None
        while not (terminated or truncated):
            action = int(policy(env, obs))
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            if info.get("events", {}).get("false_investigation"):
                false_investigations += 1
            if info.get("events", {}).get("immediate_backtrack"):
                immediate_backtracks += 1
            if info.get("events", {}).get("reverse_move"):
                reverse_moves += 1
            if info.get("events", {}).get("shielded_action"):
                shielded_actions += 1
            if first_confirm is None and int(info.get("confirmed_targets", 0)) > 0:
                first_confirm = env.step_index
        rewards.append(episode_reward)
        coverages.append(float(info["coverage_fraction"]))
        confirmed.append(int(info["confirmed_targets"]))
        steps.append(env.step_index)
        backtracks.append(int(info.get("backtrack_count", 0)))
        shielded_counts.append(int(info.get("shielded_action_count", 0)))
        if info.get("success"):
            successes += 1
        if first_confirm is not None:
            first_confirm_steps.append(first_confirm)

    return {
        "episodes": max(1, int(episodes)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(successes / max(len(rewards), 1)),
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_confirmed_targets": float(np.mean(confirmed)) if confirmed else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "false_investigation_rate": float(false_investigations / max(sum(steps), 1)),
        "immediate_backtrack_rate": float(immediate_backtracks / max(sum(steps), 1)),
        "reverse_move_rate": float(reverse_moves / max(sum(steps), 1)),
        "shielded_action_rate": float(shielded_actions / max(sum(steps), 1)),
        "mean_backtracks": float(np.mean(backtracks)) if backtracks else 0.0,
        "mean_shielded_actions": float(np.mean(shielded_counts)) if shielded_counts else 0.0,
        "mean_time_to_first_confirm_step": (
            float(np.mean(first_confirm_steps)) if first_confirm_steps else None
        ),
    }


def evaluate_model(
    *,
    config: SearchTaskConfigV2,
    model: Any,
    episodes: int,
) -> dict[str, Any]:
    def policy(_env: SearchTaskEnvV2, obs):
        action, _state = model.predict(obs, deterministic=True)
        return int(action)

    return run_episodes(config=config, episodes=episodes, policy=policy)


def compare_baselines(
    *,
    config: SearchTaskConfigV2,
    model: Any | None,
    episodes: int,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {
        "random": run_episodes(
            config=config,
            episodes=episodes,
            policy=lambda env, _obs: int(env.action_space.sample()),
        ),
        "lawnmower": run_episodes(
            config=config,
            episodes=episodes,
            policy=lambda env, _obs: env.lawnmower_action(),
        ),
        "greedy_frontier": run_episodes(
            config=config,
            episodes=episodes,
            policy=lambda env, _obs: env.greedy_frontier_action(),
        ),
        "greedy_hybrid": run_episodes(
            config=config,
            episodes=episodes,
            policy=lambda env, _obs: env.greedy_hybrid_action(),
        ),
    }
    if model is not None:
        comparisons["dqn_policy"] = evaluate_model(
            config=config,
            model=model,
            episodes=episodes,
        )
    return comparisons


class ProgressCallbackFactory:
    def __init__(
        self,
        *,
        config: SearchTaskConfigV2,
        artifact_root: Path,
        eval_interval_timesteps: int,
        eval_episodes: int,
        stage_name: str,
        stage_index: int,
        num_stages: int,
        total_timesteps: int,
        clear_progress_file: bool,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def _on_training_start(self) -> None:
                outer.on_training_start()

            def _on_step(self) -> bool:
                return outer.on_step(self)

            def _on_training_end(self) -> None:
                outer.on_training_end(self)

        self.callback = _Callback()
        self.config = config
        self.artifact_root = artifact_root
        self.eval_interval_timesteps = max(0, int(eval_interval_timesteps))
        self.eval_episodes = max(1, int(eval_episodes))
        self.stage_name = stage_name
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.total_timesteps = max(1, int(total_timesteps))
        self.clear_progress_file = clear_progress_file
        self.progress_path = artifact_root / "training_progress.jsonl"
        self.started_at = 0.0
        self.next_eval = self.eval_interval_timesteps

    def _write(self, payload: dict[str, Any]) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        message = json.dumps(payload, sort_keys=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(f"TRAIN_PROGRESS {message}", flush=True)

    def on_training_start(self) -> None:
        if self.clear_progress_file and self.progress_path.exists():
            self.progress_path.unlink()
        self.started_at = time.perf_counter()
        self._write(
            {
                "event": "stage_start",
                "stage_name": self.stage_name,
                "stage_index": self.stage_index,
                "num_stages": self.num_stages,
                "eval_interval_timesteps": self.eval_interval_timesteps,
                "eval_episodes": self.eval_episodes,
            }
        )

    def _evaluate(self, callback: Any, *, event: str) -> None:
        metrics = evaluate_model(
            config=self.config,
            model=callback.model,
            episodes=self.eval_episodes,
        )
        self._write(
            {
                "event": event,
                "elapsed_wall_s": round(time.perf_counter() - self.started_at, 3),
                "stage_name": self.stage_name,
                "stage_index": self.stage_index,
                "num_stages": self.num_stages,
                "timesteps": int(callback.model.num_timesteps),
                "timesteps_fraction": round(callback.model.num_timesteps / self.total_timesteps, 4),
                **metrics,
            }
        )

    def on_step(self, callback: Any) -> bool:
        if self.eval_interval_timesteps <= 0:
            return True
        if int(callback.model.num_timesteps) >= self.next_eval:
            self._evaluate(callback, event="progress_eval")
            self.next_eval += self.eval_interval_timesteps
        return True

    def on_training_end(self, callback: Any) -> None:
        self._evaluate(callback, event="stage_end")


def train_search_policy_v2(
    *,
    config_path: str | Path | None = None,
    total_timesteps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    from gymnasium.utils.env_checker import check_env as gym_check_env
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env as sb3_check_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    resolved_config_path, raw = load_config(config_path)
    env_config = SearchTaskConfigV2.from_mapping(raw.get("env"))
    dqn_config = dict(raw.get("dqn") or {})
    train_config = dict(raw.get("train") or {})

    artifact_root = resolve_repo_path(
        artifact_dir or train_config.get("artifact_dir"),
        default=repo_root() / "artifacts" / "rl" / "search_policy_v2",
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    tensorboard_root = resolve_repo_path(
        train_config.get("tensorboard_log"),
        default=repo_root() / "artifacts" / "rl" / "tensorboard_v2",
    )
    tensorboard_log = str(tensorboard_root) if tensorboard_is_available() else None
    if tensorboard_log is not None:
        tensorboard_root.mkdir(parents=True, exist_ok=True)

    policy = dqn_config.pop("policy", "MlpPolicy")
    seed = int(dqn_config.pop("seed", 7))
    eval_episodes = max(1, int(train_config.get("eval_episodes", 20)))
    baseline_episodes = max(1, int(train_config.get("baseline_episodes", 100)))
    progress_eval_episodes = max(1, int(train_config.get("progress_eval_episodes", 5)))
    progress_interval = max(0, int(train_config.get("progress_eval_interval_timesteps", 50_000)))
    verbose = int(train_config.get("verbose", 1))
    log_interval = max(1, int(train_config.get("log_interval", 10)))

    stages = build_training_stages(
        base_env_config=env_config,
        train_config=train_config,
        total_timesteps_override=total_timesteps,
    )
    total = sum(int(stage["total_timesteps"]) for stage in stages)

    for stage in stages:
        gym_check_env(SearchTaskEnvV2(config=stage["config"]))
        sb3_check_env(SearchTaskEnvV2(config=stage["config"]), warn=True)

    model = None
    current_env = None
    summaries: list[dict[str, Any]] = []
    final_config = stages[-1]["config"]

    for index, stage in enumerate(stages):
        stage_config = stage["config"]
        stage_env = DummyVecEnv([lambda cfg=stage_config: make_env(cfg, seed=seed + index)])
        if model is None:
            model = DQN(
                policy,
                stage_env,
                verbose=verbose,
                tensorboard_log=tensorboard_log,
                seed=seed,
                **dqn_config,
            )
        else:
            if current_env is not None:
                current_env.close()
            model.set_env(stage_env)
        current_env = stage_env

        callback = ProgressCallbackFactory(
            config=stage_config,
            artifact_root=artifact_root,
            eval_interval_timesteps=progress_interval,
            eval_episodes=progress_eval_episodes,
            stage_name=str(stage["name"]),
            stage_index=index + 1,
            num_stages=len(stages),
            total_timesteps=total,
            clear_progress_file=index == 0,
        )
        model.learn(
            total_timesteps=int(stage["total_timesteps"]),
            callback=callback.callback,
            log_interval=log_interval,
            reset_num_timesteps=(index == 0),
        )
        stage_eval = evaluate_model(
            config=stage_config,
            model=model,
            episodes=eval_episodes,
        )
        summaries.append(
            {
                "name": str(stage["name"]),
                "index": index + 1,
                "total_timesteps": int(stage["total_timesteps"]),
                "env": stage_config.as_dict(),
                "evaluation": stage_eval,
            }
        )

    if model is None:
        raise RuntimeError("No DQN training stages were executed")

    model_path = artifact_root / "model.zip"
    summary_path = artifact_root / "training_summary.json"
    eval_path = artifact_root / "eval_metrics.json"
    comparison_path = artifact_root / "baseline_comparison.json"
    copied_config_path = artifact_root / resolved_config_path.name

    model.save(str(model_path))
    shutil.copyfile(resolved_config_path, copied_config_path)
    eval_metrics = evaluate_model(config=final_config, model=model, episodes=eval_episodes)
    comparison = compare_baselines(
        config=final_config,
        model=model,
        episodes=baseline_episodes,
    )
    eval_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    summary = {
        "algorithm": "DQN",
        "config_path": str(copied_config_path),
        "model_path": str(model_path),
        "progress_path": str(artifact_root / "training_progress.jsonl"),
        "eval_path": str(eval_path),
        "baseline_comparison_path": str(comparison_path),
        "total_timesteps": total,
        "stages": summaries,
        "evaluation": eval_metrics,
        "baseline_comparison": comparison,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if current_env is not None:
        current_env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to the V2 training YAML config")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override YAML timesteps")
    parser.add_argument("--artifact-dir", default=None, help="Override artifact output directory")
    args = parser.parse_args()
    summary = train_search_policy_v2(
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
