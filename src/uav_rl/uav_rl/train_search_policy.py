"""Train a PPO search policy in the task-level Gymnasium environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import time
from typing import Any

from gymnasium.utils.env_checker import check_env as gym_check_env
import yaml

from uav_rl.rl_common import SearchTaskConfig, repo_root, resolve_repo_path
from uav_rl.search_task_env import SearchTaskEnv


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "search_policy.yaml"


def load_config(path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_repo_path(path, default=default_config_path())
    with config_path.open("r", encoding="utf-8") as handle:
        return config_path, yaml.safe_load(handle) or {}


def tensorboard_is_available() -> bool:
    return importlib.util.find_spec("tensorboard") is not None


def merge_search_task_config(
    base_config: SearchTaskConfig,
    overrides: dict[str, Any] | None,
) -> SearchTaskConfig:
    merged = dict(base_config.as_dict())
    merged.update(dict(overrides or {}))
    return SearchTaskConfig.from_mapping(merged)


def scale_stage_timesteps(weights: list[int], total_timesteps: int) -> list[int]:
    if not weights:
        return []
    sanitized = [max(1, int(weight)) for weight in weights]
    total_timesteps = max(len(sanitized), int(total_timesteps))
    weight_sum = sum(sanitized)

    scaled = [max(1, int(total_timesteps * weight / weight_sum)) for weight in sanitized]
    diff = total_timesteps - sum(scaled)
    if diff == 0:
        return scaled

    fractional_order = sorted(
        range(len(sanitized)),
        key=lambda idx: (total_timesteps * sanitized[idx] / weight_sum) - scaled[idx],
        reverse=(diff > 0),
    )

    while diff != 0:
        changed = False
        for idx in fractional_order:
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
    base_env_config: SearchTaskConfig,
    train_config: dict[str, Any],
    total_timesteps_override: int | None,
    eval_episodes: int,
    progress_eval_episodes: int,
    progress_eval_interval_timesteps: int,
) -> list[dict[str, Any]]:
    curriculum = list(train_config.get("curriculum") or [])
    if not curriculum:
        stage_total_timesteps = int(
            total_timesteps_override or train_config.get("total_timesteps", 50000)
        )
        return [
            {
                "name": "default",
                "config": base_env_config,
                "total_timesteps": max(1, stage_total_timesteps),
                "eval_episodes": eval_episodes,
                "progress_eval_episodes": progress_eval_episodes,
                "progress_eval_interval_timesteps": progress_eval_interval_timesteps,
            }
        ]

    stage_weights = [max(1, int(stage.get("total_timesteps", 1))) for stage in curriculum]
    desired_total_timesteps = int(
        total_timesteps_override
        or train_config.get("total_timesteps", sum(stage_weights))
    )
    scaled_stage_timesteps = scale_stage_timesteps(stage_weights, desired_total_timesteps)

    stages: list[dict[str, Any]] = []
    for index, stage in enumerate(curriculum):
        stage_config = merge_search_task_config(base_env_config, stage.get("env"))
        stages.append(
            {
                "name": str(stage.get("name") or f"stage_{index + 1}"),
                "config": stage_config,
                "total_timesteps": scaled_stage_timesteps[index],
                "eval_episodes": max(1, int(stage.get("eval_episodes", eval_episodes))),
                "progress_eval_episodes": max(
                    1,
                    int(stage.get("progress_eval_episodes", progress_eval_episodes)),
                ),
                "progress_eval_interval_timesteps": max(
                    0,
                    int(
                        stage.get(
                            "progress_eval_interval_timesteps",
                            progress_eval_interval_timesteps,
                        )
                    ),
                ),
            }
        )

    return stages


def make_env_factory(config: SearchTaskConfig, seed: int):
    from stable_baselines3.common.monitor import Monitor

    def factory():
        env = SearchTaskEnv(config=config)
        env.reset(seed=seed)
        return Monitor(env)

    return factory


def evaluate_model(
    *,
    config: SearchTaskConfig,
    model: Any,
    vec_normalize: Any,
    episodes: int,
) -> dict[str, Any]:
    import numpy as np

    rewards: list[float] = []
    successes = 0
    coverages: list[float] = []

    restore_training = None
    restore_norm_reward = None
    if vec_normalize is not None:
        restore_training = vec_normalize.training
        restore_norm_reward = vec_normalize.norm_reward
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    try:
        for episode_seed in range(episodes):
            env = SearchTaskEnv(config=config)
            obs, info = env.reset(seed=episode_seed)
            episode_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                inference_obs = obs
                if vec_normalize is not None:
                    inference_obs = vec_normalize.normalize_obs(obs.reshape(1, -1)).squeeze(0)
                action, _ = model.predict(inference_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
            rewards.append(episode_reward)
            coverages.append(float(info["coverage_fraction"]))
            if info.get("success"):
                successes += 1
    finally:
        if vec_normalize is not None:
            vec_normalize.training = restore_training
            vec_normalize.norm_reward = restore_norm_reward

    rewards_array = np.asarray(rewards, dtype=float)
    coverages_array = np.asarray(coverages, dtype=float)
    return {
        "episodes": episodes,
        "mean_reward": float(rewards_array.mean()) if rewards else 0.0,
        "success_rate": float(successes / max(episodes, 1)),
        "mean_coverage": float(coverages_array.mean()) if coverages else 0.0,
    }


def evaluate_artifact(
    *,
    config: SearchTaskConfig,
    model_path: Path,
    vecnormalize_path: Path,
    episodes: int,
) -> dict[str, Any]:
    from stable_baselines3 import PPO

    from uav_rl.rl_common import load_vecnormalize_for_inference

    model = PPO.load(str(model_path))
    vec_normalize = load_vecnormalize_for_inference(vecnormalize_path, config)
    return evaluate_model(
        config=config,
        model=model,
        vec_normalize=vec_normalize,
        episodes=episodes,
    )


class TrainingProgressCallback:
    """Emit periodic task-level evaluation metrics during PPO training."""

    def __init__(
        self,
        *,
        config: SearchTaskConfig,
        artifact_root: Path,
        total_timesteps: int,
        eval_interval_timesteps: int,
        eval_episodes: int,
        stage_name: str,
        stage_index: int,
        num_stages: int,
        stage_total_timesteps: int,
        stage_start_timesteps: int,
        clear_progress_file: bool,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "TrainingProgressCallback") -> None:
                super().__init__()
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.on_training_start()

            def _on_step(self) -> bool:
                return self.outer.on_step(self)

            def _on_training_end(self) -> None:
                self.outer.on_training_end(self)

        self.callback = _Callback(self)
        self.config = config
        self.artifact_root = artifact_root
        self.total_timesteps = total_timesteps
        self.eval_interval_timesteps = max(0, int(eval_interval_timesteps))
        self.eval_episodes = max(1, int(eval_episodes))
        self.stage_name = stage_name
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.stage_total_timesteps = max(1, int(stage_total_timesteps))
        self.stage_start_timesteps = max(0, int(stage_start_timesteps))
        self.clear_progress_file = clear_progress_file
        self.progress_path = self.artifact_root / "training_progress.jsonl"
        self.started_at = 0.0
        self.next_eval_timesteps = self.stage_start_timesteps + self.eval_interval_timesteps

    def _write_progress(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload, sort_keys=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(f"TRAIN_PROGRESS {message}", flush=True)

    def on_training_start(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if self.clear_progress_file and self.progress_path.exists():
            self.progress_path.unlink()
        self.started_at = time.perf_counter()
        self._write_progress(
            {
                "event": "stage_start",
                "eval_episodes": self.eval_episodes,
                "eval_interval_timesteps": self.eval_interval_timesteps,
                "total_timesteps": self.total_timesteps,
                "stage_index": self.stage_index,
                "num_stages": self.num_stages,
                "stage_name": self.stage_name,
                "stage_total_timesteps": self.stage_total_timesteps,
            }
        )

    def _evaluate(self, callback: Any, *, event: str) -> None:
        model = callback.model
        vec_normalize = None
        if hasattr(model, "get_vec_normalize_env"):
            vec_normalize = model.get_vec_normalize_env()

        metrics = evaluate_model(
            config=self.config,
            model=model,
            vec_normalize=vec_normalize,
            episodes=self.eval_episodes,
        )
        elapsed_s = time.perf_counter() - self.started_at
        stage_timesteps = max(0, int(model.num_timesteps) - self.stage_start_timesteps)
        payload = {
            "event": event,
            "elapsed_wall_s": round(elapsed_s, 3),
            "stage_index": self.stage_index,
            "num_stages": self.num_stages,
            "stage_name": self.stage_name,
            "timesteps": int(model.num_timesteps),
            "timesteps_fraction": round(
                float(model.num_timesteps) / max(float(self.total_timesteps), 1.0),
                4,
            ),
            "stage_timesteps": stage_timesteps,
            "stage_timesteps_fraction": round(
                float(stage_timesteps) / max(float(self.stage_total_timesteps), 1.0),
                4,
            ),
            **metrics,
        }
        self._write_progress(payload)

    def on_step(self, callback: Any) -> bool:
        if self.eval_interval_timesteps <= 0:
            return True
        while int(callback.model.num_timesteps) >= self.next_eval_timesteps:
            self._evaluate(callback, event="progress_eval")
            self.next_eval_timesteps += self.eval_interval_timesteps
        return True

    def on_training_end(self, callback: Any) -> None:
        self._evaluate(callback, event="stage_end")


def train_search_policy(
    *,
    config_path: str | Path | None = None,
    total_timesteps: int | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env as sb3_check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    resolved_config_path, raw = load_config(config_path)
    env_config = SearchTaskConfig.from_mapping(raw.get("env"))
    ppo_config = dict(raw.get("ppo") or {})
    train_config = dict(raw.get("train") or {})

    artifact_root = resolve_repo_path(
        artifact_dir or train_config.get("artifact_dir"),
        default=repo_root() / "artifacts" / "rl" / "search_policy",
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    tensorboard_root = resolve_repo_path(
        train_config.get("tensorboard_log"),
        default=repo_root() / "artifacts" / "rl" / "tensorboard",
    )
    tensorboard_log = str(tensorboard_root) if tensorboard_is_available() else None
    if tensorboard_log is not None:
        tensorboard_root.mkdir(parents=True, exist_ok=True)

    policy = ppo_config.pop("policy", "MlpPolicy")
    seed = int(ppo_config.get("seed", 7))
    total_timesteps = int(
        total_timesteps or train_config.get("total_timesteps", 50000)
    )
    num_envs = max(1, int(train_config.get("num_envs", 4)))
    eval_episodes = max(1, int(train_config.get("eval_episodes", 5)))
    progress_eval_episodes = max(
        1,
        int(train_config.get("progress_eval_episodes", eval_episodes)),
    )
    progress_eval_interval_timesteps = max(
        0,
        int(train_config.get("progress_eval_interval_timesteps", 5000)),
    )
    verbose = int(train_config.get("verbose", 1))
    log_interval = max(1, int(train_config.get("log_interval", 1)))

    stages = build_training_stages(
        base_env_config=env_config,
        train_config=train_config,
        total_timesteps_override=total_timesteps,
        eval_episodes=eval_episodes,
        progress_eval_episodes=progress_eval_episodes,
        progress_eval_interval_timesteps=progress_eval_interval_timesteps,
    )
    total_timesteps = sum(int(stage["total_timesteps"]) for stage in stages)

    for stage in stages:
        stage_config = stage["config"]
        gym_check_env(SearchTaskEnv(config=stage_config))
        sb3_check_env(SearchTaskEnv(config=stage_config), warn=True)

    model = None
    current_vec_env = None
    stage_summaries: list[dict[str, Any]] = []
    final_config = stages[-1]["config"]

    for index, stage in enumerate(stages):
        stage_config = stage["config"]
        stage_total_timesteps = int(stage["total_timesteps"])
        stage_vec_env = DummyVecEnv(
            [make_env_factory(stage_config, seed + env_idx) for env_idx in range(num_envs)]
        )
        stage_vec_env = VecNormalize(
            stage_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

        if model is None:
            model = PPO(
                policy,
                stage_vec_env,
                verbose=verbose,
                tensorboard_log=tensorboard_log,
                **ppo_config,
            )
        else:
            previous_vec_env = current_vec_env
            model.set_env(stage_vec_env)
            if previous_vec_env is not None:
                previous_vec_env.close()

        current_vec_env = stage_vec_env
        progress_callback = TrainingProgressCallback(
            config=stage_config,
            artifact_root=artifact_root,
            total_timesteps=total_timesteps,
            eval_interval_timesteps=int(stage["progress_eval_interval_timesteps"]),
            eval_episodes=int(stage["progress_eval_episodes"]),
            stage_name=str(stage["name"]),
            stage_index=index + 1,
            num_stages=len(stages),
            stage_total_timesteps=stage_total_timesteps,
            stage_start_timesteps=0 if model is None else int(model.num_timesteps),
            clear_progress_file=(index == 0),
        )
        model.learn(
            total_timesteps=stage_total_timesteps,
            callback=progress_callback.callback,
            log_interval=log_interval,
            reset_num_timesteps=(index == 0),
        )
        stage_eval = evaluate_model(
            config=stage_config,
            model=model,
            vec_normalize=model.get_vec_normalize_env(),
            episodes=int(stage["eval_episodes"]),
        )
        stage_summaries.append(
            {
                "name": str(stage["name"]),
                "index": index + 1,
                "total_timesteps": stage_total_timesteps,
                "env": stage_config.as_dict(),
                "evaluation": stage_eval,
            }
        )

    model_path = artifact_root / "model.zip"
    vecnormalize_path = artifact_root / "vecnormalize.pkl"
    summary_path = artifact_root / "training_summary.json"
    eval_path = artifact_root / "eval_metrics.json"
    progress_path = artifact_root / "training_progress.jsonl"
    copied_config_path = artifact_root / resolved_config_path.name

    if model is None or current_vec_env is None:
        raise RuntimeError("No training stages were executed")

    model.save(str(model_path))
    current_vec_env.save(str(vecnormalize_path))
    shutil.copyfile(resolved_config_path, copied_config_path)

    eval_metrics = evaluate_artifact(
        config=final_config,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        episodes=eval_episodes,
    )
    eval_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    summary = {
        "config_path": str(copied_config_path),
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path),
        "progress_path": str(progress_path),
        "total_timesteps": total_timesteps,
        "num_envs": num_envs,
        "stages": stage_summaries,
        "evaluation": eval_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to the training YAML config")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the YAML total timesteps",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Override the YAML artifact output directory",
    )
    args = parser.parse_args()

    summary = train_search_policy(
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
