#!/usr/bin/env python3
"""Train the lean PyFlyt triage-search policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from uav_triage_rl.config import load_config, package_root, resolve_path, write_yaml
from uav_triage_rl.env import TriageSearchEnv


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def evaluate_policy(model: Any, config: dict[str, Any], *, episodes: int) -> dict[str, Any]:
    rewards: list[float] = []
    coverages: list[float] = []
    confirmed: list[float] = []
    successes = 0
    steps: list[float] = []
    for seed in range(max(1, int(episodes))):
        eval_config = dict(config)
        eval_config["env"] = {**dict(config.get("env", {})), "backend": "kinematic"}
        env = TriageSearchEnv(config=eval_config)
        obs, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        info: dict[str, Any] = {}
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            done = bool(terminated or truncated)
        rewards.append(episode_reward)
        coverages.append(float(info.get("coverage_fraction", 0.0)))
        confirmed.append(float(info.get("confirmed_victims", 0)))
        steps.append(float(info.get("step", 0)))
        successes += int(bool(info.get("success")))
        env.close()
    return {
        "episodes": max(1, int(episodes)),
        "mean_reward": _mean(rewards),
        "success_rate": successes / max(1, len(rewards)),
        "mean_coverage": _mean(coverages),
        "mean_confirmed_victims": _mean(confirmed),
        "mean_steps": _mean(steps),
    }


def train(config_path: str | Path | None, *, timesteps: int | None, artifact_dir: str | Path | None,
          progress_bar: bool, perception_mode: str | None) -> dict[str, Any]:
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    resolved_config_path, config = load_config(config_path)
    if perception_mode is not None:
        config["perception"] = {**dict(config.get("perception", {}) or {}), "mode": perception_mode}
    train_config = dict(config.get("train", {}) or {})
    dqn_config = dict(config.get("dqn", {}) or {})
    total_timesteps = int(timesteps or train_config.get("total_timesteps", 50000))
    artifact_root = resolve_path(
        artifact_dir or train_config.get("artifact_dir", "artifacts/latest"),
        base=package_root(),
    )
    (artifact_root / "monitor").mkdir(parents=True, exist_ok=True)

    env = Monitor(TriageSearchEnv(config=config), filename=str(artifact_root / "monitor" / "train"))
    policy = str(dqn_config.pop("policy", "MlpPolicy"))
    seed = int(dqn_config.pop("seed", 7))
    started = time.perf_counter()
    model = DQN(
        policy,
        env,
        verbose=1,
        device="cpu",
        seed=seed,
        **dqn_config,
    )
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=int(train_config.get("log_interval", 10)),
        progress_bar=progress_bar,
    )
    model_path = artifact_root / "model.zip"
    model.save(str(model_path))
    env.close()

    copied_config = artifact_root / "config.yaml"
    write_yaml(copied_config, config)
    eval_metrics = evaluate_policy(
        model,
        config,
        episodes=int(train_config.get("eval_episodes", 10)),
    )
    (artifact_root / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    summary = {
        "algorithm": "DQN",
        "config_source": str(resolved_config_path),
        "config_path": str(copied_config),
        "model_path": str(model_path),
        "perception_mode": str(config.get("perception", {}).get("mode", "bbox")),
        "total_timesteps": total_timesteps,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "evaluation": eval_metrics,
    }
    (artifact_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument(
        "--perception",
        choices=["bbox", "point"],
        default=None,
        help="Override perception.mode for training. Default comes from config.",
    )
    args = parser.parse_args()
    summary = train(
        args.config,
        timesteps=args.timesteps,
        artifact_dir=args.artifact_dir,
        progress_bar=not args.no_progress_bar,
        perception_mode=args.perception,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
