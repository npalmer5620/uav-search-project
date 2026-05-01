#!/usr/bin/env python3
"""Evaluate a trained triage-search policy and simple baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from uav_triage_rl.config import load_config, package_root, resolve_path
from uav_triage_rl.env import TriageSearchEnv
from uav_triage_rl.yolo import YoloDetector, detections_to_jsonable


def run_episodes(config: dict[str, Any], *, episodes: int, policy: Callable[[TriageSearchEnv, Any], int]) -> dict[str, Any]:
    rewards: list[float] = []
    coverage: list[float] = []
    confirmed: list[float] = []
    successes = 0
    steps: list[float] = []
    for seed in range(max(1, int(episodes))):
        env = TriageSearchEnv(config=config)
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        info: dict[str, Any] = {}
        while not done:
            action = int(policy(env, obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total_reward)
        coverage.append(float(info.get("coverage_fraction", 0.0)))
        confirmed.append(float(info.get("confirmed_victims", 0.0)))
        successes += int(bool(info.get("success")))
        steps.append(float(info.get("step", 0)))
        env.close()
    denom = max(1, len(rewards))
    return {
        "episodes": max(1, int(episodes)),
        "mean_reward": sum(rewards) / denom,
        "success_rate": successes / denom,
        "mean_coverage": sum(coverage) / denom,
        "mean_confirmed_victims": sum(confirmed) / denom,
        "mean_steps": sum(steps) / denom,
    }


def evaluate(config_path: str | Path | None, *, model_path: str | Path, episodes: int,
             output_path: str | Path | None, yolo: bool, yolo_episodes: int,
             perception_mode: str | None) -> dict[str, Any]:
    from stable_baselines3 import DQN

    _, config = load_config(config_path)
    if perception_mode is not None:
        config["perception"] = {**dict(config.get("perception", {}) or {}), "mode": perception_mode}
    # Evaluation defaults to the fast kinematic backend unless the config
    # explicitly asks for PyFlyt. This keeps baseline comparisons quick.
    config["env"] = {**dict(config.get("env", {})), "backend": dict(config.get("env", {})).get("eval_backend", "kinematic")}
    resolved_model = resolve_path(model_path, base=package_root())
    model = DQN.load(str(resolved_model), device="cpu")
    results = {
        "model_path": str(resolved_model),
        "perception_mode": str(config.get("perception", {}).get("mode", "bbox")),
        "dqn_policy": run_episodes(
            config,
            episodes=episodes,
            policy=lambda _env, obs: int(model.predict(obs, deterministic=True)[0]),
        ),
        "random": run_episodes(
            config,
            episodes=episodes,
            policy=lambda env, _obs: int(env.action_space.sample()),
        ),
        "lawnmower": run_episodes(
            config,
            episodes=episodes,
            policy=lambda env, _obs: env.lawnmower_action(),
        ),
    }
    if yolo:
        sim_config = dict(config.get("sim", {}) or {})
        detector = YoloDetector(
            sim_config.get("yolo_model_path", "../yolo11n.pt"),
            confidence=float(sim_config.get("yolo_confidence", 0.30)),
        )
        yolo_config = dict(config)
        yolo_config["env"] = {**dict(config.get("env", {})), "backend": "pyflyt"}
        interval = max(1, int(sim_config.get("yolo_interval_steps", 3)))
        events: list[dict[str, Any]] = []
        for seed in range(max(1, int(yolo_episodes))):
            env = TriageSearchEnv(config=yolo_config, render_mode="rgb_array")
            obs, _ = env.reset(seed=seed)
            done = False
            step = 0
            while not done:
                action = int(model.predict(obs, deterministic=True)[0])
                obs, _reward, terminated, truncated, info = env.step(action)
                if step % interval == 0:
                    frame = env.render()
                    if frame is not None:
                        detections = detector.detect(frame)
                        events.append(
                            {
                                "episode": seed,
                                "step": step,
                                "detections": detections_to_jsonable(detections),
                            }
                        )
                step += 1
                done = bool(terminated or truncated)
            env.close()
        results["yolo_diagnostics"] = {
            "episodes": max(1, int(yolo_episodes)),
            "events": events,
            "total_detections": sum(len(event["detections"]) for event in events),
        }
    if output_path:
        target = resolve_path(output_path, base=package_root())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--yolo", action="store_true", help="Run a small PyFlyt visual YOLO diagnostic pass")
    parser.add_argument("--yolo-episodes", type=int, default=1)
    parser.add_argument("--perception", choices=["bbox", "point"], default=None)
    args = parser.parse_args()
    results = evaluate(
        args.config,
        model_path=args.model,
        episodes=args.episodes,
        output_path=args.output,
        yolo=args.yolo,
        yolo_episodes=args.yolo_episodes,
        perception_mode=args.perception,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
