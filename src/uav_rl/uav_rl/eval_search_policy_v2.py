"""Evaluate a saved V2 DQN search policy and compare baselines."""

from __future__ import annotations

import argparse
import json

import yaml

from uav_rl.rl_common import resolve_repo_path
from uav_rl.search_task_env_v2 import SearchTaskConfigV2
from uav_rl.train_search_policy_v2 import (
    compare_baselines,
    default_config_path,
    evaluate_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to V2 training YAML config")
    parser.add_argument(
        "--model-path",
        default="artifacts/rl/search_policy_v2/model.zip",
        help="Saved DQN model path",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=100,
        help="Episodes for baseline comparison",
    )
    parser.add_argument(
        "--output",
        default="artifacts/rl/search_policy_v2/eval_metrics.json",
        help="Where to write evaluation metrics JSON",
    )
    parser.add_argument(
        "--comparison-output",
        default="artifacts/rl/search_policy_v2/baseline_comparison.json",
        help="Where to write baseline comparison JSON",
    )
    args = parser.parse_args()

    from stable_baselines3 import DQN

    config_path = resolve_repo_path(args.config, default=default_config_path())
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = SearchTaskConfigV2.from_mapping(raw.get("env"))
    model = DQN.load(str(resolve_repo_path(args.model_path)))

    metrics = evaluate_model(config=config, model=model, episodes=max(1, int(args.episodes)))
    comparison = compare_baselines(
        config=config,
        model=model,
        episodes=max(1, int(args.baseline_episodes)),
    )

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison_path = resolve_repo_path(args.comparison_output)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(json.dumps({"evaluation": metrics, "baseline_comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
