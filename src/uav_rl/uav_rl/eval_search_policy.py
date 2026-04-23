"""Evaluate a saved PPO search policy against the task environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from uav_rl.rl_common import SearchTaskConfig, resolve_repo_path
from uav_rl.train_search_policy import default_config_path, evaluate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to the training YAML config")
    parser.add_argument(
        "--model-path",
        default="artifacts/rl/search_policy/model.zip",
        help="Saved PPO model path",
    )
    parser.add_argument(
        "--vecnormalize-path",
        default="artifacts/rl/search_policy/vecnormalize.pkl",
        help="Saved VecNormalize stats path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output",
        default="artifacts/rl/search_policy/eval_metrics.json",
        help="Where to write the evaluation metrics JSON",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, default=default_config_path())
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    env_config = SearchTaskConfig.from_mapping(raw.get("env"))

    metrics = evaluate_artifact(
        config=env_config,
        model_path=resolve_repo_path(args.model_path),
        vecnormalize_path=resolve_repo_path(args.vecnormalize_path),
        episodes=max(1, int(args.episodes)),
    )

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
