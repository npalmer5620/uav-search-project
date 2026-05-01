#!/usr/bin/env python3
"""Generate presentation visuals for a V2 DQN search-policy run."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
from PIL import Image
from stable_baselines3 import DQN
from tensorboard.backend.event_processing import event_accumulator
import yaml

from uav_rl.actions import ACTION_NAMES
from uav_rl.search_task_env_v2 import SearchTaskConfigV2, SearchTaskEnvV2
from uav_rl.train_search_policy_v2 import merge_config


def ema(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out: list[float] = []
    avg: float | None = None
    for value in values:
        value = float(value)
        avg = value if avg is None else alpha * value + (1.0 - alpha) * avg
        out.append(avg)
    return np.asarray(out, dtype=float)


def scalar_series(ea: event_accumulator.EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    scalars = ea.Scalars(tag)
    return (
        np.asarray([scalar.step for scalar in scalars], dtype=float),
        np.asarray([scalar.value for scalar in scalars], dtype=float),
    )


def read_progress_rows(policy_dir: Path) -> list[dict]:
    progress_path = policy_dir / "training_progress.jsonl"
    rows: list[dict] = []
    if not progress_path.exists():
        return rows
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "timesteps" in row and row.get("event") in {"progress_eval", "stage_end"}:
                rows.append(row)
    return rows


def plot_training_curves(
    *,
    policy_dir: Path,
    tensorboard_run: Path,
    output_dir: Path,
) -> dict[str, float | str | None]:
    ea = event_accumulator.EventAccumulator(str(tensorboard_run), size_guidance={"scalars": 0})
    ea.Reload()
    reward_steps, reward_vals = scalar_series(ea, "rollout/ep_rew_mean")
    loss_steps, loss_vals = scalar_series(ea, "train/loss")
    eps_steps, eps_vals = scalar_series(ea, "rollout/exploration_rate")
    progress_rows = read_progress_rows(policy_dir)

    comparison_path = policy_dir / "smoke_final_stage_comparison.json"
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.exists()
        else {}
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        "DQN RL Search Training: Candidate-Action Policy",
        fontsize=16,
        fontweight="bold",
    )

    ax = axs[0, 0]
    ax.plot(reward_steps, reward_vals, color="#9dc6be", lw=1.0, alpha=0.55, label="raw")
    ax.plot(reward_steps, ema(reward_vals, 0.08), color="#087f6f", lw=2.5, label="EMA")
    if len(reward_vals) >= 20:
        early = float(np.mean(reward_vals[:10]))
        late = float(np.mean(reward_vals[-10:]))
        ax.annotate(
            f"early avg: {early:.0f}",
            xy=(reward_steps[8], reward_vals[8]),
            xytext=(0.03, 0.15),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#555"},
        )
        ax.annotate(
            f"late avg: {late:.0f}",
            xy=(reward_steps[-8], reward_vals[-8]),
            xytext=(0.58, 0.80),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#555"},
        )
    ax.set_title("Rollout reward improves during training")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("mean episode reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axs[0, 1]
    ax.plot(loss_steps, loss_vals, color="#f0b77f", lw=0.9, alpha=0.45, label="raw TD loss")
    ax.plot(loss_steps, ema(loss_vals, 0.10), color="#b75e00", lw=2.2, label="EMA")
    ax2 = ax.twinx()
    ax2.plot(eps_steps, eps_vals, color="#2f5d9e", lw=2.0, alpha=0.85, label="exploration eps")
    ax.set_title("DQN TD loss is noisy; exploration anneals")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("TD loss")
    ax2.set_ylabel("exploration epsilon")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

    ax = axs[1, 0]
    if progress_rows:
        steps = np.asarray([row["timesteps"] for row in progress_rows], dtype=float)
        confirmed = np.asarray(
            [row.get("mean_confirmed_targets", 0.0) for row in progress_rows],
            dtype=float,
        )
        success = np.asarray([row.get("success_rate", 0.0) for row in progress_rows], dtype=float)
        coverage = np.asarray([row.get("mean_coverage", 0.0) for row in progress_rows], dtype=float)
        ax.plot(steps, confirmed, marker="o", color="#7d3c98", lw=2.2, label="confirmed targets")
        ax.plot(steps, coverage, marker="s", color="#2e86c1", lw=2.0, label="coverage fraction")
        ax.plot(steps, success, marker="^", color="#239b56", lw=2.0, label="success rate")
        y_top = max(float(np.max(confirmed)), float(np.max(coverage)), float(np.max(success)), 1.0)
        ax.set_ylim(bottom=0.0, top=y_top * 1.18)
        for row in progress_rows:
            if row.get("event") == "stage_end":
                ax.axvline(row["timesteps"], color="#999", ls="--", lw=0.8, alpha=0.5)
                ax.text(
                    row["timesteps"],
                    y_top * 1.15,
                    f"S{row.get('stage_index')}",
                    ha="right",
                    va="top",
                    color="#555",
                )
    ax.set_title("Curriculum eval checkpoints")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("metric value")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")

    ax = axs[1, 1]
    if comparison:
        labels = ["DQN", "Random", "Lawnmower", "Greedy hybrid"]
        keys = ["dqn_policy", "random", "lawnmower", "greedy_hybrid"]
        rows = [(label, key) for label, key in zip(labels, keys) if key in comparison]
        rewards = [comparison[key]["mean_reward"] for _label, key in rows]
        confirmed = [comparison[key]["mean_confirmed_targets"] for _label, key in rows]
        colors = ["#087f6f", "#777777", "#d4942a", "#7d3c98"][: len(rows)]
        bars = ax.bar([label for label, _key in rows], rewards, color=colors, alpha=0.88)
        ax.axhline(0, color="#333", lw=0.8)
        for bar, confirmed_count in zip(bars, confirmed):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = 4 if y >= 0 else -4
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                f"{confirmed_count:.1f} confirmed",
                ha="center",
                va=va,
                fontsize=9,
            )
    ax.set_title("Final-stage smoke comparison")
    ax.set_ylabel("mean reward")
    ax.grid(True, axis="y", alpha=0.25)

    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "training_png": str(output_path),
        "reward_early_10_mean": float(np.mean(reward_vals[:10])) if len(reward_vals) >= 10 else None,
        "reward_late_10_mean": float(np.mean(reward_vals[-10:])) if len(reward_vals) >= 10 else None,
        "loss_early_10_mean": float(np.mean(loss_vals[:10])) if len(loss_vals) >= 10 else None,
        "loss_late_10_mean": float(np.mean(loss_vals[-10:])) if len(loss_vals) >= 10 else None,
    }


def stage_config_from_policy(policy_dir: Path) -> SearchTaskConfigV2:
    raw = yaml.safe_load((policy_dir / "search_policy_v2.yaml").read_text(encoding="utf-8"))
    base_config = SearchTaskConfigV2.from_mapping(raw["env"])
    curriculum = list(raw.get("train", {}).get("curriculum") or [])
    if not curriculum:
        return base_config
    return merge_config(base_config, curriculum[-1].get("env"))


def capture_rollout(
    *,
    policy_dir: Path,
    seed: int,
) -> tuple[SearchTaskEnvV2, list[dict]]:
    config = stage_config_from_policy(policy_dir)
    model = DQN.load(str(policy_dir / "model.zip"))
    env = SearchTaskEnvV2(config=config)
    obs, _info = env.reset(seed=seed)

    states: list[dict] = []
    path_xy: list[tuple[float, float]] = []
    cumulative_reward = 0.0

    def capture(*, reward: float = 0.0, action_name: str = "reset") -> None:
        path_xy.append((float(env.x), float(env.y)))
        states.append(
            {
                "grid": env.belief.grid.copy(),
                "x": float(env.x),
                "y": float(env.y),
                "yaw": float(env.yaw),
                "z": float(env.z),
                "coverage": float(env.belief.coverage_fraction),
                "confirmed": int(len(env.confirmed_truth_ids)),
                "required": int(env.config.required_targets(len(env.targets))),
                "step": int(env.step_index),
                "reward": float(reward),
                "cumulative_reward": float(cumulative_reward),
                "action": action_name,
                "targets": [
                    (float(target.x), float(target.y), target.class_name, idx in env.confirmed_truth_ids)
                    for idx, target in enumerate(env.targets)
                ],
                "path": list(path_xy),
            }
        )

    capture()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        cumulative_reward += float(reward)
        capture(
            reward=float(reward),
            action_name=info.get("last_action") or ACTION_NAMES.get(action_int, str(action_int)),
        )
        if len(states) > 80:
            break
    return env, states


def draw_rollout_state(env: SearchTaskEnvV2, state: dict, *, figsize: tuple[float, float]):
    x_min, x_max = env.geometry.x_limits
    y_min, y_max = env.geometry.y_limits
    cell_size = env.geometry.cell_size_m
    hfov_deg = math.degrees(env.camera.horizontal_fov_rad)
    _min_frustum, max_frustum = env.camera.ground_visibility_band(state["z"])

    fig, ax = plt.subplots(figsize=figsize)
    grid = state["grid"]
    observed = grid[:, :, 0].T
    obstacle = grid[:, :, 1].T
    victim = grid[:, :, 2].T
    confirmed_grid = grid[:, :, 5].T

    ax.imshow(
        observed,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.34,
        interpolation="nearest",
    )
    if np.max(victim) > 0:
        ax.imshow(
            np.ma.masked_where(victim <= 0.02, victim),
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.55,
            interpolation="nearest",
        )
    if np.max(confirmed_grid) > 0:
        ax.imshow(
            np.ma.masked_where(confirmed_grid <= 0.5, confirmed_grid),
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="Greens",
            vmin=0,
            vmax=1,
            alpha=0.65,
            interpolation="nearest",
        )
    if np.max(obstacle) > 0:
        ax.imshow(
            np.ma.masked_where(obstacle <= 0.5, obstacle),
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="gray_r",
            vmin=0,
            vmax=1,
            alpha=0.8,
            interpolation="nearest",
        )

    for tx, ty, _class_name, confirmed_target in state["targets"]:
        ax.scatter(
            tx,
            ty,
            marker="*",
            s=150,
            c="#2fb344" if confirmed_target else "#cc2f2f",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

    path = state["path"]
    if len(path) >= 2:
        ax.plot(
            [point[0] for point in path],
            [point[1] for point in path],
            color="#1764c8",
            lw=2.3,
            alpha=0.9,
            zorder=4,
        )
    ax.scatter(
        state["x"],
        state["y"],
        s=75,
        c="#0b1f4d",
        edgecolors="white",
        linewidths=1.2,
        zorder=6,
    )
    ax.arrow(
        state["x"],
        state["y"],
        math.cos(state["yaw"]) * cell_size * 0.7,
        math.sin(state["yaw"]) * cell_size * 0.7,
        head_width=0.7,
        head_length=0.9,
        color="#0b1f4d",
        zorder=7,
    )

    theta = math.degrees(state["yaw"])
    ax.add_patch(
        Wedge(
            (state["x"], state["y"]),
            max_frustum,
            theta - hfov_deg / 2.0,
            theta + hfov_deg / 2.0,
            color="#3f8efc",
            alpha=0.12,
            zorder=2,
        )
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, color="#d9d9d9", lw=0.6, alpha=0.7)
    ax.set_title(
        (
            f"Gymnasium V2 DQN rollout | step {state['step']} | "
            f"confirmed {state['confirmed']}/{state['required']} | "
            f"coverage {state['coverage'] * 100:.0f}%"
        ),
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.99,
        f"action: {state['action']}\nreward: {state['reward']:.1f} | cumulative: {state['cumulative_reward']:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.35",
        },
    )
    ax.text(
        0.99,
        0.01,
        "gray=observed, red=victim belief, green=confirmed, stars=ground truth for visualization",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    return fig


def write_rollout_visuals(
    *,
    policy_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, float | int | str]:
    env, states = capture_rollout(policy_dir=policy_dir, seed=seed)
    final_png = output_dir / f"gym_rollout_seed{seed}_final.png"
    fig = draw_rollout_state(env, states[-1], figsize=(8, 8))
    fig.savefig(final_png, dpi=180)
    plt.close(fig)

    frames: list[Image.Image] = []
    for state in states:
        fig = draw_rollout_state(env, state, figsize=(7, 7))
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=115)
        plt.close(fig)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))

    gif_path = output_dir / f"gym_rollout_seed{seed}.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=450,
            loop=0,
            optimize=True,
        )

    final_state = states[-1]
    return {
        "rollout_final_png": str(final_png),
        "rollout_gif": str(gif_path),
        "rollout_seed": seed,
        "rollout_steps": int(final_state["step"]),
        "rollout_confirmed": int(final_state["confirmed"]),
        "rollout_required": int(final_state["required"]),
        "rollout_coverage": float(final_state["coverage"]),
        "rollout_cumulative_reward": float(final_state["cumulative_reward"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir",
        default="artifacts/rl/search_policy_v2_candidate_actions_50k",
        help="Directory containing model.zip, search_policy_v2.yaml, and progress JSONL.",
    )
    parser.add_argument(
        "--tensorboard-run",
        default="artifacts/rl/tensorboard_v2/DQN_11",
        help="TensorBoard DQN run directory with train/loss and rollout scalars.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rl/visualizations/candidate_actions_50k",
        help="Directory for PNG/GIF outputs.",
    )
    parser.add_argument("--rollout-seed", type=int, default=3)
    args = parser.parse_args()

    policy_dir = Path(args.policy_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_policy_dir": str(policy_dir),
        "tensorboard_run": str(args.tensorboard_run),
    }
    summary.update(
        plot_training_curves(
            policy_dir=policy_dir,
            tensorboard_run=Path(args.tensorboard_run),
            output_dir=output_dir,
        )
    )
    summary.update(
        write_rollout_visuals(
            policy_dir=policy_dir,
            output_dir=output_dir,
            seed=args.rollout_seed,
        )
    )

    summary_path = output_dir / "visualization_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
