#!/usr/bin/env python3
"""Run visual triage-search rollouts with optional CPU YOLO diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from uav_triage_rl.config import load_config, package_root, resolve_path
from uav_triage_rl.env import TriageSearchEnv
from uav_triage_rl.yolo import YoloDetector, annotate_detections, detections_to_jsonable


def _write_frame(path: Path, frame: np.ndarray) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to write sim frames") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _annotate_synthetic_bboxes(frame: np.ndarray, bboxes: list[dict[str, Any]]) -> np.ndarray:
    if not bboxes:
        return frame
    try:
        import cv2
    except ImportError:
        return frame

    annotated = frame.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox["xyxy"])
        confidence = float(bbox.get("confidence", 0.0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (240, 190, 20), 2)
        label = f"synth {bbox.get('class_name', 'person')} {confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 190, 20),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_sim(
    config_path: str | Path | None,
    *,
    model_path: str | Path | None,
    episodes: int,
    max_steps: int | None,
    render: bool,
    gui: bool,
    yolo_enabled: bool,
    policy_name: str,
    output_dir: str | Path | None,
    perception_mode: str | None,
) -> dict[str, Any]:
    _, config = load_config(config_path)
    if perception_mode is not None:
        config["perception"] = {**dict(config.get("perception", {}) or {}), "mode": perception_mode}
    sim_config = dict(config.get("sim", {}) or {})
    if render:
        config["env"] = {**dict(config.get("env", {})), "backend": "pyflyt"}
    else:
        # Non-rendered sims can use the fast backend.
        config["env"] = {**dict(config.get("env", {})), "backend": "kinematic"}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = resolve_path(output_dir or sim_config.get("output_dir", "artifacts/sim"), base=package_root()) / run_id
    root.mkdir(parents=True, exist_ok=True)

    model = None
    if model_path is not None:
        from stable_baselines3 import DQN

        model = DQN.load(str(resolve_path(model_path, base=package_root())), device="cpu")
    detector = None
    if yolo_enabled:
        detector = YoloDetector(
            sim_config.get("yolo_model_path", "../yolo11n.pt"),
            confidence=float(sim_config.get("yolo_confidence", 0.30)),
        )
    yolo_interval = max(1, int(sim_config.get("yolo_interval_steps", 3)))

    events: list[dict[str, Any]] = []
    synthetic_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for episode in range(max(1, int(episodes))):
        render_mode = "human" if gui else "rgb_array"
        env = TriageSearchEnv(config=config, render_mode=render_mode)
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        step = 0
        video_writer = None
        video_path = root / f"episode_{episode:02d}.mp4"
        while not done:
            if model is not None:
                action = int(model.predict(obs, deterministic=True)[0])
            elif policy_name == "lawnmower":
                action = env.lawnmower_action()
            else:
                action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            synthetic_bboxes = list(info.get("events", {}).get("bboxes", []) or [])
            frame = env.render()
            if frame is not None:
                detections_json: list[dict[str, Any]] = []
                annotated = _annotate_synthetic_bboxes(frame, synthetic_bboxes)
                if detector is not None and step % yolo_interval == 0:
                    detections = detector.detect(frame)
                    detections_json = detections_to_jsonable(detections)
                    annotated = annotate_detections(annotated, detections)
                frame_path = root / f"episode_{episode:02d}" / f"frame_{step:04d}.jpg"
                _write_frame(frame_path, annotated)
                if synthetic_bboxes:
                    synthetic_events.append(
                        {
                            "episode": episode,
                            "step": step,
                            "frame": str(frame_path),
                            "bboxes": synthetic_bboxes,
                        }
                    )
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("opencv-python is required to write sim video") from exc
                if video_writer is None:
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    pyflyt_config = dict(config.get("env", {}).get("pyflyt", {}) or {})
                    fps = int(pyflyt_config.get("camera_fps", 15) or 15)
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        fps,
                        (width, height),
                    )
                video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                if detections_json:
                    events.append(
                        {
                            "episode": episode,
                            "step": step,
                            "frame": str(frame_path),
                            "detections": detections_json,
                        }
                    )
            step += 1
            done = bool(terminated or truncated or (max_steps is not None and step >= max_steps))
        if video_writer is not None:
            video_writer.release()
        summaries.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": step,
                "success": bool(info.get("success")),
                "coverage_fraction": float(info.get("coverage_fraction", 0.0)),
                "confirmed_victims": int(info.get("confirmed_victims", 0)),
                "video": str(video_path) if video_path.exists() else None,
            }
        )
        env.close()

    summary = {
        "output_dir": str(root),
        "episodes": summaries,
        "synthetic_bbox_events": synthetic_events,
        "yolo_events": events,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "synthetic_bboxes.json").write_text(json.dumps(synthetic_events, indent=2), encoding="utf-8")
    (root / "yolo_detections.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Use PyFlyt rendered camera frames")
    parser.add_argument("--gui", action="store_true", help="Open the PyBullet GUI while rendering")
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--policy", choices=["random", "lawnmower"], default="random")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--perception", choices=["bbox", "point"], default=None)
    args = parser.parse_args()
    result = run_sim(
        args.config,
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        gui=args.gui,
        yolo_enabled=args.yolo,
        policy_name=args.policy,
        output_dir=args.output_dir,
        perception_mode=args.perception,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
