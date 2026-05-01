"""CPU-only YOLOv11n helper for visual diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_triage_rl.config import package_root, repo_root


@dataclass(frozen=True)
class YoloDetection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class YoloDetector:
    """Small Ultralytics wrapper that always runs inference on CPU."""

    def __init__(self, model_path: str | Path | None = None, *, confidence: float = 0.30, imgsz: int = 640) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for --yolo. Run: bash scripts/setup_macos_cpu.sh"
            ) from exc

        self.model_path = self.resolve_model_path(model_path)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.model = YOLO(str(self.model_path), task="detect")
        if str(self.model_path).endswith(".pt"):
            self.model.to("cpu")

    @staticmethod
    def resolve_model_path(model_path: str | Path | None = None) -> Path:
        if model_path is None or str(model_path).strip() == "":
            return repo_root() / "yolo11n.pt"
        candidate = Path(model_path).expanduser()
        if candidate.is_absolute():
            return candidate
        # Commands are expected to run from pyflyt_rl/, but support repo-root
        # invocation too by falling back to package-root-relative paths.
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (package_root() / candidate).resolve()

    def detect(self, image: np.ndarray) -> list[YoloDetection]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False,
        )
        detections: list[YoloDetection] = []
        if not results:
            return detections
        result = results[0]
        names: dict[int, str] = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            xyxy_raw = box.xyxy[0].tolist()
            detections.append(
                YoloDetection(
                    class_name=str(names.get(cls_id, cls_id)),
                    confidence=conf,
                    xyxy=tuple(float(v) for v in xyxy_raw),
                )
            )
        return detections


def annotate_detections(image: np.ndarray, detections: list[YoloDetection]) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return image

    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 220, 40), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 220, 40),
            1,
            cv2.LINE_AA,
        )
    return annotated


def detections_to_jsonable(detections: list[YoloDetection]) -> list[dict[str, Any]]:
    return [
        {
            "class_name": det.class_name,
            "confidence": det.confidence,
            "xyxy": list(det.xyxy),
        }
        for det in detections
    ]
