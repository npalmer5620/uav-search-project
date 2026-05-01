"""Configuration helpers for the standalone PyFlyt RL stack."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return package_root() / "configs" / "default.yaml"


def resolve_path(path: str | Path | None, *, base: Path | None = None) -> Path:
    if path is None or str(path).strip() == "":
        raise ValueError("A path is required")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base or package_root()) / candidate


def deep_update(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_path(path, base=package_root()) if path else default_config_path()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Run: bash scripts/setup_macos_cpu.sh"
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return config_path, data


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write YAML configs") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)

