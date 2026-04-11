from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "compact_mtcnn_mobilefacenet.yaml"


def _resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    project_root = config_path.parent.parent
    preprocess_path_keys = {"aligned_dir", "aligned_manifest"}
    training_path_keys = {
        "output_dir",
        "best_checkpoint_path",
        "last_checkpoint_path",
        "training_state_path",
        "training_metrics_path",
    }

    def resolve(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((project_root / path).resolve())

    config["data"]["root_dir"] = resolve(config["data"]["root_dir"])
    config["recognizer"]["checkpoint_path"] = resolve(config["recognizer"].get("checkpoint_path"))
    preprocess = config.get("preprocess", {})
    for key, value in preprocess.items():
        if key in preprocess_path_keys and isinstance(value, str):
            preprocess[key] = resolve(value)
    training = config.get("training", {})
    for key, value in training.items():
        if key in training_path_keys and isinstance(value, str):
            training[key] = resolve(value)
    outputs = config.get("outputs", {})
    for key, value in outputs.items():
        outputs[key] = resolve(value)
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_config_path()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _resolve_config_paths(config, config_path)
