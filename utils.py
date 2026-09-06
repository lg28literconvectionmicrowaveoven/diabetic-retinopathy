from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(config_device: str = "auto") -> torch.device:
    if config_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return torch.device("cuda")
    if config_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but no MPS device is available.")
        return torch.device("mps")
    if config_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    output_cfg = cfg["output"]
    for key in [
        "root",
        "checkpoint_dir",
        "embedding_dir",
        "prediction_dir",
        "metric_dir",
        "log_dir",
    ]:
        Path(output_cfg[key]).mkdir(parents=True, exist_ok=True)


def discover_fold_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """Return the ensemble's head checkpoints, newest scheme first.

    Prefers the grouped five-fold layout ``multiclass/fold_<i>/best.pt``;
    falls back to the legacy repeated-seed layout ``multiclass/seed_<i>/best.pt``
    (only the first seed — that scheme was a single split, not an ensemble).
    """
    base = Path(checkpoint_dir) / "multiclass"

    folds = sorted(base.glob("fold_*/best.pt"), key=lambda p: int(p.parent.name.split("_")[1]))
    if folds:
        return folds

    legacy = sorted(base.glob("seed_*/best.pt"))
    return legacy[:1]
