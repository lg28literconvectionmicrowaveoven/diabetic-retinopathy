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
    if config_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
