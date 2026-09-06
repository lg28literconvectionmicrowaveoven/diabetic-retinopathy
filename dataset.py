from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from preproc.denoise import preprocess_fundus_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}


def _resolve_image(index: dict[str, Path], image_id: str) -> Path | None:
    key = str(image_id).strip()

    if key in index:
        return index[key]

    stem = Path(key).stem
    if stem in index:
        return index[stem]

    return None


def build_image_index(image_root: Path) -> dict[str, Path]:
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS:
            index[path.stem] = path
            index[path.name] = path
    return index


def preprocess_image(image: Image.Image, cfg: dict[str, Any] | None = None) -> Image.Image:
    prep_cfg = cfg.get("preprocessing", {}) if cfg else {}
    if not prep_cfg.get("enabled", True):
        return image.convert("RGB")
    return preprocess_fundus_image(image, prep_cfg)


def load_dataset(
    cfg: dict[str, Any],
    dataset_name: str,
    *,
    require_gradable: bool = False,
) -> pd.DataFrame:
    data_cfg = cfg["data"]

    if dataset_name not in data_cfg:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")

    ds = data_cfg[dataset_name]
    root = Path(data_cfg["data_root"]).expanduser().resolve()
    ds_root = root / ds["root"]
    image_root = ds_root / ds["images"]
    csv_path = ds_root / ds["csv"]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    id_col = ds["image_id_column"]
    label_col = ds["label_column"]

    missing = [c for c in [id_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} CSV missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["image_id"] = df[id_col].astype(str)
    df["grade"] = pd.to_numeric(df[label_col], errors="raise").astype(int)

    if not df["grade"].between(0, 4).all():
        raise ValueError(f"{dataset_name}: DR grades must be in [0, 4].")

    df["binary_label"] = (
        df["grade"] >= cfg["experiment"]["referable_threshold"]
    ).astype(int)

    gradable_col = ds.get("gradable_column")
    if require_gradable and gradable_col and gradable_col in df.columns:
        values = df[gradable_col]
        if values.dtype == bool:
            mask = values
        else:
            mask = values.astype(str).str.strip().str.lower().isin(
                {"true", "1", "yes", "gradable"}
            )
        df = df[mask].copy()

    image_index = build_image_index(image_root)
    df["image_path"] = df["image_id"].map(
        lambda x: _resolve_image(image_index, x)
    )

    missing_count = int(df["image_path"].isna().sum())
    if missing_count:
        missing_ids = (
            df.loc[df["image_path"].isna(), "image_id"].head(10).tolist()
        )
        raise FileNotFoundError(
            f"{dataset_name}: {missing_count} images could not be resolved. "
            f"Examples: {missing_ids}"
        )

    df["image_path"] = df["image_path"].astype(str)
    df["dataset"] = dataset_name
    return df.reset_index(drop=True)


class FundusDataset(Dataset):
    """Image-level dataset used by the frozen encoder."""

    def __init__(self, dataframe: pd.DataFrame, cfg: dict[str, Any]):
        self.df = dataframe.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image = preprocess_image(image, self.cfg)

        return {
            "image": image,
            "image_id": row["image_id"],
            "grade": int(row["grade"]),
            "binary_label": int(row["binary_label"]),
            "image_path": row["image_path"],
        }


class EmbeddingDataset(Dataset):
    """Dataset used to train the MLP head on cached embeddings."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.x = np.asarray(embeddings, dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
