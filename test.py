from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import load_dataset
from metrics import compute_multiclass_metrics, softmax
from models import MLPHead, MedSigLIPEncoder
from utils import (
    discover_fold_checkpoints,
    ensure_output_dirs,
    load_config,
    resolve_device,
    save_json,
)


def load_head(checkpoint_path: Path, device: torch.device) -> MLPHead:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = MLPHead(
        in_dim=checkpoint["in_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        out_dim=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_checkpoints(cfg, checkpoint_arg: str | None) -> list[Path]:
    """Ensemble members: explicit --checkpoint (file or directory) or the
    standard fold layout under checkpoint_dir/multiclass."""
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if path.is_dir():
            found = sorted(path.glob("fold_*/best.pt")) or sorted(path.glob("*/best.pt"))
            if not found:
                raise FileNotFoundError(f"No best.pt checkpoints under: {path}")
            return found
        return [path]
    return discover_fold_checkpoints(cfg["output"]["checkpoint_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Single best.pt or a directory of fold_*/best.pt heads")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    device = resolve_device(cfg["model"]["device"])
    external_name = (
        args.dataset
        or cfg["experiment"]["external_test_dataset"]
    )

    checkpoint_paths = resolve_checkpoints(cfg, args.checkpoint)
    missing = [p for p in checkpoint_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint(s) not found: {[str(m) for m in missing]}. Run train.py first."
        )

    external_df = load_dataset(
        cfg,
        external_name,
        # Filters to the gradable flag when the dataset declares one; no-op otherwise.
        require_gradable=True,
    )

    print(
        f"External dataset: {external_name} | "
        f"images={len(external_df)}"
    )
    print(
        "Grade distribution:",
        external_df["grade"].value_counts().sort_index().to_dict(),
    )

    encoder = MedSigLIPEncoder(cfg, device)

    emb_prefix = (
        Path(cfg["output"]["embedding_dir"])
        / "medsiglip"
        / external_name
    )

    emb_path, _, _ = encoder.extract_dataframe(
        external_df,
        output_prefix=emb_prefix,
        batch_size=cfg["model"]["embedding_batch_size"],
    )

    embeddings = np.load(emb_path, mmap_mode="r")

    heads = [load_head(path, device) for path in checkpoint_paths]
    print(f"Ensemble: {len(heads)} head(s) from {[str(p) for p in checkpoint_paths]}")

    x = torch.from_numpy(
        np.asarray(embeddings, dtype=np.float32)
    ).to(device)

    with torch.inference_mode():
        per_head_probs = torch.stack(
            [torch.softmax(head(x), dim=-1) for head in heads], dim=0
        ).cpu().numpy()  # (n_heads, N, C)

    mean_probs = per_head_probs.mean(axis=0)
    # metrics helpers operate on logits; log of averaged probabilities is the
    # canonical inverse (softmax(log p) == p)
    mean_log_probs = np.log(mean_probs + 1e-12)

    predictions = mean_probs.argmax(axis=1)
    targets = external_df["grade"].to_numpy(dtype=np.int64)

    metrics = compute_multiclass_metrics(
        targets,
        mean_log_probs,
        cfg["experiment"]["referable_threshold"],
    )

    pred_dir = (
        Path(cfg["output"]["prediction_dir"])
        / "external"
        / external_name
    )
    pred_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        pred_dir / "predictions.npz",
        image_ids=external_df["image_id"].to_numpy(dtype=object),
        targets=targets,
        probs=mean_probs,
        per_head_probs=per_head_probs,
        predictions=predictions,
    )

    prediction_df = external_df[
        ["image_id", "image_path", "grade", "binary_label"]
    ].copy()

    prediction_df["predicted_grade"] = predictions
    prediction_df["prediction_confidence"] = mean_probs.max(axis=1)
    prediction_df["predicted_referable"] = (
        predictions
        >= cfg["experiment"]["referable_threshold"]
    ).astype(int)
    head_predictions = per_head_probs.argmax(axis=2)  # (n_heads, N)
    prediction_df["head_agreement"] = (
        (head_predictions == predictions[None, :]).all(axis=0).astype(int)
    )

    prediction_df.to_csv(
        pred_dir / "predictions.csv",
        index=False,
    )

    save_json(
        {
            "dataset": external_name,
            "role": "external",
            "ensemble_size": len(heads),
            "checkpoints": [str(p) for p in checkpoint_paths],
            "metrics": metrics,
        },
        pred_dir / "metrics.json",
    )

    print("\nExternal evaluation (probability-averaged ensemble):")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "referable_sensitivity",
        "referable_specificity",
        "referable_auc",
    ]:
        print(f"  {key}: {metrics[key]:.4f}")

if __name__ == "__main__":
    main()
