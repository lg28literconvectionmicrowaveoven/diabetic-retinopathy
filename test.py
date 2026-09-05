from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import load_dataset
from metrics import compute_multiclass_metrics, softmax
from models import MLPHead, MedSigLIPEncoder
from utils import ensure_output_dirs, load_config, resolve_device, save_json


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    device = resolve_device(cfg["model"]["device"])
    external_name = (
        args.dataset
        or cfg["experiment"]["external_test_dataset"]
    )

    checkpoint_path = Path(
        args.checkpoint
        or (
            Path(cfg["output"]["checkpoint_dir"])
            / "multiclass"
            / f"seed_{cfg['experiment']['seeds'][0]}"
            / "best.pt"
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py first."
        )

    external_df = load_dataset(
        cfg,
        external_name,
        require_gradable=(external_name == "messidor2"),
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
    head = load_head(checkpoint_path, device)

    x = torch.from_numpy(
        np.asarray(embeddings, dtype=np.float32)
    ).to(device)

    with torch.inference_mode():
        logits = head(x).cpu().numpy()

    probs = softmax(logits)
    predictions = probs.argmax(axis=1)
    targets = external_df["grade"].to_numpy(dtype=np.int64)

    metrics = compute_multiclass_metrics(
        targets,
        logits,
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
        logits=logits,
        probs=probs,
        predictions=predictions,
    )

    prediction_df = external_df[
        ["image_id", "image_path", "grade", "binary_label"]
    ].copy()

    prediction_df["predicted_grade"] = predictions
    prediction_df["prediction_confidence"] = probs.max(axis=1)
    prediction_df["predicted_referable"] = (
        predictions
        >= cfg["experiment"]["referable_threshold"]
    ).astype(int)

    prediction_df.to_csv(
        pred_dir / "predictions.csv",
        index=False,
    )

    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "dataset": external_name,
            "metrics": metrics,
        },
        pred_dir / "metrics.json",
    )

    print("\nExternal evaluation:")
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
