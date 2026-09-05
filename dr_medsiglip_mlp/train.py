from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import EmbeddingDataset, load_dataset
from metrics import compute_multiclass_metrics, softmax
from models import MLPHead, MedSigLIPEncoder
from utils import (
    ensure_output_dirs,
    load_config,
    resolve_device,
    save_json,
    set_seed,
)

def make_class_weights(
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)

def split_indices(df, cfg):
    indices = np.arange(len(df))
    labels = df["grade"].to_numpy()

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=cfg["experiment"]["test_size"],
        stratify=labels,
        random_state=cfg["experiment"]["split_seed"],
    )

    val_fraction_of_train_val = cfg["experiment"]["val_size"] / (
        1.0 - cfg["experiment"]["test_size"]
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        stratify=labels[train_val_idx],
        random_state=cfg["experiment"]["split_seed"],
    )

    return train_idx, val_idx, test_idx

def train_one_seed(
    cfg,
    embeddings,
    labels,
    train_idx,
    val_idx,
    test_idx,
    seed,
):
    set_seed(seed)
    device = resolve_device(cfg["model"]["device"])

    head_cfg = cfg["head"]
    num_classes = cfg["experiment"]["num_classes"]

    model = MLPHead(
        in_dim=embeddings.shape[1],
        hidden_dim=head_cfg["hidden_dim"],
        out_dim=num_classes,
        dropout=head_cfg["dropout"],
    ).to(device)

    train_loader = DataLoader(
        EmbeddingDataset(embeddings[train_idx], labels[train_idx]),
        batch_size=head_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(
            labels[train_idx], num_classes, device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=head_cfg["learning_rate"],
        weight_decay=head_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=head_cfg["max_epochs"],
    )

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    checkpoint_dir = (
        Path(cfg["output"]["checkpoint_dir"])
        / "multiclass"
        / f"seed_{seed}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"

    x_val = torch.from_numpy(
        np.asarray(embeddings[val_idx], dtype=np.float32)
    ).to(device)

    for epoch in range(1, head_cfg["max_epochs"] + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        scheduler.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val).cpu().numpy()

        val_f1 = float(
            f1_score(
                labels[val_idx],
                val_logits.argmax(axis=1),
                average="macro",
                zero_division=0,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_macro_f1": val_f1,
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        )

        print(
            f"seed={seed} epoch={epoch:02d} "
            f"loss={np.mean(epoch_losses):.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "in_dim": embeddings.shape[1],
                    "hidden_dim": head_cfg["hidden_dim"],
                    "dropout": head_cfg["dropout"],
                    "num_classes": num_classes,
                    "best_val_macro_f1": best_f1,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= head_cfg["patience"]:
                break

    save_json(history, checkpoint_dir / "history.json")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict(indices):
        x = torch.from_numpy(
            np.asarray(embeddings[indices], dtype=np.float32)
        ).to(device)

        with torch.inference_mode():
            return model(x).cpu().numpy()

    val_logits = predict(val_idx)
    test_logits = predict(test_idx)

    val_metrics = compute_multiclass_metrics(
        labels[val_idx],
        val_logits,
        cfg["experiment"]["referable_threshold"],
    )
    test_metrics = compute_multiclass_metrics(
        labels[test_idx],
        test_logits,
        cfg["experiment"]["referable_threshold"],
    )

    pred_dir = (
        Path(cfg["output"]["prediction_dir"])
        / "development"
        / f"seed_{seed}"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        pred_dir / "predictions.npz",
        val_indices=val_idx,
        val_targets=labels[val_idx],
        val_logits=val_logits,
        val_probs=softmax(val_logits),
        test_indices=test_idx,
        test_targets=labels[test_idx],
        test_logits=test_logits,
        test_probs=softmax(test_logits),
    )

    save_json(
        {"seed": seed, "val": val_metrics, "test": test_metrics},
        pred_dir / "metrics.json",
    )

    return checkpoint_path, test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    device = resolve_device(cfg["model"]["device"])
    print(f"Device: {device}")

    train_dataset_name = cfg["experiment"]["train_dataset"]
    df = load_dataset(cfg, train_dataset_name)

    print(f"{train_dataset_name}: {len(df)} images")
    print(
        "Grade distribution:",
        df["grade"].value_counts().sort_index().to_dict(),
    )

    train_idx, val_idx, test_idx = split_indices(df, cfg)

    print(
        f"Split sizes: train={len(train_idx)}, "
        f"val={len(val_idx)}, test={len(test_idx)}"
    )

    encoder = MedSigLIPEncoder(cfg, device)

    emb_prefix = (
        Path(cfg["output"]["embedding_dir"])
        / "medsiglip"
        / train_dataset_name
    )

    emb_path, _, _ = encoder.extract_dataframe(
        df,
        output_prefix=emb_prefix,
        batch_size=cfg["model"]["embedding_batch_size"],
    )

    embeddings = np.load(emb_path, mmap_mode="r")
    labels = df["grade"].to_numpy(dtype=np.int64)

    results = []

    for seed in cfg["experiment"]["seeds"]:
        print(f"\n{'=' * 70}\nTraining seed {seed}\n{'=' * 70}")

        checkpoint_path, test_metrics = train_one_seed(
            cfg,
            embeddings,
            labels,
            train_idx,
            val_idx,
            test_idx,
            seed,
        )

        results.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "test_macro_f1": test_metrics["macro_f1"],
                "test_referable_sensitivity": test_metrics[
                    "referable_sensitivity"
                ],
                "test_referable_specificity": test_metrics[
                    "referable_specificity"
                ],
                "test_referable_auc": test_metrics["referable_auc"],
            }
        )

    save_json(
        {
            "train_dataset": train_dataset_name,
            "num_images": len(df),
            "split_sizes": {
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            },
            "runs": results,
        },
        Path(cfg["output"]["metric_dir"]) / "training_summary.json",
    )

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
