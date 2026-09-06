from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
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

def make_folds(df, cfg):
    """Grouped 5-fold CV plan: 5 outer folds of ~20% test each, with an
    inner ~15% validation split carved from the remaining 80% development
    portion (effective sizes ~68% train / 12% val / 20% internal test).

    Both eyes of one patient share a fold whenever ``patient_id`` is present
    (see dataset.derive_patient_ids): outer folds use StratifiedGroupKFold and
    the inner split is grouped as well, so checkpoint selection never sees the
    fellow eye of a training image either.
    """
    experiment_cfg = cfg["experiment"]
    n_folds = int(experiment_cfg["n_folds"])
    fold_seed = int(experiment_cfg["fold_seed"])
    dev_val_size = float(experiment_cfg["dev_val_size"])

    labels = df["grade"].to_numpy()
    indices = np.arange(len(df))

    grouped = "patient_id" in df.columns and df["patient_id"].notna().all()
    groups = df["patient_id"].to_numpy() if grouped else None

    if grouped:
        outer = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=fold_seed
        )
        outer_splits = list(outer.split(indices, labels, groups))
    else:
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
        outer_splits = list(outer.split(indices, labels))

    folds = []
    for fold_index, (dev_idx, test_idx) in enumerate(outer_splits):
        if grouped:
            inner = GroupShuffleSplit(
                n_splits=1, test_size=dev_val_size, random_state=fold_seed + fold_index
            )
            inner_train, inner_val = next(
                inner.split(dev_idx, labels[dev_idx], groups[dev_idx])
            )
            train_idx = dev_idx[inner_train]
            val_idx = dev_idx[inner_val]
        else:
            train_idx, val_idx = train_test_split(
                dev_idx,
                test_size=dev_val_size,
                stratify=labels[dev_idx],
                random_state=fold_seed + fold_index,
            )

        folds.append(
            {
                "fold": fold_index,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
        )

    return folds


def train_one_fold(
    cfg,
    embeddings,
    labels,
    fold,
    seed,
):
    """Train one MLP head on one CV fold's development split.

    Checkpoint selection uses validation macro-F1 with early stopping;
    predictions on the untouched 20% internal test split are produced once
    with the selected checkpoint.
    """
    fold_index = fold["fold"]
    train_idx = fold["train_idx"]
    val_idx = fold["val_idx"]
    test_idx = fold["test_idx"]

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
        / f"fold_{fold_index}"
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
            f"fold={fold_index} epoch={epoch:02d} "
            f"loss={np.mean(epoch_losses):.4f} "
            f"val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": fold_index,
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
        / f"fold_{fold_index}"
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
        {"fold": fold_index, "seed": seed, "val": val_metrics, "test": test_metrics},
        pred_dir / "metrics.json",
    )

    return {
        "checkpoint": checkpoint_path,
        "test_metrics": test_metrics,
        "test_indices": test_idx,
        "test_logits": test_logits,
    }


def aggregate_oof(cfg, labels, fold_results):
    """Pool every fold's held-out predictions into one internal result.

    Each MESSIDOR-2 image appears in exactly one fold's test split, so the
    pooled predictions cover the whole eligible dataset out-of-fold.
    """
    indices = np.concatenate([r["test_indices"] for r in fold_results])
    logits = np.concatenate([r["test_logits"] for r in fold_results])
    order = np.argsort(indices)
    indices, logits = indices[order], logits[order]

    metrics = compute_multiclass_metrics(
        labels[indices],
        logits,
        cfg["experiment"]["referable_threshold"],
    )

    pred_dir = Path(cfg["output"]["prediction_dir"]) / "development" / "oof"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / "predictions.npz",
        indices=indices,
        targets=labels[indices],
        logits=logits,
        probs=softmax(logits),
    )
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    device = resolve_device(cfg["model"]["device"])
    print(f"Device: {device}")

    train_dataset_name = cfg["experiment"]["train_dataset"]
    # require_gradable filters to adjudicated_gradable == 1 when the dataset
    # declares a gradable_column (MESSIDOR-2); no-op for datasets without one.
    df = load_dataset(cfg, train_dataset_name, require_gradable=True)

    print(f"{train_dataset_name}: {len(df)} images")
    print(
        "Grade distribution:",
        df["grade"].value_counts().sort_index().to_dict(),
    )

    folds = make_folds(df, cfg)
    grouped = "patient_id" in df.columns and df["patient_id"].notna().all()

    print(
        f"Grouped five-fold CV: {'patient-grouped' if grouped else 'ungrouped (leakage risk)'}"
    )
    for fold in folds:
        n_test_patients = (
            df.iloc[fold["test_idx"]]["patient_id"].nunique() if grouped else "-"
        )
        print(
            f"fold {fold['fold']}: train={len(fold['train_idx'])} "
            f"val={len(fold['val_idx'])} test={len(fold['test_idx'])} "
            f"test_patients={n_test_patients}"
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

    fold_seed = int(cfg["experiment"]["fold_seed"])
    fold_results = []

    for fold in folds:
        seed = fold_seed + fold["fold"]
        print(
            f"\n{'=' * 70}\nTraining fold {fold['fold']} (seed {seed})\n{'=' * 70}"
        )

        result = train_one_fold(cfg, embeddings, labels, fold, seed)
        fold_results.append(result)

    oof_metrics = aggregate_oof(cfg, labels, fold_results)

    save_json(
        {
            "train_dataset": train_dataset_name,
            "num_images": len(df),
            "num_patients": int(df["patient_id"].nunique()) if grouped else None,
            "grouped": bool(grouped),
            "n_folds": len(folds),
            "folds": [
                {
                    "fold": fold["fold"],
                    "seed": fold_seed + fold["fold"],
                    "checkpoint": str(result["checkpoint"]),
                    "test_macro_f1": result["test_metrics"]["macro_f1"],
                    "test_referable_sensitivity": result["test_metrics"][
                        "referable_sensitivity"
                    ],
                    "test_referable_specificity": result["test_metrics"][
                        "referable_specificity"
                    ],
                    "test_referable_auc": result["test_metrics"]["referable_auc"],
                }
                for fold, result in zip(folds, fold_results)
            ],
            "oof_internal": oof_metrics,
        },
        Path(cfg["output"]["metric_dir"]) / "training_summary.json",
    )

    print("\nOut-of-fold internal metrics (all folds pooled):")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "referable_sensitivity",
        "referable_specificity",
        "referable_auc",
    ]:
        print(f"  {key}: {oof_metrics[key]:.4f}")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
