from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from PIL import Image

sys.modules.setdefault("torchaudio", None)

from dataset import preprocess_image, build_image_index
from explainability import MedSigLIPExplainableModel, heatmap_to_image, overlay_heatmap
from metrics import compute_multiclass_metrics, softmax
from models import MedSigLIPEncoder, MLPHead
from utils import ensure_output_dirs, load_config, resolve_device, save_json, set_seed


def prepare_messidor2_dataframe(messidor_dir: Path) -> pd.DataFrame:
    csv_path = messidor_dir / "messidor_data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV at {csv_path}")

    df_raw = pd.read_csv(csv_path)
    id_col = [c for c in df_raw.columns if "id" in c.lower() or "image" in c.lower() or "code" in c.lower()][0]
    grade_col = [c for c in df_raw.columns if "diag" in c.lower() or "grade" in c.lower() or "retinopathy" in c.lower()][0]
    gradable_col = [c for c in df_raw.columns if "gradable" in c.lower()]

    df_clean = df_raw.copy()
    if gradable_col:
        df_clean = df_clean[df_clean[gradable_col[0]] == 1].copy()

    # Index images in messidor-2 subdirectories
    img_index = build_image_index(messidor_dir / "messidor-2")
    records = []
    for _, row in df_clean.iterrows():
        img_id = str(row[id_col]).strip()
        img_path = img_index.get(img_id) or img_index.get(Path(img_id).stem)
        if img_path is not None and img_path.exists():
            records.append({
                "image_id": img_id,
                "image_path": str(img_path),
                "grade": int(row[grade_col]),
                "binary_label": int(int(row[grade_col]) >= 2),
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} gradable Messidor-2 images across 5 grades.")
    print("Grade distribution:")
    print(df["grade"].value_counts().sort_index())
    return df


def extract_embeddings_with_pipeline(
    df: pd.DataFrame,
    encoder: MedSigLIPEncoder,
    cfg: dict,
    batch_size: int = 16,
) -> np.ndarray:
    """Pass images through our preprocessing pipeline and extract MedSigLIP embeddings."""
    total = len(df)
    embeddings = []
    print(f"Processing {total} images through preprocessing pipeline + MedSigLIP...")

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_rows = df.iloc[start_idx:end_idx]
        batch_images = []

        for _, row in batch_rows.iterrows():
            with Image.open(row["image_path"]) as img:
                img_rgb = img.convert("RGB")
                prepped = preprocess_image(img_rgb, cfg)
                batch_images.append(prepped)

        batch_emb = encoder.encode(batch_images)
        embeddings.append(batch_emb)

        if (start_idx // batch_size) % 10 == 0 or end_idx == total:
            print(f" -> Preprocessed & embedded [{end_idx}/{total}] images...")

    return np.concatenate(embeddings, axis=0)


def train_mlp_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    in_dim: int = 1152,
    hidden_dim: int = 512,
    num_classes: int = 5,
    dropout: float = 0.10,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 40,
    patience: int = 8,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, dict]:
    # Compute inverse class weights
    counts = np.bincount(train_y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = MLPHead(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y)),
        batch_size=32,
        shuffle=True,
    )

    best_val_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_x)

        # Validation
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(val_x.astype(np.float32)).to(device)
            val_logits = model(vx).cpu().numpy()

        val_metrics = compute_multiclass_metrics(val_y, val_logits, referable_threshold=2)
        val_f1 = val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_f1": best_val_f1, "epochs_trained": epoch, "history": history}


def main():
    cfg = load_config("config.yaml")
    ensure_output_dirs(cfg)
    set_seed(13)
    device = resolve_device(cfg["model"].get("device", "auto"))
    print(f"Active Device: {device}")

    # 1. Load Messidor-2 for Training
    print("\n--- [1/4] Preparing Messidor-2 Dataset for Training ---")
    messidor_dir = Path("data/messidor2")
    m_df = prepare_messidor2_dataframe(messidor_dir)

    # 2. Extract Embeddings through Preprocessing Pipeline + MedSigLIP
    print("\n--- [2/4] Initializing MedSigLIP Foundation Encoder ---")
    encoder = MedSigLIPEncoder(cfg, device)

    emb_cache_path = Path("outputs/embeddings/medsiglip/messidor2.npy")
    emb_cache_path.parent.mkdir(parents=True, exist_ok=True)

    if emb_cache_path.is_file():
        print(f"Loading cached embeddings from {emb_cache_path}...")
        m_embeddings = np.load(emb_cache_path)
    else:
        m_embeddings = extract_embeddings_with_pipeline(m_df, encoder, cfg, batch_size=16)
        np.save(emb_cache_path, m_embeddings)
        print(f"Saved Messidor-2 embeddings ({m_embeddings.shape}) to {emb_cache_path}")

    # 3. Train MLP Head on Messidor-2
    print("\n--- [3/4] Training MLP Head on Messidor-2 ---")
    m_labels = m_df["grade"].to_numpy(dtype=np.int64)
    train_idx, val_idx = train_test_split(
        np.arange(len(m_df)),
        test_size=0.20,
        stratify=m_labels,
        random_state=13,
    )

    trained_head, train_info = train_mlp_head(
        train_x=m_embeddings[train_idx],
        train_y=m_labels[train_idx],
        val_x=m_embeddings[val_idx],
        val_y=m_labels[val_idx],
        in_dim=encoder.embedding_dim,
        hidden_dim=cfg["head"].get("hidden_dim", 512),
        num_classes=cfg["experiment"].get("num_classes", 5),
        dropout=cfg["head"].get("dropout", 0.10),
        device=device,
    )
    print(f"Training finished in {train_info['epochs_trained']} epochs. Best Val Macro-F1: {train_info['best_val_f1']:.4f}")

    # Save Checkpoint
    ckpt_dir = Path("outputs/checkpoints/multiclass/seed_13")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    torch.save({
        "model_state_dict": trained_head.state_dict(),
        "in_dim": encoder.embedding_dim,
        "hidden_dim": cfg["head"].get("hidden_dim", 512),
        "num_classes": 5,
        "dropout": cfg["head"].get("dropout", 0.10),
        "best_val_macro_f1": train_info["best_val_f1"],
    }, ckpt_path)
    print(f"Saved trained MLP checkpoint to: {ckpt_path}")

    # 4. Independent Validation on IDRiD (Indian Dataset)
    print("\n--- [4/4] Validating on Indian Dataset (IDRiD) ---")
    idrid_labels_path = Path("data/idrid/labels.csv")
    if not idrid_labels_path.is_file():
        from test_indian_dataset import setup_idrid_dataset
        id_df, _ = setup_idrid_dataset(Path("data/idrid"))
    else:
        id_df = pd.read_csv(idrid_labels_path)

    idrid_cache_path = Path("outputs/embeddings/medsiglip/idrid.npy")
    if idrid_cache_path.is_file():
        print(f"Loading cached IDRiD embeddings from {idrid_cache_path}...")
        id_embeddings = np.load(idrid_cache_path)
    else:
        id_embeddings = extract_embeddings_with_pipeline(id_df, encoder, cfg, batch_size=16)
        np.save(idrid_cache_path, id_embeddings)
        print(f"Saved IDRiD embeddings ({id_embeddings.shape}) to {idrid_cache_path}")

    trained_head.eval()
    with torch.no_grad():
        id_x = torch.from_numpy(id_embeddings.astype(np.float32)).to(device)
        id_logits = trained_head(id_x).cpu().numpy()

    id_labels = id_df["grade"].to_numpy(dtype=np.int64)
    id_metrics = compute_multiclass_metrics(id_labels, id_logits, referable_threshold=2)

    print("\n" + "=" * 60)
    print("       IDRiD (Indian Dataset) Validation Results       ")
    print("=" * 60)
    print(f" - Macro F1 Score:             {id_metrics['macro_f1']:.4f}")
    print(f" - Quadratic Weighted Kappa:    {id_metrics['qwk']:.4f}")
    print(f" - Referable DR AUC:           {id_metrics['referable_auc']:.4f}")
    print(f" - Referable Sensitivity:      {id_metrics['referable_sensitivity']:.4f}")
    print(f" - Referable Specificity:      {id_metrics['referable_specificity']:.4f}")
    print("=" * 60)

    # Save summary
    val_summary = {
        "training_dataset": "Messidor-2",
        "validation_dataset": "IDRiD (Indian Dataset)",
        "num_training_images": len(m_df),
        "num_validation_images": len(id_df),
        "metrics": id_metrics,
    }
    save_json(val_summary, Path("outputs/metrics/messidor2_to_idrid_results.json"))
    print("Summary saved to outputs/metrics/messidor2_to_idrid_results.json")


if __name__ == "__main__":
    main()
