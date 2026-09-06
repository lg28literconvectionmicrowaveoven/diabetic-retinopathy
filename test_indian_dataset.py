from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image

sys.modules.setdefault("torchaudio", None)

from dataset import preprocess_image
from explainability import MedSigLIPExplainableModel, overlay_heatmap
from models import MedSigLIPEncoder, MLPHead
from utils import load_config, resolve_device


def setup_idrid_dataset(data_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Unpack and parse the IDRiD Disease Grading dataset."""
    zip_path = data_dir / "disease_grading.zip"
    extract_dir = data_dir / "extracted"

    if not extract_dir.exists():
        print(f"[1/5] Extracting {zip_path.name} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")

    # Locate groundtruth CSVs
    csv_files = list(extract_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in dataset:")
    for c in csv_files:
        print(f" - {c.relative_to(extract_dir)}")

    # Look for testing or training labels
    test_csvs = [c for c in csv_files if "testing" in c.name.lower() or "test" in c.name.lower()]
    train_csvs = [c for c in csv_files if "training" in c.name.lower() or "train" in c.name.lower()]
    target_csv = test_csvs[0] if test_csvs else (train_csvs[0] if train_csvs else csv_files[0])
    print(f"Using groundtruth: {target_csv.name}")

    df_raw = pd.read_csv(target_csv)
    print(f"Raw CSV columns: {list(df_raw.columns)}")

    # Standardize column names
    id_col = [c for c in df_raw.columns if "image" in c.lower() or "id" in c.lower()][0]
    grade_col = [c for c in df_raw.columns if "retinopathy" in c.lower() or "grade" in c.lower()][0]

    # Index image files
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_map = {}
    for p in extract_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_exts:
            image_map[p.stem] = p
            image_map[p.name] = p

    records = []
    for _, row in df_raw.iterrows():
        img_id = str(row[id_col]).strip()
        if pd.isna(row[grade_col]):
            continue
        grade = int(row[grade_col])
        img_path = image_map.get(img_id) or image_map.get(Path(img_id).stem)
        if img_path is not None and img_path.exists():
            records.append({
                "image_id": img_id,
                "image_path": str(img_path),
                "grade": grade,
                "binary_label": int(grade >= 2),
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} images with groundtruth DR grades across {df['grade'].nunique()} grades.")
    print("Grade distribution:")
    print(df["grade"].value_counts().sort_index())

    # Save standardized labels.csv
    labels_csv = data_dir / "labels.csv"
    df.to_csv(labels_csv, index=False)
    return df, extract_dir


def populate_gui_samples(df: pd.DataFrame, sample_dir: Path, n_per_grade: int = 1) -> None:
    """Populate GUI sample folder with representative IDRiD images of each DR grade."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/5] Populating GUI sample directory: {sample_dir}")

    for grade in range(5):
        subset = df[df["grade"] == grade]
        if not subset.empty:
            for _, row in subset.head(n_per_grade).iterrows():
                src = Path(row["image_path"])
                dst = sample_dir / f"IDRiD_grade{grade}_{src.name}"
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
                print(f" - Sample Grade {grade}: {dst.name}")


def run_indian_dataset_evaluation(cfg: dict, df: pd.DataFrame, n_samples: int = 15) -> None:
    """Run preprocessing, MedSigLIP inference, and Grad-CAM on IDRiD images."""
    print(f"\n[3/5] Initializing MedSigLIP + Explainability model...")
    device = resolve_device(cfg["model"].get("device", "auto"))

    encoder = MedSigLIPEncoder(cfg, device)
    classifier = MLPHead(
        in_dim=encoder.embedding_dim,
        hidden_dim=cfg["head"].get("hidden_dim", 512),
        out_dim=cfg["experiment"].get("num_classes", 5),
        dropout=cfg["head"].get("dropout", 0.1),
    ).to(device)

    # Check for existing checkpoint
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / "multiclass" / "seed_13" / "best.pt"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        classifier.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained classifier checkpoint from {ckpt_path}")
    else:
        print(f"No trained checkpoint at {ckpt_path}; evaluating initialized classifier.")

    model = MedSigLIPExplainableModel(encoder=encoder, classifier=classifier).to(device).eval()

    eval_df = df.head(n_samples).copy()
    output_dir = Path("outputs/idrid_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/5] Running preprocessing, inference, and Grad-CAM on {len(eval_df)} IDRiD images...")
    results = []

    for i, row in eval_df.iterrows():
        raw_path = Path(row["image_path"])
        with Image.open(raw_path) as raw_img:
            raw_rgb = raw_img.convert("RGB")

        # Preprocess fundus image
        prep_img = preprocess_image(raw_rgb, cfg)

        # Vision processor
        inputs = encoder.processor(images=[prep_img], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # Forward with explainability
        logits, exp_output = model.forward_with_explainability(pixel_values, target_size=(448, 448))
        probs = torch.softmax(logits, dim=-1)[0]
        pred_grade = int(exp_output.predicted_grade)
        confidence = float(probs[pred_grade].item())

        # Generate overlay
        overlay_img = exp_output.overlay(raw_rgb, alpha=0.5)

        # Save result images
        sample_name = f"{row['image_id']}_trueG{row['grade']}_predG{pred_grade}"
        overlay_path = output_dir / f"{sample_name}_overlay.png"
        heatmap_path = output_dir / f"{sample_name}_heatmap.png"
        overlay_img.save(overlay_path)
        exp_output.heatmap_image.save(heatmap_path)

        results.append({
            "image_id": row["image_id"],
            "true_grade": row["grade"],
            "pred_grade": pred_grade,
            "confidence": round(confidence, 4),
            "referable_true": row["binary_label"],
            "referable_pred": int(pred_grade >= 2),
            "overlay_path": str(overlay_path),
        })

        print(
            f"[{i+1}/{len(eval_df)}] IDRiD: {row['image_id']} | "
            f"True: Grade {row['grade']} | Predicted: Grade {pred_grade} "
            f"(Conf: {confidence*100:.1f}%) | Overlay saved."
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    print(f"\n[5/5] Evaluation finished! Results and heatmap overlays saved to: {output_dir}")


def main():
    cfg = load_config(Path("config.yaml"))
    data_dir = Path("data/idrid")

    if not (data_dir / "disease_grading.zip").is_file():
        print(f"Error: {data_dir / 'disease_grading.zip'} not found.")
        sys.exit(1)

    df, _ = setup_idrid_dataset(data_dir)
    populate_gui_samples(df, Path("gui/samples"), n_per_grade=2)
    run_indian_dataset_evaluation(cfg, df, n_samples=10)


if __name__ == "__main__":
    main()
