"""
Patient-Level Bilateral DR Screening Evaluation Suite
======================================================
Evaluates the trained MedSigLIP + DR MLP classifier on bilateral patient pairs:
  - Both Left Eye (OS) and Right Eye (OD) evaluated per patient
  - Aggregated diagnosis: Grade_patient = max(Grade_left, Grade_right)
  - Referral triage: Referable = 1 if Grade_patient >= 2 else 0

Evaluates on:
  1. Messidor-2 (889 patients, 1744 images)
  2. IDRiD Test Cohort (66 patients, 103 images)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.modules.setdefault("torchaudio", None)

from dataset import preprocess_image
from models import MedSigLIPEncoder, MLPHead
from utils import load_config, resolve_device, save_json


def load_head_model(device):
    cfg = load_config("config.yaml")
    checkpoint_path = Path("outputs/checkpoints/multiclass/seed_13/best.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head = MLPHead(
        in_dim=checkpoint["in_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        out_dim=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()
    return head, cfg


def evaluate_patient_cohort(
    pairs_csv: str,
    emb_dict: dict[str, np.ndarray],
    cohort_name: str,
    head,
    device,
):
    print("\n" + "=" * 75)
    print(f" EVALUATING PATIENT-LEVEL COHORT: {cohort_name}")
    print("=" * 75)

    df_pairs = pd.read_csv(pairs_csv)
    print(f"Loaded {len(df_pairs)} patient examinations from {pairs_csv}")
    print(f" - Bilateral (Both Eyes): {df_pairs['has_both_eyes'].sum()}")
    print(f" - Single Eye:           {(~df_pairs['has_both_eyes']).sum()}")

    patient_results = []
    
    for _, row in df_pairs.iterrows():
        pid = row["patient_id"]
        has_both = bool(row["has_both_eyes"])
        true_pat_grade = int(row["patient_grade"])
        true_pat_ref = int(row["patient_referable"])

        preds = []
        ref_probs = []

        # Left eye
        left_img = str(row["left_image"]).strip() if pd.notna(row["left_image"]) else ""
        if left_img and left_img in emb_dict:
            emb_l = emb_dict[left_img]
            with torch.no_grad():
                logits_l = head(torch.from_numpy(emb_l.astype(np.float32)).unsqueeze(0).to(device))
                probs_l = torch.softmax(logits_l, dim=-1).cpu().numpy()[0]
            pred_l = int(probs_l.argmax())
            ref_p_l = float(probs_l[2:].sum())
            preds.append(pred_l)
            ref_probs.append(ref_p_l)
        else:
            pred_l, ref_p_l = -1, 0.0

        # Right eye
        right_img = str(row["right_image"]).strip() if pd.notna(row["right_image"]) else ""
        if right_img and right_img in emb_dict:
            emb_r = emb_dict[right_img]
            with torch.no_grad():
                logits_r = head(torch.from_numpy(emb_r.astype(np.float32)).unsqueeze(0).to(device))
                probs_r = torch.softmax(logits_r, dim=-1).cpu().numpy()[0]
            pred_r = int(probs_r.argmax())
            ref_p_r = float(probs_r[2:].sum())
            preds.append(pred_r)
            ref_probs.append(ref_p_r)
        else:
            pred_r, ref_p_r = -1, 0.0

        # Clinical Bilateral Patient Aggregation:
        # Grade_patient = max(Grade_left, Grade_right)
        pred_pat_grade = max(preds) if preds else 0
        pred_pat_ref_prob = max(ref_probs) if ref_probs else 0.0
        pred_pat_ref = 1 if pred_pat_grade >= 2 else 0

        patient_results.append({
            "patient_id": pid,
            "has_both_eyes": has_both,
            "true_left_grade": row["left_grade"],
            "true_right_grade": row["right_grade"],
            "true_patient_grade": true_pat_grade,
            "true_patient_referable": true_pat_ref,
            "pred_left_grade": pred_l,
            "pred_right_grade": pred_r,
            "pred_patient_grade": pred_pat_grade,
            "pred_patient_referable_prob": round(pred_pat_ref_prob, 4),
            "pred_patient_referable": pred_pat_ref,
        })

    df_res = pd.DataFrame(patient_results)

    # Calculate Patient-Level Metrics
    y_true = df_res["true_patient_grade"].to_numpy()
    y_pred = df_res["pred_patient_grade"].to_numpy()
    y_true_ref = df_res["true_patient_referable"].to_numpy()
    y_pred_ref = df_res["pred_patient_referable"].to_numpy()
    y_prob_ref = df_res["pred_patient_referable_prob"].to_numpy()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if len(np.unique(y_true)) > 1 else 0.0

    sens = float(recall_score(y_true_ref, y_pred_ref, zero_division=0))
    spec = float(recall_score(1 - y_true_ref, 1 - y_pred_ref, zero_division=0))
    prec = float(precision_score(y_true_ref, y_pred_ref, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true_ref, y_prob_ref))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    # Bilateral inter-eye concordance on patients with both eyes
    df_bilateral = df_res[df_res["has_both_eyes"]].copy()
    if len(df_bilateral) > 0:
        true_concordance = float((df_bilateral["true_left_grade"] == df_bilateral["true_right_grade"]).mean())
        pred_concordance = float((df_bilateral["pred_left_grade"] == df_bilateral["pred_right_grade"]).mean())
    else:
        true_concordance, pred_concordance = 1.0, 1.0

    metrics = {
        "cohort_name": cohort_name,
        "total_patients": len(df_res),
        "bilateral_patients": int(df_res["has_both_eyes"].sum()),
        "single_eye_patients": int((~df_res["has_both_eyes"]).sum()),
        "patient_accuracy": round(acc, 4),
        "patient_balanced_accuracy": round(bal_acc, 4),
        "patient_macro_f1": round(macro_f1, 4),
        "patient_weighted_f1": round(weighted_f1, 4),
        "patient_qwk": round(qwk, 4),
        "patient_referable_sensitivity": round(sens, 4),
        "patient_referable_specificity": round(spec, 4),
        "patient_referable_precision": round(prec, 4),
        "patient_referable_auc": round(auc, 4),
        "true_bilateral_concordance": round(true_concordance, 4),
        "pred_bilateral_concordance": round(pred_concordance, 4),
        "confusion_matrix": cm,
    }

    print("-" * 60)
    print(f" PATIENT-LEVEL RESULTS: {cohort_name}")
    print("-" * 60)
    print(f" Patient-Level Accuracy:        {acc*100:.2f}%")
    print(f" Patient-Level Balanced Acc:    {bal_acc*100:.2f}%")
    print(f" Patient-Level Macro F1:        {macro_f1:.4f}")
    print(f" Patient-Level Weighted F1:     {weighted_f1:.4f}")
    print(f" Patient-Level QWK:             {qwk:.4f}")
    print(f" Referable DR Sensitivity:      {sens*100:.2f}%")
    print(f" Referable DR Specificity:      {spec*100:.2f}%")
    print(f" Referable DR Precision:        {prec*100:.2f}%")
    print(f" Referable DR AUC-ROC:          {auc:.4f}")
    print(f" True Inter-Eye Concordance:    {true_concordance*100:.2f}%")
    print(f" Pred Inter-Eye Concordance:    {pred_concordance*100:.2f}%")
    print("-" * 60)

    return df_res, metrics


def main():
    print("=================================================================")
    print("   PATIENT-LEVEL BILATERAL EVALUATION SUITE")
    print("=================================================================")

    device = resolve_device("auto")
    print(f"Active Device: {device}")

    head, cfg = load_head_model(device)

    # Build embedding dictionaries
    # 1. Messidor-2
    df_m = pd.read_csv("data/messidor2/messidor_data.csv")
    emb_m = np.load("outputs/embeddings/medsiglip/messidor2.npy")
    m2_emb_dict = {row["id_code"]: emb_m[idx] for idx, row in df_m.iterrows()}
    print(f"Built Messidor-2 embedding index ({len(m2_emb_dict)} images)")

    # 2. IDRiD Test
    df_it = pd.read_csv("data/idrid/labels.csv")
    emb_it = np.load("outputs/embeddings/medsiglip/idrid.npy")
    idrid_emb_dict = {}
    for idx, row in df_it.iterrows():
        id_code = f"{row['image_id']}.jpg"
        idrid_emb_dict[id_code] = emb_it[idx]
        idrid_emb_dict[row['image_id']] = emb_it[idx]
    print(f"Built IDRiD test embedding index ({len(emb_it)} images)")

    # Evaluate Messidor-2
    m2_res, m2_metrics = evaluate_patient_cohort(
        pairs_csv="data/messidor2/patient_pairs.csv",
        emb_dict=m2_emb_dict,
        cohort_name="Messidor-2 (Multi-Center Reference)",
        head=head,
        device=device,
    )
    os.makedirs("outputs/metrics", exist_ok=True)
    m2_res.to_csv("outputs/metrics/patient_level_messidor2_predictions.csv", index=False)

    # Evaluate IDRiD Test
    id_res, id_metrics = evaluate_patient_cohort(
        pairs_csv="data/idrid/patient_pairs_test.csv",
        emb_dict=idrid_emb_dict,
        cohort_name="IDRiD Indian Cohort (Test Split)",
        head=head,
        device=device,
    )
    id_res.to_csv("outputs/metrics/patient_level_idrid_test_predictions.csv", index=False)

    # Save summary
    summary = {
        "messidor2": m2_metrics,
        "idrid_test": id_metrics,
    }
    save_json(summary, Path("outputs/metrics/patient_level_evaluation_summary.json"))
    print("\nSaved full evaluation summary to: outputs/metrics/patient_level_evaluation_summary.json")


if __name__ == "__main__":
    main()
