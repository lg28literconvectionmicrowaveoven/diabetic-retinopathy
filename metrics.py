from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def compute_multiclass_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    referable_threshold: int = 2,
) -> dict:
    probs = softmax(logits)
    pred = probs.argmax(axis=1)

    y_true_ref = (y_true >= referable_threshold).astype(int)
    y_pred_ref = (pred >= referable_threshold).astype(int)
    ref_probs = probs[:, referable_threshold:].sum(axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, pred, average="weighted", zero_division=0)
        ),
        "qwk": float(
            cohen_kappa_score(y_true, pred, weights="quadratic")
        ) if len(np.unique(y_true)) > 1 else 0.0,
        "referable_sensitivity": float(
            recall_score(y_true_ref, y_pred_ref, zero_division=0)
        ),
        "referable_specificity": float(
            recall_score(1 - y_true_ref, 1 - y_pred_ref, zero_division=0)
        ),
        "referable_precision": float(
            precision_score(y_true_ref, y_pred_ref, zero_division=0)
        ),
        "referable_auc": float(roc_auc_score(y_true_ref, ref_probs))
        if len(np.unique(y_true_ref)) == 2
        else float("nan"),
        "confusion_matrix": confusion_matrix(
            y_true, pred, labels=[0, 1, 2, 3, 4]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1, 2, 3, 4],
            output_dict=True,
            zero_division=0,
        ),
    }
