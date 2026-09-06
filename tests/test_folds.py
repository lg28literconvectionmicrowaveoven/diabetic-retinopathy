"""Unit tests for grouped fold construction and patient-id derivation.

Run with:  python -m pytest tests/test_folds.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset import derive_patient_ids
from train import make_folds


def make_paired_df(n_patients: int = 60, seed: int = 7) -> pd.DataFrame:
    """Synthetic paired dataset: two eyes per patient, imbalanced grades."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n_patients):
        base = int(rng.choice([0, 0, 1, 1, 2, 3, 4]))
        fellow = int(np.clip(base + rng.choice([-1, 0, 0, 1]), 0, 4))
        rows.append(
            {
                "image_id": f"20050101_{patient:05d}_0100.tif",
                "grade": base,
                "patient_id": f"{patient:05d}",
            }
        )
        rows.append(
            {
                "image_id": f"20050101_{patient:05d}_0200.tif",
                "grade": fellow,
                "patient_id": f"{patient:05d}",
            }
        )
    return pd.DataFrame(rows)


CFG = {
    "experiment": {
        "n_folds": 5,
        "fold_seed": 42,
        "dev_val_size": 0.15,
    }
}


def test_grouped_folds_keep_eyes_together():
    df = make_paired_df()
    folds = make_folds(df, CFG)

    assert len(folds) == 5
    all_test = np.concatenate([f["test_idx"] for f in folds])
    assert sorted(all_test) == list(range(len(df)))  # every image tested exactly once

    test_groups = [set(df.iloc[f["test_idx"]]["patient_id"]) for f in folds]
    for i, a in enumerate(test_groups):
        for b in test_groups[i + 1 :]:
            assert not (a & b), "a patient leaked into two different test folds"

    # ~20% per fold
    sizes = [len(f["test_idx"]) for f in folds]
    assert all(0.15 <= s / len(df) <= 0.25 for s in sizes)


def test_inner_split_is_grouped():
    df = make_paired_df()
    folds = make_folds(df, CFG)
    for fold in folds:
        train_patients = set(df.iloc[fold["train_idx"]]["patient_id"])
        val_patients = set(df.iloc[fold["val_idx"]]["patient_id"])
        assert not (train_patients & val_patients), "inner val split leaks fellow eyes"
        # val ~15% of dev portion (~12% of all)
        val_frac = len(fold["val_idx"]) / len(df)
        assert 0.07 <= val_frac <= 0.18


def test_effective_sizes_68_12_20():
    df = make_paired_df()
    fold = make_folds(df, CFG)[0]
    total = len(df)
    assert len(fold["train_idx"]) / total == pytest.approx(0.68, abs=0.05)
    assert len(fold["val_idx"]) / total == pytest.approx(0.12, abs=0.04)
    assert len(fold["test_idx"]) / total == pytest.approx(0.20, abs=0.05)


def test_ungrouped_fallback_still_covers_once():
    df = make_paired_df().drop(columns=["patient_id"])  # plain stratified path
    folds = make_folds(df, CFG)
    all_test = np.concatenate([f["test_idx"] for f in folds])
    assert sorted(all_test) == list(range(len(df)))
    for fold in folds:
        assert not set(fold["train_idx"]) & set(fold["val_idx"])
        assert not set(fold["val_idx"]) & set(fold["test_idx"])


def test_derive_patient_ids_filename_regex():
    df = make_paired_df().drop(columns=["patient_id"])
    cfg = {"mode": "filename_regex", "pattern": r"^\d{8}_(\d+)_"}
    out = derive_patient_ids(df, cfg)
    assert set(out["patient_id"]) == {f"{p:05d}" for p in range(60)}
    assert out.groupby("patient_id").size().eq(2).all()


def test_derive_patient_ids_column_and_none():
    df = make_paired_df().drop(columns=["patient_id"]).assign(subj="A")
    assert set(derive_patient_ids(df, {"mode": "column", "column": "subj"})["patient_id"]) == {"A"}
    none_out = derive_patient_ids(df, {"mode": "none"})
    assert none_out["patient_id"].nunique() == len(df)


def test_derive_patient_ids_fails_loudly():
    df = make_paired_df().drop(columns=["patient_id"])
    bad = {"mode": "filename_regex", "pattern": r"^ZZZ(\d+)_"}
    with pytest.raises(ValueError, match="patient_id"):
        derive_patient_ids(df, bad)
    with pytest.raises(ValueError, match="grouping.column"):
        derive_patient_ids(df, {"mode": "column", "column": "missing"})
