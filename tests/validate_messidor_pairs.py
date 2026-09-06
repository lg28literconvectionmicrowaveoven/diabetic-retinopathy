"""Validate the MESSIDOR-2 patient-pairing rule before grouped CV depends on it.

Run on the machine holding the dataset (set data_root in config.yaml first):

    python tests/validate_messidor_pairs.py --config config.yaml --dataset messidor2

Reports:
  - total images and unique patients implied by the grouping rule
  - eyes-per-patient histogram (MESSIDOR-2 is mostly 2 images per patient)
  - images the rule cannot assign to a patient (these would be ungrouped)
  - eye-code consistency (_0100_/_0200_ codes must not repeat within a patient)
  - inter-eye grade agreement for 2-image patients (informational)

Exit status: 0 only if every image gets a patient_id and no patient repeats
an eye code (either failure makes grouped CV unreliable).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import load_dataset
from utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="messidor2")
    parser.add_argument(
        "--csv",
        default=None,
        help="Directly validate a CSV (id_column,grade_column) instead of loading via config",
    )
    parser.add_argument("--id-column", default="image_id")
    parser.add_argument("--grade-column", default="grade")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        df = df.rename(
            columns={args.id_column: "image_id", args.grade_column: "grade"}
        )
        df["image_id"] = df["image_id"].astype(str)
    else:
        cfg = load_config(args.config)
        df = load_dataset(cfg, args.dataset)

    grouping = (args.config and load_config(args.config).get("grouping")) if not args.csv else None
    if grouping is None:
        grouping = {"mode": "filename_regex", "pattern": r"^\d{8}_(\d+)_"}
        print("No config grouping block found; validating default MESSIDOR-2 rule\n")

    from dataset import derive_patient_ids

    try:
        tagged = derive_patient_ids(df, grouping, strict=True)
    except ValueError as exc:
        print(f"[FAIL] {exc}\n")
        tagged = derive_patient_ids(df, grouping, strict=False)

    counts = Counter(tagged["patient_id"])
    ungrouped = [p for p in counts if str(p).startswith("__ungrouped__")]

    print(f"Images: {len(tagged)}")
    print(f"Unique patients: {len(counts) - len(ungrouped)}")
    print(f"Images without a patient id: {len(ungrouped)}")
    if ungrouped:
        examples = tagged.loc[tagged['patient_id'].str.startswith('__ungrouped__'), 'image_id'].head(10).tolist()
        print(f"  examples: {examples}")

    histogram = Counter(counts.values())
    print("Eyes-per-patient histogram:", dict(sorted(histogram.items())))

    two_eye = tagged.groupby("patient_id")["grade"]
    pair_grades = two_eye.agg(["nunique", "count"])
    pairs = pair_grades[(pair_grades["count"] == 2)]
    if len(pairs):
        agreement = float((pairs["nunique"] == 1).mean())
        print(f"Inter-eye grade agreement (2-image patients): {agreement:.1%}")

    # Eye-code consistency: MESSIDOR-2 names end with a 4-digit code where
    # _0100_ / _0200_ mark the two eyes. A patient with two identical codes
    # means the pairing rule misfired (e.g. wrong capture group).
    eye_codes = tagged["image_id"].astype(str).str.extract(r"_(\d{4})_")[0]
    tagged = tagged.assign(eye_code=eye_codes)
    duplicate_eye_count = 0
    if tagged["eye_code"].notna().all():
        duplicated_eyes = tagged.groupby("patient_id")["eye_code"].agg(["nunique", "count"])
        bad_pairs = duplicated_eyes[
            (duplicated_eyes["count"] > 1) & (duplicated_eyes["nunique"] < duplicated_eyes["count"])
        ]
        duplicate_eye_count = len(bad_pairs)
        codes_seen = Counter(tagged["eye_code"])
        print(f"Eye codes seen: {dict(sorted(codes_seen.items()))}")
        if duplicate_eye_count:
            examples = tagged.loc[
                tagged["patient_id"].isin(bad_pairs.index), "image_id"
            ].head(10).tolist()
            print(f"Patients with duplicate eye codes: {duplicate_eye_count}")
            print(f"  examples: {examples}")
        else:
            print("Eye-code check: OK - no patient has the same eye code twice.")
    else:
        print("Eye codes: not extractable from these image ids (skipped).")

    if ungrouped:
        print("\nRESULT: FAIL - fix grouping.pattern before grouped CV.")
        return 1
    if duplicate_eye_count:
        print("\nRESULT: FAIL - duplicate eye codes mean the pairing rule misfired.")
        return 1
    print("\nRESULT: OK - grouping rule assigns every image to a patient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
