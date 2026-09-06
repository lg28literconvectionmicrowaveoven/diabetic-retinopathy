#!/usr/bin/env python3
"""Classify cropped retinal scans with the released QuickQual quality gate.

QuickQual extracts DenseNet-121 ImageNet features from a square RGB image at
512 px and passes them to the authors' released SVM.  It returns Good, Usable,
or Bad.  This gate deliberately rejects *only* Bad scans; Good and Usable scans
continue to the DR pipeline.

Download ``quickqual_dn121_512.pkl`` from the QuickQual GitHub release, then:

    python preproc/quickqual_quality_gate.py prepared_retinas \
        --svm path/to/quickqual_dn121_512.pkl \
        --output reports/quickqual_quality.csv

Input should be raw, pre-CLAHE fundus images.  The gate detects and crops the
retinal FOV to a square before resizing its in-memory copy to 512 px for
scoring; it never rewrites input images.  Run it before the final 448 px
MedSigLIP resize, so quality assessment is not based on an upscaled 448 image.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import torch
import timm
from PIL import Image
from torchvision.transforms import functional as transform

from prepare_messidor_retina_448 import containing_square, fov_bounds

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
QUICKQUAL_LABELS = ("good", "usable", "bad")  # Published QuickQual SVM probability order.


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image format: {source.suffix}")
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    raise ValueError(f"input does not exist: {source}")


def make_tensor(path: Path, device: torch.device, threshold: int | None) -> torch.Tensor:
    """Apply the RGB/512/[-1, 1] preprocessing used in QuickQual's README."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    left, top, right, bottom, _ = fov_bounds(image, threshold)
    image = image.crop(containing_square(left, top, right, bottom))
    image = transform.resize(image, 512)
    tensor = transform.to_tensor(image)
    tensor = transform.normalize(tensor, [0.5] * 3, [0.5] * 3)
    return tensor.unsqueeze(0).to(device)


def load_models(svm_path: Path, device: torch.device):
    if not svm_path.is_file():
        raise FileNotFoundError(
            f"QuickQual SVM not found: {svm_path}. Download quickqual_dn121_512.pkl from the QuickQual release."
        )
    classifier = joblib.load(svm_path)
    if not hasattr(classifier, "predict_proba"):
        raise TypeError("the supplied model does not implement predict_proba; expected QuickQual's SVM .pkl")
    encoder = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
    encoder.eval().to(device)
    return encoder, classifier


def score(path: Path, encoder, classifier, device: torch.device, threshold: int | None) -> dict[str, object]:
    with torch.inference_mode():
        features = encoder(make_tensor(path, device, threshold)).squeeze().cpu().numpy().reshape(1, -1)
    probabilities = classifier.predict_proba(features)[0]
    if len(probabilities) != len(QUICKQUAL_LABELS):
        raise ValueError(f"expected a 3-class QuickQual SVM, received {len(probabilities)} probabilities")
    by_label = dict(zip(QUICKQUAL_LABELS, map(float, probabilities), strict=True))
    predicted = max(by_label, key=by_label.get)
    return {
        "image": str(path),
        "good_probability": by_label["good"],
        "usable_probability": by_label["usable"],
        "bad_probability": by_label["bad"],
        "quality": predicted,
        "decision": "reject_and_reacquire" if predicted == "bad" else "continue_to_dr_grading",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Retina-square image or directory of pre-CLAHE images")
    parser.add_argument("--svm", type=Path, required=True, help="Path to quickqual_dn121_512.pkl from the official release")
    parser.add_argument("--output", type=Path, default=Path("reports/quickqual_quality.csv"), help="CSV report path")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu (default: cuda if available)")
    parser.add_argument("--threshold", type=int, default=None, help="Fixed green-channel FOV threshold (0-255); default: automatic")
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 255:
        parser.error("--threshold must be in the range 0-255")

    images = find_images(args.input)
    if not images:
        raise SystemExit(f"no supported images found under {args.input}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder, classifier = load_models(args.svm, device)

    rows = []
    for image in images:
        try:
            row = score(image, encoder, classifier, device, args.threshold)
            rows.append(row)
            print(f"{row['quality'].upper():6} {row['decision']:23} {image}")
        except (OSError, ValueError) as exc:
            print(f"ERROR  manual_review_required  {image}: {exc}")

    if not rows:
        raise SystemExit("no scans were scored")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rejected = sum(row["decision"] == "reject_and_reacquire" for row in rows)
    print(f"wrote {args.output}; {rejected}/{len(rows)} scan(s) marked reject_and_reacquire")


if __name__ == "__main__":
    main()
