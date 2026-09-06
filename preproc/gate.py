#!/usr/bin/env python3
"""Importable QuickQual quality-assessment functions for retinal scans.

QuickQual extracts DenseNet-121 ImageNet features from a square RGB image at
512 px and passes them to the authors' released SVM.  It returns Good, Usable,
or Bad.  This gate deliberately rejects *only* Bad scans; Good and Usable scans
continue to the DR pipeline.

Download ``quickqual_dn121_512.pkl`` from the QuickQual GitHub release, then:

    result = assess_image_quality(pil_image, "models/quickqual_dn121_512.pkl")

Input should be raw, pre-CLAHE fundus images.  The gate detects and crops the
retinal FOV to a square before resizing its in-memory copy to 512 px for
scoring; it never rewrites input images.  Run it before the final 448 px
MedSigLIP resize, so quality assessment is not based on an upscaled 448 image.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import timm
import torch
from PIL import Image
from torchvision.transforms import functional as transform

from .crop import containing_square, fov_bounds

QUICKQUAL_LABELS = (
    "good",
    "usable",
    "bad",
)  # Published QuickQual SVM probability order.


def make_tensor(
    image: Image.Image, device: torch.device, threshold: int | None
) -> torch.Tensor:
    """Apply the RGB/512/[-1, 1] preprocessing used in QuickQual's README."""
    image = image.convert("RGB")
    left, top, right, bottom, _ = fov_bounds(image, threshold)
    image = image.crop(containing_square(left, top, right, bottom))
    image = transform.resize(image, 512)
    tensor = transform.to_tensor(image)
    tensor = transform.normalize(tensor, [0.5] * 3, [0.5] * 3)
    return tensor.unsqueeze(0).to(device)


def load_quickqual_model(
    svm_path: str | Path, device: str | torch.device | None = None
):
    """Load and return the QuickQual DenseNet encoder, SVM, and Torch device."""
    svm_path = Path(svm_path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not svm_path.is_file():
        raise FileNotFoundError(
            f"QuickQual SVM not found: {svm_path}. Download quickqual_dn121_512.pkl from the QuickQual release."
        )
    classifier = joblib.load(svm_path)
    if not hasattr(classifier, "predict_proba"):
        raise TypeError(
            "the supplied model does not implement predict_proba; expected QuickQual's SVM .pkl"
        )
    encoder = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
    encoder.eval().to(device)
    return encoder, classifier, device


def assess_loaded_image(
    image: Image.Image,
    encoder,
    classifier,
    device: str | torch.device,
    threshold: int | None = None,
) -> dict[str, object]:
    """Assess one raw, pre-CLAHE PIL fundus image with an already loaded model."""
    device = torch.device(device)
    with torch.inference_mode():
        features = (
            encoder(make_tensor(image, device, threshold))
            .squeeze()
            .cpu()
            .numpy()
            .reshape(1, -1)
        )
    probabilities = classifier.predict_proba(features)[0]
    if len(probabilities) != len(QUICKQUAL_LABELS):
        raise ValueError(
            f"expected a 3-class QuickQual SVM, received {len(probabilities)} probabilities"
        )
    by_label = dict(zip(QUICKQUAL_LABELS, map(float, probabilities), strict=True))
    predicted = max(by_label, key=by_label.get)
    return {
        "good_probability": by_label["good"],
        "usable_probability": by_label["usable"],
        "bad_probability": by_label["bad"],
        "quality": predicted,
        "decision": "reject_and_reacquire"
        if predicted == "bad"
        else "continue_to_dr_grading",
    }


def assess_image_quality(
    image: Image.Image,
    svm_path: str | Path,
    device: str | torch.device | None = None,
    threshold: int | None = None,
) -> dict[str, object]:
    """Load QuickQual and assess one raw, pre-CLAHE PIL fundus image.

    For a batch, call :func:`load_quickqual_model` once and reuse
    :func:`assess_loaded_image` for each image.
    """
    encoder, classifier, resolved_device = load_quickqual_model(svm_path, device)
    return assess_loaded_image(image, encoder, classifier, resolved_device, threshold)
