from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from dataset import preprocess_image
from preprocessing import (
    apply_clahe,
    correct_illumination,
    detect_fundus_mask,
    extract_monochrome,
    normalize_fundus,
    preprocess_fundus_image,
)


def _make_synthetic_fundus(
    h: int = 448,
    w: int = 448,
    radius: int = 200,
    with_gradient: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)
    cv2.circle(img, center, radius, (180, 80, 40), -1)

    # Anatomical features: optic disc, exudates, microaneurysms
    cv2.circle(img, (center[0] - 80, center[1]), 25, (230, 210, 120), -1)
    cv2.circle(img, (center[0] + 60, center[1] + 30), 4, (255, 255, 200), -1)
    cv2.circle(img, (center[0] + 40, center[1] - 50), 3, (80, 10, 10), -1)

    if with_gradient:
        y, x = np.ogrid[:h, :w]
        grad = (x * 0.5 + y * 0.5) / (h + w) * 70.0
        fg = img > 0
        img = np.clip(img.astype(np.float32) + grad[:, :, None] * fg, 0, 255).astype(np.uint8)

    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
    return img, mask


def test_detect_fundus_mask():
    img, _ = _make_synthetic_fundus()
    mask = detect_fundus_mask(img, threshold=10)

    assert mask.shape == (448, 448)
    assert mask.dtype == bool
    assert mask[224, 224] is np.True_
    assert mask[5, 5] is np.False_

    # Edge cases
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detect_fundus_mask(black).all()

    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert detect_fundus_mask(white).all()


def test_extract_monochrome():
    img, _ = _make_synthetic_fundus()
    green = extract_monochrome(img, mode="green")
    assert green.shape == (448, 448)
    assert np.array_equal(green, img[:, :, 1])

    lum = extract_monochrome(img, mode="luminance")
    assert lum.shape == (448, 448)


def test_correct_illumination():
    img, _ = _make_synthetic_fundus(with_gradient=True)
    corrected = correct_illumination(img, sigma=30.0)

    assert corrected.shape == img.shape
    assert corrected.dtype == np.uint8
    assert not np.array_equal(corrected, img)


def test_apply_clahe():
    img, _ = _make_synthetic_fundus()
    enhanced_rgb = apply_clahe(img, clip_limit=2.0, grid_size=(8, 8))
    assert enhanced_rgb.shape == img.shape
    assert enhanced_rgb.dtype == np.uint8

    mono = img[:, :, 1]
    enhanced_mono = apply_clahe(mono, clip_limit=2.0, grid_size=(8, 8))
    assert enhanced_mono.shape == mono.shape
    assert enhanced_mono.dtype == np.uint8


def test_normalize_fundus():
    img, mask = _make_synthetic_fundus()
    norm = normalize_fundus(img, mask=mask, p_min=1.0, p_max=99.0, mask_background=True)

    assert norm.shape == img.shape
    assert norm.dtype == np.uint8
    assert (norm[~mask] == 0).all()
    assert norm[mask].max() > 200

    # Flat image fallback
    flat = np.full((50, 50, 3), 128, dtype=np.uint8)
    norm_flat = normalize_fundus(flat, mask=None)
    assert np.array_equal(norm_flat, flat)


def test_preprocess_fundus_image_pil_contract():
    raw_np, _ = _make_synthetic_fundus()
    raw_pil = Image.fromarray(raw_np)

    for gray in (True, False):
        cfg = {
            "grayscale": gray,
            "grayscale_mode": "green",
            "illumination_correction": True,
            "clahe": True,
            "normalization": True,
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": [8, 8],
            "gaussian_sigma": 30.0,
            "mask_background": True,
        }

        out_pil = preprocess_fundus_image(raw_pil, cfg)
        assert isinstance(out_pil, Image.Image)
        assert out_pil.size == (448, 448)
        assert out_pil.mode == "RGB"

        arr = np.asarray(out_pil)
        if gray:
            assert np.array_equal(arr[:, :, 0], arr[:, :, 1])
            assert np.array_equal(arr[:, :, 1], arr[:, :, 2])


def test_preprocess_image_dataset_hook():
    raw_np, _ = _make_synthetic_fundus()
    raw_pil = Image.fromarray(raw_np)

    cfg_disabled = {"preprocessing": {"enabled": False}}
    out_disabled = preprocess_image(raw_pil, cfg_disabled)
    assert np.array_equal(np.asarray(out_disabled), raw_np)

    cfg_enabled = {
        "preprocessing": {
            "enabled": True,
            "grayscale": True,
            "grayscale_mode": "green",
            "illumination_correction": True,
            "clahe": True,
            "normalization": True,
        }
    }
    out_enabled = preprocess_image(raw_pil, cfg_enabled)
    assert isinstance(out_enabled, Image.Image)
    assert not np.array_equal(np.asarray(out_enabled), raw_np)
