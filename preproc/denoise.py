from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from PIL import Image


def detect_fundus_mask(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Isolate the retinal field from the black camera aperture border."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    mask = gray > threshold
    if not mask.any() or mask.all():
        return np.ones(gray.shape, dtype=bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def extract_monochrome(img: np.ndarray, mode: str = "green") -> np.ndarray:
    """Extract optimal contrast fundus channel (green channel or weighted luminance)."""
    if img.ndim == 2:
        return img
    if mode.lower() == "green":
        return img[:, :, 1]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def correct_illumination(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Graham's subtraction of local low-frequency lighting variations."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 4.0, blur, -4.0, 128)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance structural micro-contrast in CIELAB space (RGB) or directly (monochrome)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if img.ndim == 2:
        return clahe.apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize_fundus(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    p_min: float = 1.0,
    p_max: float = 99.0,
    mask_background: bool = True,
) -> np.ndarray:
    """Robust percentile scaling to [0, 255] with border halo suppression."""
    fg = img[mask] if mask is not None and mask.any() else img
    p_low, p_high = np.percentile(fg, (p_min, p_max))

    if p_high <= p_low:
        scaled = img.copy()
    else:
        scaled = np.clip((img.astype(np.float32) - p_low) / (p_high - p_low + 1e-8) * 255.0, 0, 255).astype(np.uint8)

    if mask_background and mask is not None:
        scaled[~mask] = 0
    return scaled


def preprocess_fundus_image(
    image: Image.Image | np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> Image.Image:
    """
    Standard competitive fundus preprocessing:
    1. Circular mask detection
    2. Optional monochrome/green-channel extraction
    3. Illumination equalization (Graham)
    4. Contrast enhancement (CLAHE)
    5. Dynamic range normalization + aperture border masking
    6. 3-channel RGB projection for vision transformer compatibility
    """
    cfg = cfg or {}

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image, dtype=np.uint8)

    mask = detect_fundus_mask(arr, threshold=int(cfg.get("mask_threshold", 10)))

    if cfg.get("grayscale", True):
        mode = str(cfg.get("grayscale_mode", "green"))
        arr = extract_monochrome(arr, mode=mode)

    if cfg.get("illumination_correction", True):
        arr = correct_illumination(arr, sigma=float(cfg.get("gaussian_sigma", 30.0)))

    if cfg.get("clahe", True):
        grid = tuple(cfg.get("clahe_grid_size", (8, 8)))
        arr = apply_clahe(arr, clip_limit=float(cfg.get("clahe_clip_limit", 2.0)), grid_size=grid)

    if cfg.get("normalization", True):
        arr = normalize_fundus(
            arr,
            mask=mask,
            p_min=float(cfg.get("p_min", 1.0)),
            p_max=float(cfg.get("p_max", 99.0)),
            mask_background=bool(cfg.get("mask_background", True)),
        )

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(arr)
