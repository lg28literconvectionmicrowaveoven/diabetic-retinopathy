#!/usr/bin/env python3
"""Prepare an in-memory retinal image for the 448x448 MedSigLIP encoder.

The retinal field is detected from the image itself; input images are never
modified.  For every image, the script finds the non-black field-of-view (FOV),
creates the smallest *square* crop that contains the complete detected FOV, and
then resizes that crop to 448x448.  A circular FOV necessarily leaves black
corners in a square crop.  They are intentionally preserved so no retinal edge
is cut off.

Example
-------
    prepared, metadata = prepare_for_medsiglip(pil_image)
"""

from __future__ import annotations

from typing import Iterable

from PIL import Image


def otsu_threshold(values: Iterable[int]) -> int:
    """Compute an 8-bit Otsu threshold without requiring OpenCV or NumPy."""
    histogram = [0] * 256
    total = 0
    for value in values:
        histogram[value] += 1
        total += 1
    if total == 0:
        raise ValueError("cannot threshold an empty image")

    total_sum = sum(index * count for index, count in enumerate(histogram))
    background_count = background_sum = 0
    best_threshold, best_variance = 0, -1.0
    for threshold, count in enumerate(histogram):
        background_count += count
        background_sum += threshold * count
        foreground_count = total - background_count
        if background_count == 0 or foreground_count == 0:
            continue
        mean_background = background_sum / background_count
        mean_foreground = (total_sum - background_sum) / foreground_count
        variance = background_count * foreground_count * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_threshold, best_variance = threshold, variance
    return best_threshold


def fov_bounds(image: Image.Image, threshold: int | None) -> tuple[int, int, int, int, int]:
    """Return (left, top, right, bottom, threshold) for the retinal FOV.

    The green channel is used because fundus FOVs remain visible there even when
    red illumination varies.  The row/column support test rejects isolated
    bright artifacts in the otherwise black border.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    green = list(rgb.getchannel("G").getdata())
    chosen_threshold = threshold if threshold is not None else max(8, otsu_threshold(green) // 2)

    # Count pixels above threshold in each row and column.  A real FOV occupies
    # many pixels along either direction; a label or specular border artifact
    # does not.  The 0.5% support is deliberately permissive at the circular rim.
    column_count = [0] * width
    row_count = [0] * height
    for index, value in enumerate(green):
        if value > chosen_threshold:
            x, y = index % width, index // width
            column_count[x] += 1
            row_count[y] += 1
    min_column_support = max(2, round(height * 0.005))
    min_row_support = max(2, round(width * 0.005))
    valid_columns = [x for x, count in enumerate(column_count) if count >= min_column_support]
    valid_rows = [y for y, count in enumerate(row_count) if count >= min_row_support]
    if not valid_columns or not valid_rows:
        raise ValueError(f"no retinal FOV found with green-channel threshold {chosen_threshold}")
    return min(valid_columns), min(valid_rows), max(valid_columns) + 1, max(valid_rows) + 1, chosen_threshold


def containing_square(left: int, top: int, right: int, bottom: int) -> tuple[int, int, int, int]:
    """Return the smallest square centered on the detected FOV's bounding box."""
    side = max(right - left, bottom - top)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    square_left = round(center_x - side / 2)
    square_top = round(center_y - side / 2)
    return square_left, square_top, square_left + side, square_top + side


def prepare_for_medsiglip(
    image: Image.Image,
    threshold: int | None = None,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> tuple[Image.Image, dict[str, object]]:
    """Return a 448x448 RGB PIL image and detected FOV/crop metadata.

    The input image is not modified.  Crop regions beyond its boundary are
    black-padded by Pillow so the complete detected retinal field is preserved.
    """
    image = image.convert("RGB")
    fov_left, fov_top, fov_right, fov_bottom, used_threshold = fov_bounds(image, threshold)
    crop_left, crop_top, crop_right, crop_bottom = containing_square(fov_left, fov_top, fov_right, fov_bottom)

    # PIL fills crop regions lying outside the source image with black.  This is
    # necessary only when a full FOV is against an image border, and is safer than
    # shifting/cutting the crop because the requested policy is to retain all FOV.
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    prepared = crop.resize((448, 448), resample)
    metadata = {
        "source_width": image.width,
        "source_height": image.height,
        "threshold": used_threshold,
        "fov_left": fov_left,
        "fov_top": fov_top,
        "fov_right": fov_right,
        "fov_bottom": fov_bottom,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_right": crop_right,
        "crop_bottom": crop_bottom,
        "crop_side": crop_right - crop_left,
        "resample": resample.name.lower(),
    }
    return prepared, metadata
