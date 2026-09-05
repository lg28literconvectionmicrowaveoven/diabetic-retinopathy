#!/usr/bin/env python3
"""Measure the smallest annotated IDRiD lesion and a safe image downscale.

IDRiD provides separate binary TIFF masks for microaneurysms (MA), soft
exudates (SE), hard exudates (EX), and haemorrhages (HE).  This utility finds
8-connected foreground components in those masks; it intentionally excludes
the optic-disc mask because it is an anatomical structure, not a DR lesion.

The program does not resample images or masks.  It only reads masks supplied
at runtime and writes reports next to the chosen output path.

Example:
    python tests/measure_smallest_lesion_patch.py data --output reports/idrid_lesion_sizes

The recommended scale keeps the narrowest dimension of *every* annotated
component at least ``--min-pixels-after-resize`` pixels.  This is deliberately
conservative: any component that is one pixel wide means no downscaling can
preserve its geometry at that threshold.  Use the percentile option only after
reviewing the CSV, as it explicitly allows the smallest components to be lost.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Iterator

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - clear CLI error path
    raise SystemExit("Pillow is required: python -m pip install Pillow") from exc


LESION_TOKENS = ("ma", "microaneurysm", "he", "hemorrhage", "haemorrhage", "ex", "hardexudate", "se", "softexudate")
OPTIC_DISC_TOKENS = ("od", "opticdisc", "optic_disc")


def compact_name(path: Path) -> str:
    """Normalize a filename so naming variants such as `Hard Exudates` match."""
    return "".join(ch for ch in path.stem.lower() if ch.isalnum())


def is_optic_disc_mask(path: Path) -> bool:
    name = compact_name(path)
    return any(token in name for token in OPTIC_DISC_TOKENS)


def lesion_type(path: Path) -> str:
    name = compact_name(path)
    # Check long labels before short abbreviations to avoid accidental matches.
    for token, label in (
        ("microaneurysm", "MA"), ("hemorrhage", "HE"), ("haemorrhage", "HE"),
        ("hardexudate", "EX"), ("softexudate", "SE"),
    ):
        if token in name:
            return label
    # IDRiD's standard directory names include exactly MA/HE/EX/SE.
    return next((token.upper() for token in ("ma", "he", "ex", "se") if token in name), "unknown")


def foreground_pixels(mask_path: Path) -> tuple[int, int, set[int]]:
    """Return image width, height, and nonzero pixel indices from a binary mask."""
    with Image.open(mask_path) as image:
        # TIFFs may be palette/RGB; any non-zero channel denotes annotation.
        width, height = image.size
        pixels = image.convert("L").getdata()
        foreground = {index for index, value in enumerate(pixels) if value != 0}
    return width, height, foreground


def components(width: int, height: int, foreground: set[int]) -> Iterator[dict[str, int]]:
    """Yield 8-connected component statistics without adding a scipy dependency."""
    remaining = set(foreground)
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        area = 0
        min_x = max_x = seed % width
        min_y = max_y = seed // width
        while queue:
            index = queue.popleft()
            x, y = index % width, index // width
            area += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbour = ny * width + nx
                        if neighbour in remaining:
                            remaining.remove(neighbour)
                            queue.append(neighbour)
        yield {
            "area_px": area,
            "x": min_x,
            "y": min_y,
            "width_px": max_x - min_x + 1,
            "height_px": max_y - min_y + 1,
        }


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, avoiding a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    position = (len(ordered) - 1) * p / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="IDRiD data directory (searched recursively for .tif/.tiff masks)")
    parser.add_argument("--output", type=Path, default=Path("reports/idrid_lesion_sizes"), help="Output stem; .csv and .json are added")
    parser.add_argument("--min-pixels-after-resize", type=int, default=4, help="Minimum retained component narrow-side pixels (default: 4)")
    parser.add_argument("--ignore-smallest-percent", type=float, default=0.0, help="Allow this bottom percentage of components to be lost (default: 0; conservative)")
    args = parser.parse_args()
    if args.min_pixels_after_resize < 1 or not 0 <= args.ignore_smallest_percent < 100:
        parser.error("min pixels must be >= 1 and ignore percentile must be in [0, 100)")

    candidates = sorted(list(args.dataset_root.rglob("*.tif")) + list(args.dataset_root.rglob("*.tiff")))
    masks = [path for path in candidates if not is_optic_disc_mask(path)]
    if not masks:
        raise SystemExit(f"No non-optic-disc TIFF masks found below {args.dataset_root}")

    rows: list[dict[str, object]] = []
    for mask_path in masks:
        width, height, foreground = foreground_pixels(mask_path)
        label = lesion_type(mask_path)
        for number, component in enumerate(components(width, height, foreground), start=1):
            rows.append({"mask": str(mask_path), "lesion_type": label, "component": number, **component})
    if not rows:
        raise SystemExit("Masks were found but contain no foreground annotations.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    narrow_sides = [min(int(row["width_px"]), int(row["height_px"])) for row in rows]
    patch_sides = [max(int(row["width_px"]), int(row["height_px"])) for row in rows]
    selected_narrow_side = percentile(narrow_sides, args.ignore_smallest_percent)
    scale = min(1.0, args.min_pixels_after_resize / selected_narrow_side)
    summary = {
        "method": "8-connected foreground components in the four IDRiD DR lesion mask types; optic-disc masks excluded",
        "components_measured": len(rows),
        "smallest_component": min(rows, key=lambda row: (int(row["area_px"]), max(int(row["width_px"]), int(row["height_px"])))),
        "smallest_square_patch_side_px": min(patch_sides),
        "smallest_component_narrow_side_px": min(narrow_sides),
        "resize_policy": {
            "min_pixels_after_resize": args.min_pixels_after_resize,
            "ignored_bottom_percent_of_components": args.ignore_smallest_percent,
            "narrow_side_used_px": selected_narrow_side,
            "maximum_uniform_downscale_factor": scale,
            "maximum_reduction_percent": (1 - scale) * 100,
            "interpretation": "Resize each original dimension by at least this factor to retain the requested narrow-side sampling. A factor of 1 means do not downscale.",
        },
        "reports": {"components_csv": str(csv_path)},
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
