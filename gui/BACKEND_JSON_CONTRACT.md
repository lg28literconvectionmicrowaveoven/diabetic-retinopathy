# Backend → GUI JSON contract

Preferred response:

```json
{
  "dr_grade": 3,
  "confidence": 0.914,
  "referable_dr": true,
  "human_review": true,
  "explainability_time_ms": 142.3,

  "evidence": {
    "microaneurysms": 18,
    "hemorrhages": 7,
    "exudates": 12
  },

  "gradcam": {
    "regions": [
      {
        "x": 0.31,
        "y": 0.22,
        "width": 0.12,
        "height": 0.10,
        "score": 0.91,
        "normalized": true
      }
    ]
  }
}
```

The GUI supports three region forms:

1. Rectangle:
   `{x, y, width, height, score, normalized}`

2. Corner rectangle:
   `{x1, y1, x2, y2, score, normalized}`

3. Polygon:
   `{polygon: [[x,y], ...], score, normalized}`

Coordinates should preferably be normalized to `[0, 1]` relative to
the original image. The origin is the top-left.

The GUI maps those coordinates back onto the displayed image regardless
of the window/canvas scaling.

The GUI also accepts top-level `regions` or `gradcam_regions` as a
fallback.
