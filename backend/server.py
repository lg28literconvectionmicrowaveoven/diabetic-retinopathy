"""Stateless HTTP API for fundus preprocessing and DR grading."""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from explainability import MedSigLIPExplainableModel
from models import MLPHead, MedSigLIPEncoder
from preproc.crop import prepare_for_medsiglip
from preproc.denoise import preprocess_fundus_image
from utils import load_config, resolve_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))

# Deployments may add their public frontend URL with CORS_ORIGINS.  Keeping the
# development origin in the default makes the browser API usable out of the box.
cors_origins_raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
allow_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials="*" not in allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, __: RequestValidationError) -> JSONResponse:
    # Both public POST endpoints use 400, rather than FastAPI's default 422,
    # for malformed request bodies so clients have one invalid-input path.
    return JSONResponse(status_code=400, content={"detail": "invalid request body"})


class AnalyzeRequest(BaseModel):
    image: str


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _decode_data_url(value: str) -> Image.Image:
    """Decode a PNG/JPEG data URL without accepting arbitrary base64 text."""
    if not isinstance(value, str) or not value.startswith("data:"):
        raise _bad_request("image must be a PNG or JPEG data URL")
    try:
        header, payload = value.split(",", 1)
    except ValueError as exc:
        raise _bad_request("image data URL is malformed") from exc
    if header.lower() not in {"data:image/png;base64", "data:image/jpeg;base64"}:
        raise _bad_request("image must be a PNG or JPEG data URL")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _bad_request("image data URL has invalid base64 data") from exc
    if not raw:
        raise _bad_request("image data URL is empty")
    return _open_image(raw)


def _open_image(raw: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as source:
            source.load()  # force decoding while the BytesIO is still open
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise _bad_request("file is not a valid image") from exc


def _png_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _regions_from_cam(cam: torch.Tensor, max_regions: int = 5) -> list[dict[str, float | bool]]:
    """Turn a normalized Grad-CAM map into normalized connected-component boxes."""
    array = cam.detach().float().cpu().numpy()
    if array.ndim != 2 or not np.isfinite(array).all() or float(array.max()) <= 0:
        return []
    # A relative threshold is robust to CAMs that are non-zero across the FOV.
    mask = (array >= max(0.55, float(array.max()) * 0.70)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = array.shape
    candidates: list[dict[str, float | bool]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        if area < max(4, int(width * height * 0.001)):
            continue
        score = float(array[y : y + box_height, x : x + box_width].max())
        candidates.append({
            "x": round(x / width, 6), "y": round(y / height, 6),
            "width": round(box_width / width, 6), "height": round(box_height / height, 6),
            "score": round(score, 6), "normalized": True,
        })
    return sorted(candidates, key=lambda region: float(region["score"]), reverse=True)[:max_regions]


class ModelService:
    """Lazily loads the large encoder and checkpoint on the first analysis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        config_path = Path(os.environ.get("DR_CONFIG", PROJECT_ROOT / "config.yaml"))
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        cfg = load_config(config_path)
        self.device = resolve_device(cfg["model"].get("device", "auto"))
        default_checkpoint = (PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / "multiclass"
                              / f"seed_{cfg['experiment']['seeds'][0]}" / "best.pt")
        checkpoint_path = Path(os.environ.get("DR_CHECKPOINT", default_checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Model checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        head = MLPHead(
            in_dim=checkpoint["in_dim"], hidden_dim=checkpoint["hidden_dim"],
            out_dim=checkpoint["num_classes"], dropout=checkpoint["dropout"],
        ).to(self.device)
        head.load_state_dict(checkpoint["model_state_dict"])
        head.eval()
        encoder = MedSigLIPEncoder(cfg, self.device)
        self.model = MedSigLIPExplainableModel(encoder, head).to(self.device).eval()
        self.processor = encoder.processor
        self.referable_threshold = int(cfg["experiment"].get("referable_threshold", 2))
        self._loaded = True
        logger.info("DR model loaded from %s", checkpoint_path)

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        # Hooks used for Grad-CAM store activations on the model, so requests must
        # not run concurrently against one loaded model instance.
        with self._lock:
            self._load()
            started = time.perf_counter()
            inputs = self.processor(images=[image], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            explain_started = time.perf_counter()
            logits, explanation = self.model.forward_with_explainability(
                pixel_values, target_size=(image.height, image.width), num_classes=5
            )
            explain_ms = (time.perf_counter() - explain_started) * 1000
            probabilities = torch.softmax(logits, dim=-1)[0]
            grade = int(probabilities.argmax().item())
            confidence = float(probabilities[grade].item())
            return {
                "dr_grade": grade,
                "confidence": round(confidence, 6),
                "referable_dr": grade >= self.referable_threshold,
                "human_review": grade >= self.referable_threshold,
                "analyze_time_ms": round((time.perf_counter() - started) * 1000, 2),
                "explainability_time_ms": round(explain_ms, 2),
                "gradcam": {"regions": _regions_from_cam(explanation.predicted_attribution)},
            }


model_service = ModelService()


@app.get("/health")
async def health():
    return {"status": "pass"}


@app.post("/preprocess")
async def preprocess(file: UploadFile | None = File(default=None)):
    if file is None:
        raise _bad_request("file is required")
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in SUPPORTED_IMAGE_TYPES and suffix not in SUPPORTED_SUFFIXES:
        raise _bad_request("file must be a supported image (jpg, png, bmp, tif, or webp)")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file is too large")
    if not raw:
        raise _bad_request("file is empty")
    try:
        image = _open_image(raw)
        started = time.perf_counter()
        prepared, _ = prepare_for_medsiglip(image)
        result = preprocess_fundus_image(prepared)
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "image": _png_data_url(result), "preprocessing_time_ms": round(elapsed, 2),
            "width": result.width, "height": result.height,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail=f"preprocessing failed: {exc}") from exc


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    image = _decode_data_url(request.image)
    try:
        return await asyncio.to_thread(model_service.analyze, image)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
