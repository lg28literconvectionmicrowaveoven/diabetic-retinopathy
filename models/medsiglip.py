from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor


class MedSigLIPEncoder:
    """
    Frozen MedSigLIP vision encoder.

    This class is responsible only for image -> embedding conversion and
    optional embedding caching. The trainable classifier is separate.
    """

    def __init__(self, cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

        model_id = cfg["model"]["name"]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().to(device)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.embedding_dim = self._infer_embedding_dim()

        expected = cfg["model"].get("expected_embedding_dim")
        if expected is not None and self.embedding_dim != expected:
            print(
                f"[WARN] Config expected embedding dim {expected}, "
                f"but loaded MedSigLIP produced {self.embedding_dim}."
            )

        print(
            f"MedSigLIP loaded | device={self.device} | "
            f"embedding_dim={self.embedding_dim}"
        )

        # Spatial activation caching for Grad-CAM explainability
        self.cached_spatial_activation: torch.Tensor | None = None
        self.cached_embedding: torch.Tensor | None = None
        self._target_layer = self._find_late_spatial_layer()
        self._hook_handle = self._target_layer.register_forward_hook(self._spatial_hook)

    def _find_late_spatial_layer(self) -> nn.Module:
        vm = self.model.vision_model if hasattr(self.model, "vision_model") else self.model
        if hasattr(vm, "post_layernorm"):
            return vm.post_layernorm
        elif hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
            return vm.encoder.layers[-1]
        raise AttributeError("Could not locate late spatial layer in MedSigLIP vision model.")

    def _spatial_hook(self, module: Any, inputs: Any, output: Any) -> None:
        act = output[0] if isinstance(output, tuple) else output
        self.cached_spatial_activation = act

    def get_pooling_head(self) -> nn.Module:
        vm = self.model.vision_model if hasattr(self.model, "vision_model") else self.model
        if hasattr(vm, "head"):
            return vm.head
        raise AttributeError("Vision model does not have pooling head module.")

    @torch.inference_mode()
    def _infer_embedding_dim(self) -> int:
        dummy = Image.new("RGB", (448, 448))
        inputs = self.processor(images=[dummy], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_out = self.model.vision_model(
            pixel_values=inputs["pixel_values"]
        )
        return int(vision_out.pooler_output.shape[-1])

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_out = self.model.vision_model(
            pixel_values=inputs["pixel_values"]
        )
        embeddings = vision_out.pooler_output
        self.cached_embedding = embeddings

        return embeddings.detach().float().cpu().numpy()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass with pixel values; preserves spatial activation in hook."""
        vision_out = self.model.vision_model(pixel_values=pixel_values)
        embeddings = vision_out.pooler_output
        self.cached_embedding = embeddings
        return embeddings

    def extract_dataframe(
        self,
        dataframe,
        output_prefix: str | Path,
        batch_size: int | None = None,
    ) -> tuple[Path, Path, Path]:
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        emb_path = Path(str(output_prefix) + "_embeddings.npy")
        ids_path = Path(str(output_prefix) + "_ids.npy")
        labels_path = Path(str(output_prefix) + "_labels.npy")

        if emb_path.exists() and ids_path.exists() and labels_path.exists():
            cached = np.load(emb_path, mmap_mode="r")
            if cached.shape == (len(dataframe), self.embedding_dim):
                print(f"[CACHE] Using {emb_path} with shape {cached.shape}")
                return emb_path, ids_path, labels_path
            print(
                "[CACHE INVALID] Cached shape does not match current dataset. "
                "Rebuilding."
            )

        batch_size = batch_size or self.cfg["model"]["embedding_batch_size"]

        all_embeddings: list[np.ndarray] = []
        all_ids: list[str] = []
        all_labels: list[int] = []

        for start in tqdm(
            range(0, len(dataframe), batch_size),
            desc="Extracting MedSigLIP embeddings",
        ):
            batch_df = dataframe.iloc[start : start + batch_size]

            images = []
            for path in batch_df["image_path"].tolist():
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

            embeddings = self.encode(images)
            all_embeddings.append(embeddings.astype(np.float32))
            all_ids.extend(batch_df["image_id"].astype(str).tolist())
            all_labels.extend(batch_df["grade"].astype(int).tolist())

        embeddings = np.concatenate(all_embeddings, axis=0)

        if embeddings.shape != (len(dataframe), self.embedding_dim):
            raise RuntimeError(
                f"Embedding shape mismatch: got {embeddings.shape}, "
                f"expected {(len(dataframe), self.embedding_dim)}"
            )

        np.save(emb_path, embeddings)
        np.save(ids_path, np.asarray(all_ids, dtype=object))
        np.save(labels_path, np.asarray(all_labels, dtype=np.int64))

        return emb_path, ids_path, labels_path
