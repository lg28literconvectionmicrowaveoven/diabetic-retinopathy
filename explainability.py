from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ExplainabilityOutput:
    """Container for model outputs, cached activations, and attribution maps."""

    logits: torch.Tensor
    attributions: torch.Tensor
    predicted_grade: torch.Tensor | int
    predicted_attribution: torch.Tensor
    spatial_activation: torch.Tensor
    embedding: torch.Tensor

    def __getitem__(self, item: Union[str, int]) -> Any:
        if isinstance(item, int):
            return self.attributions[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Invalid attribute '{item}'.")

    def keys(self) -> list[str]:
        return [
            "logits",
            "attributions",
            "predicted_grade",
            "predicted_attribution",
            "spatial_activation",
            "embedding",
        ]

    def get_attribution(self, grade: int) -> torch.Tensor:
        if not (0 <= grade <= 4):
            raise ValueError(f"DR grade must be within [0, 4], got {grade}")
        return self.attributions[:, grade] if self.attributions.ndim == 4 else self.attributions[grade]

    def overlay(
        self,
        image: Image.Image | np.ndarray,
        grade: int | None = None,
        index: int = 0,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        mask_background: bool = True,
    ) -> Image.Image:
        """Render heatmap overlay on the fundus image for predicted or specified grade."""
        hm = self.get_attribution(grade) if grade is not None else self.predicted_attribution
        return overlay_heatmap(image, hm, index=index, alpha=alpha, colormap=colormap, mask_background=mask_background)


def overlay_heatmap(
    image: Image.Image | np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    index: int = 0,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    mask_background: bool = True,
) -> Image.Image:
    """Overlay a continuous Grad-CAM heatmap onto the original fundus image."""
    if isinstance(image, Image.Image):
        img_np = np.asarray(image.convert("RGB"))
    else:
        img_np = np.asarray(image, dtype=np.uint8)

    if isinstance(heatmap, torch.Tensor):
        hm_np = heatmap.detach().cpu().float().numpy()
    else:
        hm_np = np.asarray(heatmap, dtype=np.float32)

    while hm_np.ndim > 2:
        idx = index if hm_np.shape[0] > index else 0
        hm_np = hm_np[idx]

    h, w = img_np.shape[:2]
    if hm_np.shape != (h, w):
        hm_np = cv2.resize(hm_np, (w, h), interpolation=cv2.INTER_LINEAR)

    hm_min, hm_max = hm_np.min(), hm_np.max()
    hm_norm = (hm_np - hm_min) / (hm_max - hm_min + 1e-8)
    hm_uint8 = np.uint8(255 * hm_norm)

    color_map = cv2.applyColorMap(hm_uint8, colormap)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_np, 1.0 - alpha, color_map, alpha, 0)

    if mask_background:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blended[gray <= 10] = 0

    return Image.fromarray(blended)


def class_logit_gradient(
    classifier: nn.Module,
    embedding: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("Expected embedding shape [1, D].")

    if hasattr(classifier, "class_gradient_dz"):
        with torch.no_grad():
            _, g_z = classifier.class_gradient_dz(embedding, num_classes=5)
            return g_z[0, class_index]

    classifier.zero_grad(set_to_none=True)
    logits = classifier(embedding)
    return torch.autograd.grad(
        logits[0, class_index],
        embedding,
        retain_graph=False,
        create_graph=False,
    )[0]


class MedSigLIPExplainableModel(nn.Module):
    """
    Production-grade explainability wrapper for MedSigLIP + Classifier.

    Vectorizes multi-class Grad-CAM across batch and class dimensions with
    a single encoder forward pass and zero encoder backward passes.
    """

    def __init__(
        self,
        encoder: Any,
        classifier: nn.Module,
        grid_size: int = 28,
        embed_dim: int = 1152,
        target_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        if hasattr(encoder, "model"):
            self.encoder_wrapper = encoder
            raw = encoder.model
            self.vision_encoder = raw.vision_model if hasattr(raw, "vision_model") else raw
        elif hasattr(encoder, "vision_model"):
            self.encoder_wrapper = None
            self.vision_encoder = encoder.vision_model
        else:
            self.encoder_wrapper = None
            self.vision_encoder = encoder

        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(self.vision_encoder, "post_layernorm"):
            self.target_layer = self.vision_encoder.post_layernorm
        elif hasattr(self.vision_encoder, "encoder") and hasattr(self.vision_encoder.encoder, "layers"):
            self.target_layer = self.vision_encoder.encoder.layers[-1]
        else:
            raise AttributeError("Target spatial layer not found in vision encoder.")

        self.cached_spatial_activation: Optional[torch.Tensor] = None
        self.cached_embedding: Optional[torch.Tensor] = None
        self._hook_handle = self.target_layer.register_forward_hook(self._spatial_hook)

    def _spatial_hook(self, module: nn.Module, inputs: Any, output: Any) -> None:
        self.cached_spatial_activation = output[0] if isinstance(output, tuple) else output

    def get_pooling_head(self) -> nn.Module:
        if hasattr(self.vision_encoder, "head"):
            return self.vision_encoder.head
        raise AttributeError("Pooling head not found in vision encoder.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.vision_encoder(pixel_values=pixel_values)
        self.cached_embedding = vision_out.pooler_output
        return self.classifier(vision_out.pooler_output)

    def _compute_dz(self, z: torch.Tensor, num_classes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.classifier, "class_gradient_dz"):
            return self.classifier.class_gradient_dz(z, num_classes=num_classes)

        if hasattr(self.classifier, "net") and len(self.classifier.net) >= 4:
            linear1, relu, dropout, linear2 = self.classifier.net[:4]
            u = linear1(z)
            h = relu(u)
            logits = linear2(dropout(h) if self.training else h)
            mask = (h > 0).float()
            g_h = linear2.weight.unsqueeze(0) * mask.unsqueeze(1)
            g_z = torch.matmul(g_h, linear1.weight)
            return logits, g_z

        z_var = z.detach().requires_grad_(True)
        logits = self.classifier(z_var)
        g_z = torch.stack(
            [torch.autograd.grad(logits[:, k].sum(), z_var, retain_graph=True)[0] for k in range(num_classes)],
            dim=1,
        )
        return logits, g_z

    def forward_with_explainability(
        self,
        pixel_values: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = (448, 448),
        num_classes: int = 5,
    ) -> Tuple[torch.Tensor, ExplainabilityOutput]:
        batch_size = pixel_values.shape[0]

        vision_out = self.vision_encoder(pixel_values=pixel_values)
        z = vision_out.pooler_output
        self.cached_embedding = z

        A = self.cached_spatial_activation
        if A is None:
            raise RuntimeError("Spatial activations were not captured. Check hook registration.")

        logits, g_z = self._compute_dz(z, num_classes=num_classes)
        head = self.get_pooling_head()

        # Vectorized VJP over batch and class dimensions simultaneously
        A_rep = A.repeat_interleave(num_classes, dim=0).detach().requires_grad_(True)
        z_batch = head(A_rep)
        grad_A = torch.autograd.grad(
            outputs=z_batch,
            inputs=A_rep,
            grad_outputs=g_z.reshape(batch_size * num_classes, -1),
        )[0]

        alpha = grad_A.mean(dim=1)
        L = torch.einsum("bnd,bd->bn", A_rep, alpha)
        cams = F.relu(L).view(batch_size, num_classes, self.grid_size, self.grid_size)

        cam_min = cams.amin(dim=(-2, -1), keepdim=True)
        cam_max = cams.amax(dim=(-2, -1), keepdim=True)
        cams = (cams - cam_min) / (cam_max - cam_min + 1e-8)

        if target_size is not None:
            cams = F.interpolate(cams, size=target_size, mode="bilinear", align_corners=False)

        predicted_grade = logits.argmax(dim=-1)
        pred_cams = cams[torch.arange(batch_size, device=cams.device), predicted_grade]
        spatial_2d = A.detach().view(batch_size, self.grid_size, self.grid_size, self.embed_dim)

        is_single = batch_size == 1
        return logits, ExplainabilityOutput(
            logits=logits,
            attributions=cams[0] if is_single else cams,
            predicted_grade=predicted_grade[0].item() if is_single else predicted_grade,
            predicted_attribution=pred_cams[0] if is_single else pred_cams,
            spatial_activation=spatial_2d,
            embedding=z.detach(),
        )

    def close(self) -> None:
        if hasattr(self, "_hook_handle") and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self) -> MedSigLIPExplainableModel:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
