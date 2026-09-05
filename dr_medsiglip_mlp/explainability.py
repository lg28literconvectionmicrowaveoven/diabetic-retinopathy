"""Gradcam int:
The current model boundary is:
    image -> frozen MedSigLIP embedding -> MLP logits
This makes embedding-level attribution directly available as:
    d(logit_k) / d(embedding)
Spatial/pixel attribution requires retaining an appropriate intermediate
spatial representation from MedSigLIP rather than only pooler_output.
"""

from __future__ import annotations

import torch
from torch import nn

def class_logit_gradient(
    classifier: nn.Module,
    embedding: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    """Return d(logit_class) / d(embedding) for one sample.
    Parameters
    ----------
    classifier:
        Trained MLP head.
    embedding:
        Tensor of shape [1, D] with requires_grad=True.
    class_index:
        DR class whose logit is being explained.
    """
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("Expected embedding shape [1, D].")

    classifier.zero_grad(set_to_none=True)
    logits = classifier(embedding)
    target = logits[0, class_index]

    return torch.autograd.grad(
        target,
        embedding,
        retain_graph=False,
        create_graph=False,
    )[0]
