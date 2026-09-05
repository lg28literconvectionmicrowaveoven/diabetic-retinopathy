from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Trainable classifier operating on frozen MedSigLIP embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Helper used by the gradient-based explainability stage."""
        return self.forward(embedding)
