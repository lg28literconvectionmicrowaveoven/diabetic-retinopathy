from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Classifier head operating on frozen foundation embeddings."""

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
        return self.forward(embedding)

    def class_gradient_dz(
        self,
        z: torch.Tensor,
        num_classes: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        linear1, relu, dropout, linear2 = self.net
        u = linear1(z)
        h = relu(u)
        logits = linear2(dropout(h) if self.training else h)

        mask = (h > 0).float()
        g_h = linear2.weight.unsqueeze(0) * mask.unsqueeze(1)
        g_z = torch.matmul(g_h, linear1.weight)
        return logits, g_z
