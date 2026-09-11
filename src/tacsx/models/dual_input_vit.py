from __future__ import annotations
import torch
from torch import nn


class DualInputViT(nn.Module):
    """Downstream query + selected-context classifier.

    Coding Agent must implement this against DINOv3Adapter after inspecting
    the official token and positional-embedding APIs.
    """

    def __init__(self, backbone_adapter: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone_adapter
        self.head = nn.Linear(backbone_adapter.embedding_dim, num_classes)

    def forward(self, query: torch.Tensor, context: torch.Tensor | None = None):
        raise NotImplementedError(
            "Implement query/context patch-token concatenation and document "
            "positional-embedding handling."
        )
