from __future__ import annotations
import torch
from torch import nn

from .dinov3_adapter import TokenBatch


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
        q = self.backbone.prepare_tokens(query)
        if not isinstance(q, TokenBatch):
            raise TypeError("DualInputViT requires DINOv3Adapter TokenBatch output")
        if context is None:
            return self.head(self.backbone.forward_blocks(q.tokens, q.grid_size)[:, 0])

        c = self.backbone.prepare_tokens(context)
        if q.grid_size != c.grid_size:
            raise ValueError("query and context must have equal patch grids")
        prefix_len = q.tokens.shape[1] - q.grid_size[0] * q.grid_size[1]
        # Preserve the query CLS/storage tokens once, concatenate only patches.
        tokens = torch.cat((q.tokens[:, :prefix_len], q.tokens[:, prefix_len:], c.tokens[:, prefix_len:]), dim=1)
        grid = (q.grid_size[0], q.grid_size[1] * 2)
        return self.head(self.backbone.forward_blocks(tokens, grid)[:, 0])
