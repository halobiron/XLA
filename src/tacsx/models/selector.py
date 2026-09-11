from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class TACSSelector(nn.Module):
    """Generic selector independent of a specific DINOv3 API.

    Encoder contract:
      encode_global(images: [B,C,H,W]) -> [B,D]
      embedding_dim: int
    """

    def __init__(self, encoder: nn.Module, projection: bool = True):
        super().__init__()
        self.encoder = encoder
        dim = int(encoder.embedding_dim)
        self.projector = (
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            if projection else nn.Identity()
        )

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 5:
            raise ValueError("candidates must have shape [B,N,C,H,W]")
        b, n, c, h, w = candidates.shape
        if query.shape[0] != b:
            raise ValueError("query and candidates must share batch size")

        zq = self.encoder.encode_global(query)
        zc = self.encoder.encode_global(candidates.reshape(b*n, c, h, w)).reshape(b, n, -1)

        zq = F.normalize(self.projector(zq), dim=-1)
        zc = F.normalize(self.projector(zc), dim=-1)
        return torch.einsum("bd,bnd->bn", zq, zc)


def straight_through_select(scores, candidates, tau: float = 0.1):
    mask = F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)
    selected = torch.einsum("bn,bnchw->bchw", mask, candidates)
    return selected, mask


def topk_context_weights(scores: torch.Tensor, k: int):
    if k < 1 or k > scores.shape[-1]:
        raise ValueError("k must satisfy 1 <= k <= number of candidates")
    values, indices = torch.topk(scores, k=k, dim=-1)
    return indices, F.softmax(values, dim=-1)


def aggregate_topk_context(scores: torch.Tensor, candidates: torch.Tensor, k: int):
    """Extension: convex pixel-space aggregate of the selector's Top-K images."""
    indices, weights = topk_context_weights(scores, k)
    batch = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
    selected = candidates[batch, indices]
    return (selected * weights[..., None, None, None]).sum(dim=1), indices, weights
