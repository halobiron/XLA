from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def dino_similarity_scores(encoder, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    b, n, c, h, w = candidates.shape
    q = F.normalize(encoder.encode_global(query), dim=-1)
    cs = encoder.encode_global(candidates.reshape(b * n, c, h, w)).reshape(b, n, -1)
    return torch.einsum("bd,bnd->bn", q, F.normalize(cs, dim=-1))
