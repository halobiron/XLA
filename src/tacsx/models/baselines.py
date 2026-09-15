from __future__ import annotations
import torch
import torch.nn.functional as F


class DINOEmbeddingCache:
    """Fixed-pool embeddings for the frozen DINO-similarity baseline."""

    def __init__(self, candidate_ids: torch.Tensor, embeddings: torch.Tensor):
        self.candidate_ids = candidate_ids
        self.embeddings = embeddings

    def scores(self, encoder, query: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.searchsorted(self.candidate_ids, candidate_ids.to(query.device))
        if (positions >= self.candidate_ids.numel()).any() or not torch.equal(self.candidate_ids[positions], candidate_ids.to(query.device)):
            raise KeyError("candidate ID missing from fixed DINO embedding cache")
        q = F.normalize(encoder.encode_global(query), dim=-1)
        c = self.embeddings[positions]
        return torch.einsum("bd,bnd->bn", q, c)


@torch.no_grad()
def dino_similarity_scores(encoder, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    b, n, c, h, w = candidates.shape
    q = F.normalize(encoder.encode_global(query), dim=-1)
    cs = encoder.encode_global(candidates.reshape(b * n, c, h, w)).reshape(b, n, -1)
    return torch.einsum("bd,bnd->bn", q, F.normalize(cs, dim=-1))
