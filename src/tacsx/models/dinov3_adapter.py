from __future__ import annotations
from pathlib import Path
import torch
from torch import nn


class DINOv3Adapter(nn.Module):
    """Thin compatibility layer around official DINOv3 ViT-S/16."""

    def __init__(self, repo_dir: str, weights: str | None = None):
        super().__init__()
        repo = Path(repo_dir)
        if not repo.exists():
            raise FileNotFoundError(
                f"DINOv3 repo not found at {repo}. Run scripts/bootstrap.sh first."
            )
        kwargs = {"source": "local"}
        if weights:
            kwargs["weights"] = weights
        self.model = torch.hub.load(str(repo), "dinov3_vits16", **kwargs)
        self.embedding_dim = self._infer_embedding_dim()

    def _infer_embedding_dim(self):
        for name in ("embed_dim", "num_features"):
            value = getattr(self.model, name, None)
            if value is not None:
                return int(value)
        raise AttributeError("Could not infer DINOv3 embedding dimension")

    def encode_global(self, images):
        # Coding Agent: inspect official API and replace this fallback if needed.
        out = self.model(images)
        if isinstance(out, torch.Tensor) and out.ndim == 2:
            return out
        raise NotImplementedError(
            "Map official DINOv3 output to a [B,D] global embedding here."
        )

    def prepare_tokens(self, images):
        raise NotImplementedError

    def forward_blocks(self, tokens):
        raise NotImplementedError
