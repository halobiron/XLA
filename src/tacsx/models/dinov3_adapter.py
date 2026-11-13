from __future__ import annotations
from pathlib import Path
import sys
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class TokenBatch:
    """Tokens accompanied by their RoPE spatial grid."""

    tokens: torch.Tensor
    grid_size: tuple[int, int]


class DINOv3Adapter(nn.Module):
    """Thin compatibility layer around official DINOv3 ViT-S/16."""

    def __init__(self, repo_dir: str, weights: str | None = None, pretrained: bool | None = None,
                 pair_rope_layout: str = "legacy"):
        super().__init__()
        if pair_rope_layout not in {"legacy", "horizontal", "per_image"}:
            raise ValueError("pair_rope_layout must be legacy, horizontal or per_image")
        self.pair_rope_layout = pair_rope_layout
        repo = Path(repo_dir)
        if not repo.exists():
            raise FileNotFoundError(
                f"DINOv3 repo not found at {repo}. Run scripts/bootstrap.sh first."
            )
        # Import the official backbone factory directly. torch.hub imports every
        # hubconf export, including optional segmentation dependencies, which is
        # unnecessary for the ViT-S/16 backbone used here.
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from dinov3.hub.backbones import dinov3_vits16
        kwargs = {}
        local_weights = Path(weights) if weights and Path(weights).is_file() else None
        if weights and local_weights is None:
            kwargs["weights"] = weights
        elif pretrained is False:
            kwargs["pretrained"] = False
        if local_weights is not None:
            # The official factory delegates local paths through torch.hub's
            # cache, which is inaccessible in some Windows environments. Load
            # the identical official state dict directly instead.
            self.model = dinov3_vits16(pretrained=False)
            state = torch.load(local_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        else:
            self.model = dinov3_vits16(**kwargs)
        self.embedding_dim = self._infer_embedding_dim()
        self.patch_size = int(self.model.patch_size)

    def _infer_embedding_dim(self):
        for name in ("embed_dim", "num_features"):
            value = getattr(self.model, name, None)
            if value is not None:
                return int(value)
        raise AttributeError("Could not infer DINOv3 embedding dimension")

    def encode_global(self, images):
        return self.model.forward_features(images)["x_norm_clstoken"]

    def prepare_tokens(self, images):
        tokens, grid_size = self.model.prepare_tokens_with_masks(images)
        return TokenBatch(tokens=tokens, grid_size=grid_size)

    def _rope_for_tokens(self, grid_size, num_images=1):
        """Keep all DINO-specific multi-image position handling in this adapter.

        Tokens are image-major: [all query patches, all context patches].
        A flattened H x (2W) grid is row-major and MUST be reordered to match.
        ``legacy`` deliberately reproduces the old mismatch for old checkpoints.
        ``per_image`` reuses each image's native grid (an explicit ablation).
        """
        h, w = grid_size
        if num_images == 1 or self.pair_rope_layout == "per_image":
            rope = self.model.rope_embed(H=h, W=w)
            return tuple(part.repeat(num_images, 1) for part in rope)
        rope = self.model.rope_embed(H=h, W=w * num_images)
        if self.pair_rope_layout == "legacy":
            return rope
        return tuple(torch.cat(part.reshape(h, w * num_images, -1).split(w, dim=1), dim=0)
                     .reshape(h * w * num_images, -1) for part in rope)

    def forward_blocks(self, tokens: torch.Tensor, grid_size: tuple[int, int] | None = None,
                       *, num_images: int = 1):
        """Run official blocks then their final normalization.

        ``grid_size`` is required for custom concatenated token layouts because
        DINOv3's rotary embedding is generated at block time.
        """
        if grid_size is None:
            raise ValueError("grid_size is required for DINOv3 RoPE")
        if num_images < 1:
            raise ValueError("num_images must be positive")
        expected = 1 + self.model.n_storage_tokens + num_images * grid_size[0] * grid_size[1]
        if tokens.shape[1] != expected:
            raise ValueError(f"expected {expected} tokens for grid and image count, got {tokens.shape[1]}")
        x = tokens
        for block in self.model.blocks:
            # Official DINOv3 regenerates RoPE per block, including training-time
            # coordinate augmentation. One cached draw per forward differed from it.
            rope = self._rope_for_tokens(grid_size, num_images)
            x = block(x, rope)
        return self.model.norm(x)
