"""Explicit TACS-X extension: frozen query logits plus patch-attention correction."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class PatchContextClassifier(nn.Module):
    def __init__(self, classifier_weight, use_context=True, key_dim=64, value_dim=128):
        super().__init__()
        dim = classifier_weight.shape[1]
        self.use_context = use_context
        self.register_buffer("classifier_weight", classifier_weight.detach().clone())
        self.query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, key_dim)
        self.k = nn.Linear(dim, key_dim)
        self.v = nn.Linear(dim, value_dim)
        self.output = nn.Linear(value_dim, dim)
        self.dropout = nn.Dropout(.1)
        self.context_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query_cls, query_patches, baseline_logits, context_patches=None):
        if self.use_context:
            if context_patches is None:
                raise ValueError("Context patches required")
            patches = torch.cat((query_patches, context_patches), dim=1)
        else:
            patches = query_patches
        q = self.q(self.query_norm(query_cls.float()))
        patches = self.patch_norm(patches.float())
        scores = torch.einsum("bd,bpd->bp", q, self.k(patches)) / math.sqrt(q.shape[-1])
        if self.use_context:
            bias = torch.cat((scores.new_zeros(query_patches.shape[1]),
                              self.context_bias.expand(context_patches.shape[1])))
            scores = scores + bias
        weights = self.dropout(scores.softmax(-1))
        pooled = torch.einsum("bp,bpd->bd", weights, self.v(patches))
        correction = F.linear(self.output(pooled), self.classifier_weight)
        # Preserve the original AMP query logits exactly when correction is zero.
        return baseline_logits.float() + correction
