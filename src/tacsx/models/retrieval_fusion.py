"""Image-only native-feature retrieval fusion (explicit TACS-X extension).

Candidate labels are intentionally absent from this API. Query and context
predictions use the same saved classification head and fixed image features.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionConfig:
    k: int = 16
    retrieval_temperature: float = .07
    alpha: float = .25
    method: str = "probability_pool"
    gate: str = "fixed"
    context_temperature: float = 1.
    query_temperature: float = 1.

    def validate(self):
        if self.k < 1 or not 0 <= self.alpha <= 1:
            raise ValueError("k must be positive and alpha in [0,1]")
        if min(self.retrieval_temperature, self.context_temperature, self.query_temperature) <= 0:
            raise ValueError("temperatures must be positive")
        if self.method not in {"probability_pool", "feature_pool", "logit_pool", "log_probability_pool", "pseudo_label_pool", "pseudo_log_pool"}:
            raise ValueError("unknown fusion method")
        if self.gate not in {"fixed", "uncertainty", "relative_confidence"}:
            raise ValueError("unknown gate")


def retrieve(query_features, pool_features, pool_ids, k, query_ids=None,
             query_identity_ids=None, pool_identity_ids=None):
    if query_features.ndim != 2 or pool_features.ndim != 2:
        raise ValueError("features must be matrices")
    if query_features.shape[1] != pool_features.shape[1] or not 1 <= k < len(pool_features):
        raise ValueError("incompatible feature dimensions or candidate count")
    similarities = F.normalize(query_features, dim=-1) @ F.normalize(pool_features, dim=-1).T
    if query_ids is not None:
        similarities = similarities.masked_fill(query_ids[:, None] == pool_ids[None, :], -torch.inf)
    if (query_identity_ids is None) != (pool_identity_ids is None):
        raise ValueError("Both query and pool identity IDs are required")
    if query_identity_ids is not None:
        if query_identity_ids.shape != (len(query_features),) or pool_identity_ids.shape != (len(pool_features),):
            raise ValueError("Identity shape mismatch")
        similarities = similarities.masked_fill(query_identity_ids[:, None] == pool_identity_ids[None, :], -torch.inf)
    values, indices = similarities.topk(k, dim=-1)
    if not torch.isfinite(values).all():
        raise ValueError("insufficient non-self candidates")
    return values, indices


def combine_predictions(query_logits, context_probs, config, context_logits=None):
    query_probs = (query_logits / config.query_temperature).softmax(-1)
    alpha = query_logits.new_full((len(query_logits), 1), config.alpha)
    if config.gate == "uncertainty":
        alpha = alpha * (1 - query_probs.max(-1, keepdim=True).values)
    elif config.gate == "relative_confidence":
        delta = context_probs.max(-1, keepdim=True).values - query_probs.max(-1, keepdim=True).values
        alpha = alpha * torch.sigmoid(8 * delta)
    if config.method == "logit_pool":
        logits = (1 - alpha) * query_logits / config.query_temperature + alpha * context_logits / config.context_temperature
        return logits.softmax(-1)
    if config.method in {"log_probability_pool", "pseudo_log_pool"}:
        logits = (1 - alpha) * query_probs.clamp_min(1e-12).log() + alpha * context_probs.clamp_min(1e-12).log()
        return logits.softmax(-1)
    return (1 - alpha) * query_probs + alpha * context_probs


def fuse_context_logits(query_logits, selected_logits, similarities, config):
    """Consensus from predictions of any fixed image classifier (no labels)."""
    config.validate()
    weights = (similarities / config.retrieval_temperature).softmax(-1)
    context_logits = (selected_logits * weights[..., None]).sum(1)
    if config.method in {"probability_pool", "log_probability_pool"}:
        cp = ((selected_logits / config.context_temperature).softmax(-1) * weights[..., None]).sum(1)
    elif config.method in {"pseudo_label_pool", "pseudo_log_pool"}:
        cp = torch.zeros_like(query_logits).scatter_add_(1, selected_logits.argmax(-1), weights)
    else:
        cp = (context_logits / config.context_temperature).softmax(-1)
    return combine_predictions(query_logits, cp, config, context_logits), weights


def fuse_retrieved(query_logits, selected_features, similarities, weight, bias, config):
    config.validate()
    if selected_features.shape[1] != config.k or similarities.shape != selected_features.shape[:2]:
        raise ValueError("retrieved context shapes do not match k")
    weights = (similarities / config.retrieval_temperature).softmax(-1)
    pooled = (selected_features * weights[..., None]).sum(1)
    context_logits = F.linear(pooled, weight, bias)
    if config.method in {"probability_pool", "log_probability_pool"}:
        per_context = (F.linear(selected_features, weight, bias) / config.context_temperature).softmax(-1)
        context_probs = (per_context * weights[..., None]).sum(1)
    elif config.method in {"pseudo_label_pool", "pseudo_log_pool"}:
        predicted = F.linear(selected_features, weight, bias).argmax(-1)
        context_probs = query_logits.new_zeros(query_logits.shape)
        context_probs.scatter_add_(1, predicted, weights)
    else:
        context_probs = (context_logits / config.context_temperature).softmax(-1)
    probs = combine_predictions(query_logits, context_probs, config, context_logits)
    return probs, weights


class NativeContextFusion(nn.Module):
    def __init__(self, pool_features, pool_ids, weight, bias, config):
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer("pool_features", pool_features)
        self.register_buffer("pool_ids", pool_ids)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, query_features, query_ids=None):
        query_logits = F.linear(query_features, self.weight, self.bias)
        values, indices = retrieve(query_features, self.pool_features, self.pool_ids,
                                   self.config.k, query_ids)
        probs, weights = fuse_retrieved(query_logits, self.pool_features[indices], values,
                                        self.weight, self.bias, self.config)
        return {"logits": probs.clamp_min(1e-12).log(), "context_ids": self.pool_ids[indices],
                "context_weights": weights, "config": asdict(self.config)}


class NativeResidualClassifier(nn.Module):
    """Same-capacity query-only control or image-context residual extension.

    Both start EXACTLY at the frozen query classifier. The control substitutes
    existing query features for the context slot; it performs no self-retrieval.
    """
    def __init__(self, weight, bias, *, use_context=True, hidden=384, dropout=.2):
        super().__init__()
        self.register_buffer("weight", weight.clone())
        self.register_buffer("bias", bias.clone())
        self.use_context = use_context
        dim, classes = weight.shape[1], weight.shape[0]
        size = 4 * dim + 4 * classes
        self.correction = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, classes))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, query_features, context_features=None):
        if self.use_context:
            if context_features is None:
                raise ValueError("Context features required")
            c = context_features
        else:
            c = query_features
        q = query_features
        ql, cl = F.linear(q, self.weight, self.bias), F.linear(c, self.weight, self.bias)
        x = torch.cat((q, c, q * c, q - c, ql, cl, ql.softmax(-1), cl.softmax(-1)), dim=-1)
        return ql + self.correction(x)
