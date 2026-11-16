"""TACS-X outcome-aware utility selector, with optional No-Context fallback.

Inference inputs contain image embeddings and model predictions, never labels.
The frozen pair classifier still concatenates image patches before ViT blocks.
This second-stage selector is an explicit extension, not original TACS training.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def prediction_features(pair_logits, query_logits, similarities, allow_null=False):
    b, k, classes = pair_logits.shape
    if query_logits.shape != (b, classes) or similarities.shape != (b, k):
        raise ValueError("Incompatible query/candidate predictions")
    logits = torch.cat((query_logits[:, None], pair_logits), 1) if allow_null else pair_logits
    sim = torch.cat((similarities.new_zeros(b, 1), similarities), 1) if allow_null else similarities
    p, q = logits.float().softmax(-1), query_logits.float().softmax(-1)
    lp, lq = p.clamp_min(1e-8).log(), q.clamp_min(1e-8).log()
    top = p.topk(min(3, classes), dim=-1).values
    qt, qi = q.topk(min(3, classes), dim=-1)
    candidate_class = p.argmax(-1)
    query_class = q.argmax(-1)
    consensus = p.mean(1)
    consensus_class = consensus.argmax(-1)
    gather_qtop = p.gather(-1, qi[:, None].expand(-1, p.shape[1], -1))
    q_for_candidate = q.gather(1, candidate_class)
    consensus_for_candidate = consensus.gather(1, candidate_class)
    candidate_for_consensus = p.gather(-1, consensus_class[:, None, None].expand(-1, p.shape[1], 1)).squeeze(-1)
    entropy = -(p * lp).sum(-1) / math.log(classes)
    query_entropy = -(q * lq).sum(-1) / math.log(classes)
    kl_pq = (p * (lp - lq[:, None])).sum(-1)
    kl_qp = (q[:, None] * (lq[:, None] - lp)).sum(-1)
    margin = top[..., 0] - top[..., 1]
    qm = qt[:, 0] - qt[:, 1]
    null = torch.zeros_like(sim)
    if allow_null:
        null[:, 0] = 1
    rank = torch.arange(p.shape[1], device=p.device, dtype=p.dtype)[None].expand(b, -1) / max(1, p.shape[1] - 1)
    scalars = [entropy, query_entropy[:, None].expand_as(sim), entropy - query_entropy[:, None], margin,
               qm[:, None].expand_as(sim), margin - qm[:, None], kl_pq, kl_qp,
               (p - q[:, None]).square().sum(-1), q_for_candidate, consensus_for_candidate, candidate_for_consensus,
               candidate_class.eq(query_class[:, None]).float(), candidate_class.eq(consensus_class[:, None]).float(),
               sim, rank, null]
    return torch.cat((top, qt[:, None].expand(-1, p.shape[1], -1), gather_qtop,
                      torch.stack(scalars, dim=-1)), dim=-1), logits


class UtilitySelector(nn.Module):
    def __init__(self, embedding_dim=384, hidden=128, allow_null=False, use_embeddings=True):
        super().__init__()
        self.allow_null, self.use_embeddings = allow_null, use_embeddings
        stats_dim = 26
        dim = stats_dim + (4 * embedding_dim if use_embeddings else 0)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(.2),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query_features, candidate_features, pair_logits, query_logits, similarities):
        stats, logits = prediction_features(pair_logits, query_logits, similarities, self.allow_null)
        if self.use_embeddings:
            q = F.normalize(query_features.float(), dim=-1)
            c = F.normalize(candidate_features.float(), dim=-1)
            if self.allow_null:
                c = torch.cat((torch.zeros_like(q[:, None]), c), 1)
            q = q[:, None].expand_as(c)
            scale = math.sqrt(q.shape[-1])
            features = torch.cat((q * scale, c * scale, q * c * q.shape[-1], (q - c).abs() * scale, stats), -1)
        else:
            features = stats
        return self.net(features).squeeze(-1), logits


def expected_context_loss(scores, logits, labels):
    """Enumerated train-only CE; no sampled/vanishing detached reward estimator."""
    if labels.shape != (len(scores),) or logits.shape[:2] != scores.shape:
        raise ValueError("Unpaired oracle targets")
    k = scores.shape[1]
    losses = F.cross_entropy(logits.detach().float().flatten(0, 1), labels.repeat_interleave(k), reduction="none").reshape_as(scores)
    relative = (losses - losses.mean(1, keepdim=True)) / losses.std(1, unbiased=False, keepdim=True).clamp_min(.05)
    return (scores.softmax(-1) * relative).sum(1).mean()
