from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from tacsx.models.selector import aggregate_topk_context, straight_through_select
from tacsx.losses.policy import task_aligned_reward, standardize_advantage, reinforce_loss


@dataclass
class TACSOutput:
    logits: torch.Tensor
    task_loss: torch.Tensor
    policy_loss: torch.Tensor
    reward: torch.Tensor | None
    scores: torch.Tensor
    action: torch.Tensor | None = None
    context: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    topk_weights: torch.Tensor | None = None
    # TACS-X anchored extension (all None for the paper configuration).
    aux_loss: torch.Tensor | None = None        # already multiplied by aux_query_weight
    gate_loss: torch.Tensor | None = None       # abstain-gate loss (gate parameters only)
    null_selected: torch.Tensor | None = None   # [B] bool, True = deployed prediction is query-only


ANCHORED_MODES = {"gumbel_only", "policy_only", "full_tacs"}


def gate_features(base_logits, ctx_logits, scores):
    """Label-free, detached per-query features for the abstain gate ([B, 11]).

    Query-only confidence, context-path confidence, their agreement, and
    retrieval-score shape. Nothing here uses labels, so it is valid at inference.
    """
    pb = base_logits.detach().float().softmax(-1)
    pc = ctx_logits.detach().float().softmax(-1)

    def confidence(p):
        top = p.topk(2, dim=-1).values
        return [top[:, 0], top[:, 0] - top[:, 1], -(p * p.clamp_min(1e-12).log()).sum(-1)]

    bi, ci = pb.argmax(-1), pc.argmax(-1)
    s = scores.detach().float()
    top = s.topk(min(2, s.shape[-1]), dim=-1).values
    best = torch.where(torch.isfinite(top[:, 0]), top[:, 0], torch.zeros_like(top[:, 0]))
    gap = top[:, 0] - top[:, -1]
    gap = torch.where(torch.isfinite(gap), gap, torch.zeros_like(gap))
    features = confidence(pb) + confidence(pc) + [
        (bi == ci).float(), pb.gather(1, ci[:, None])[:, 0], pc.gather(1, bi[:, None])[:, 0], best, gap]
    return torch.stack(features, dim=-1)


class AbstainGate(nn.Module):
    """Predicts whether the selected context will lower the loss for this query."""

    n_features = 11

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.norm = nn.BatchNorm1d(self.n_features, affine=False)
        self.net = nn.Sequential(nn.Linear(self.n_features, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(self.norm(features.float())).squeeze(-1).float()


def gate_loss_from_reward(gate_logit, reward):
    """|reward|-weighted BCE on sign(reward): mistakes cost what they lose in CE."""
    weight = reward.detach().abs().float()
    target = (reward.detach() > 0).float()
    bce = F.binary_cross_entropy_with_logits(gate_logit.float(), target, reduction="none")
    return (weight * bce).sum() / (weight.sum() + 1e-8)


class TACSClassifier(nn.Module):
    """Orchestrates selector + downstream model.

    TACS-X anchored extension (explicit deviation, off by default):

    * aux_query_weight (beta > 0): the query-only forward that defines the
      reward baseline is trained too, L = CE(q,c) + beta*CE(q,0) + lambda*L_policy.
      Without it the baseline drifts out of distribution and inflates the reward.
    * abstain_gate: a small gate decides PER QUERY, at inference, whether to
      use the selected context or the query-only prediction. It sees detached
      label-free features (gate_features) and is trained only by
      gate_loss_from_reward on the reward already computed for every query.
      Selection/training is unchanged: context is always sampled and trained,
      so the gate cannot starve the context path or the policy. Replaces a
      global null-offset option that collapsed to all-null in GPU smoke tests.
    """

    def __init__(self, selector, task_model, tau=0.1, lambda_policy=1.0, topk_k=2, adaptive_threshold=0.9,
                 eval_selection="argmax", aux_query_weight=0.0, abstain_gate=False):
        super().__init__()
        self.selector = selector
        self.task_model = task_model
        self.tau = tau
        self.lambda_policy = lambda_policy
        self.topk_k = topk_k
        self.adaptive_threshold = adaptive_threshold
        if not 0 < adaptive_threshold <= 1:
            raise ValueError("adaptive_threshold must be in (0, 1]")
        if eval_selection not in {"argmax", "sampled"}:
            raise ValueError("eval_selection must be argmax or sampled")
        self.eval_selection = eval_selection
        if aux_query_weight < 0:
            raise ValueError("aux_query_weight must be non-negative")
        self.aux_query_weight = float(aux_query_weight)
        self.gate = AbstainGate() if abstain_gate else None

    @property
    def abstain_gate(self) -> bool:
        return self.gate is not None

    @property
    def anchored(self) -> bool:
        return self.abstain_gate or self.aux_query_weight > 0

    # ------------------------------------------------------------------ anchored helpers
    def _query_only(self, query, labels, with_grad):
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):
            logits = self.task_model(query, None)
            loss = F.cross_entropy(logits.float(), labels, reduction="none")
        return logits, loss

    def _deploy(self, base_logits, ctx_logits, scores):
        """Returns (gate_logit or None, use_query_only[B] bool)."""
        if self.gate is None:
            return None, torch.zeros(base_logits.shape[0], dtype=torch.bool, device=base_logits.device)
        logit = self.gate(gate_features(base_logits, ctx_logits, scores))
        return logit, logit.detach() < 0  # logit 0 at init -> context, as in paper TACS

    def anchored_step(self, query, labels, scores, gather, use_policy=True, sample_policy=False):
        """Training forward for selector modes with the anchored extension.

        gather(mask[B,N]) -> context images [B,C,H,W] selects pixels from a
        (straight-through) one-hot over the N candidates, so the same code serves
        sampled candidate tensors and a cached full pool. Selection is exactly
        the paper path (ST Gumbel / categorical over real candidates).
        """
        n = scores.shape[-1]
        base_logits, base_loss = self._query_only(query, labels, with_grad=self.aux_query_weight > 0)
        if sample_policy:
            with torch.no_grad():
                action = Categorical(logits=scores.float()).sample()
            ctx_mask = F.one_hot(action, n).to(scores.dtype)
        else:
            ctx_mask = F.gumbel_softmax(scores.float(), tau=self.tau, hard=True, dim=-1)
            action = ctx_mask.argmax(-1)
        context = gather(ctx_mask)
        ctx_logits = self.task_model(query, context)
        item_loss = F.cross_entropy(ctx_logits.float(), labels, reduction="none")
        reward = task_aligned_reward(base_loss, item_loss)
        if use_policy:
            p_loss = reinforce_loss(Categorical(logits=scores.float()).log_prob(action), standardize_advantage(reward))
        else:
            p_loss = item_loss.new_zeros(())
        gate_logit, query_only = self._deploy(base_logits, ctx_logits, scores)
        logits = torch.where(query_only[:, None], base_logits.detach(), ctx_logits.detach())
        return TACSOutput(logits=logits, task_loss=item_loss.mean(), policy_loss=p_loss,
                          reward=reward if use_policy else None, scores=scores, action=action,
                          context=context.detach(),
                          aux_loss=self.aux_query_weight * base_loss.mean() if self.aux_query_weight > 0 else None,
                          gate_loss=gate_loss_from_reward(gate_logit, reward) if gate_logit is not None else None,
                          null_selected=query_only if self.abstain_gate else None)

    def anchored_argmax(self, query, labels, scores, take):
        """Inference: argmax candidate, then the gate may fall back to query-only."""
        action = scores.argmax(-1)
        context = take(action)
        ctx_logits = self.task_model(query, context)
        base_logits, base_loss = self._query_only(query, labels, with_grad=False)
        _, query_only = self._deploy(base_logits, ctx_logits, scores)
        logits = torch.where(query_only[:, None], base_logits, ctx_logits)
        losses = F.cross_entropy(logits.float(), labels, reduction="none")
        reward = torch.where(query_only, torch.zeros_like(losses), task_aligned_reward(base_loss, losses))
        return TACSOutput(logits, losses.mean(), losses.new_zeros(()), reward, scores, action=action,
                          context=context, null_selected=query_only if self.abstain_gate else None)

    def differentiable_path(self, query, candidates, labels, scores=None):
        if scores is None:
            scores = self.selector(query, candidates)
        context, mask = straight_through_select(scores, candidates, self.tau)
        logits = self.task_model(query, context)
        loss = F.cross_entropy(logits, labels, reduction="none")
        return scores, mask, logits, loss

    def policy_terms(self, query, candidates, labels, scores, action=None, context_loss=None):
        dist = Categorical(logits=scores)
        if action is None:
            action = dist.sample()
        b = query.shape[0]
        chosen = candidates[torch.arange(b, device=query.device), action]

        with torch.no_grad():
            logits_base = self.task_model(query, None)
            loss_base = F.cross_entropy(logits_base, labels, reduction="none")

        if context_loss is None:
            with torch.no_grad():
                logits_ctx = self.task_model(query, chosen)
                context_loss = F.cross_entropy(logits_ctx, labels, reduction="none")

        reward = task_aligned_reward(loss_base, context_loss)
        advantage = standardize_advantage(reward)
        p_loss = reinforce_loss(dist.log_prob(action), advantage)
        return p_loss, reward, action

    def forward(self, query, candidates, labels, scores=None):
        if not self.training and self.eval_selection == "argmax":
            return self.run_mode("full_tacs", query, candidates, labels, scores)
        if self.anchored:
            return self.run_mode("full_tacs", query, candidates, labels, scores)
        scores, mask, logits, item_loss = self.differentiable_path(query, candidates, labels, scores)
        context = (mask[..., None, None, None] * candidates).sum(dim=1)
        action = mask.argmax(dim=-1)
        p_loss, reward, _ = self.policy_terms(query, candidates, labels, scores,
                                            action=action, context_loss=item_loss.detach())
        return TACSOutput(
            logits=logits,
            task_loss=item_loss.mean(),
            policy_loss=p_loss,
            reward=reward,
            scores=scores,
            action=action,
            context=context,
        )

    def run_mode(self, mode, query, candidates, labels, scores=None, selected_context=None, selected_action=None):
        """One shared execution surface for every required experiment mode."""
        if mode == "no_context":
            logits = self.task_model(query, None)
            loss = F.cross_entropy(logits, labels)
            return TACSOutput(logits, loss, loss.new_zeros(()), None, loss.new_empty((query.shape[0], 0)))
        if scores is None:
            scores = self.selector(query, candidates)
        if self.anchored and mode in ANCHORED_MODES:
            rows = torch.arange(query.shape[0], device=query.device)
            if not self.training and self.eval_selection == "argmax":
                return self.anchored_argmax(query, labels, scores, lambda index: candidates[rows, index])
            gather = lambda m: torch.einsum("bn,bnchw->bchw", m.to(candidates.dtype), candidates)
            return self.anchored_step(query, labels, scores, gather,
                                      use_policy=mode != "gumbel_only", sample_policy=mode == "policy_only")
        if not self.training and self.eval_selection == "argmax" and mode in {"gumbel_only", "policy_only", "full_tacs"}:
            # Paper inference uses argmax, not a fresh Gumbel/Categorical draw.
            action = scores.argmax(dim=-1)
            chosen = candidates[torch.arange(query.shape[0], device=query.device), action]
            logits = self.task_model(query, chosen)
            losses = F.cross_entropy(logits, labels, reduction="none")
            reward = None
            if mode in {"policy_only", "full_tacs"}:
                with torch.no_grad():
                    base = F.cross_entropy(self.task_model(query, None), labels, reduction="none")
                reward = task_aligned_reward(base, losses)
            return TACSOutput(logits, losses.mean(), losses.new_zeros(()), reward, scores,
                              action=action, context=chosen)
        if mode == "gumbel_only":
            selected_scores, mask, logits, losses = self.differentiable_path(query, candidates, labels, scores)
            context = (mask[..., None, None, None] * candidates).sum(dim=1)
            return TACSOutput(
                logits, losses.mean(), losses.new_zeros(()), None, selected_scores,
                action=mask.argmax(dim=-1), context=context,
            )
        if mode == "policy_only":
            dist = Categorical(logits=scores)
            action = dist.sample()
            chosen = candidates[torch.arange(query.shape[0], device=query.device), action]
            logits = self.task_model(query, chosen)
            losses = F.cross_entropy(logits, labels, reduction="none")
            policy_loss, reward, _ = self.policy_terms(query, candidates, labels, scores,
                                                       action=action, context_loss=losses.detach())
            return TACSOutput(logits, losses.mean(), policy_loss, reward, scores, action, chosen)
        if mode == "full_tacs":
            return self(query, candidates, labels, scores)
        if mode in {"random_context", "dino_similarity"}:
            action = scores.argmax(dim=-1) if selected_action is None else selected_action
            chosen = selected_context if selected_context is not None else candidates[torch.arange(query.shape[0], device=query.device), action]
            logits = self.task_model(query, chosen)
            task_loss = F.cross_entropy(logits, labels)
            return TACSOutput(logits, task_loss, task_loss.new_zeros(()), None, scores, action, chosen)
        if mode == "topk_tacs":
            chosen, indices, weights = aggregate_topk_context(scores, candidates, k=min(self.topk_k, candidates.shape[1]))
            logits = self.task_model(query, chosen)
            task_loss = F.cross_entropy(logits, labels)
            return TACSOutput(logits, task_loss, task_loss.new_zeros(()), None, scores,
                              context=chosen, topk_indices=indices, topk_weights=weights)
        if mode == "adaptive_topk_tacs":
            # The batch uses max selected K; zeroed weights retain an individual adaptive K.
            probs = scores.softmax(dim=-1)
            sorted_p, order = probs.sort(dim=-1, descending=True)
            # Include the candidate that FIRST reaches the requested coverage.
            keep = (sorted_p.cumsum(-1) - sorted_p) < self.adaptive_threshold
            weights = sorted_p * keep
            weights = weights / weights.sum(-1, keepdim=True)
            batch = torch.arange(candidates.shape[0], device=candidates.device)[:, None]
            chosen = (candidates[batch, order] * weights[..., None, None, None]).sum(1)
            logits = self.task_model(query, chosen)
            task_loss = F.cross_entropy(logits, labels)
            return TACSOutput(logits, task_loss, task_loss.new_zeros(()), None, scores,
                              context=chosen, topk_indices=order, topk_weights=weights)
        raise ValueError(f"unknown mode: {mode}")
