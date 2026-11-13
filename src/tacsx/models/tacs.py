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


class TACSClassifier(nn.Module):
    """Orchestrates selector + downstream model."""

    def __init__(self, selector, task_model, tau=0.1, lambda_policy=1.0, topk_k=2, adaptive_threshold=0.9,
                 eval_selection="argmax"):
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
