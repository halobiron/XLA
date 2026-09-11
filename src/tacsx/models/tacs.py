from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from tacsx.models.selector import straight_through_select
from tacsx.losses.policy import task_aligned_reward, standardize_advantage, reinforce_loss


@dataclass
class TACSOutput:
    logits: torch.Tensor
    task_loss: torch.Tensor
    policy_loss: torch.Tensor
    reward: torch.Tensor | None
    scores: torch.Tensor


class TACSClassifier(nn.Module):
    """Orchestrates selector + downstream model."""

    def __init__(self, selector, task_model, tau=0.1, lambda_policy=1.0):
        super().__init__()
        self.selector = selector
        self.task_model = task_model
        self.tau = tau
        self.lambda_policy = lambda_policy

    def differentiable_path(self, query, candidates, labels):
        scores = self.selector(query, candidates)
        context, mask = straight_through_select(scores, candidates, self.tau)
        logits = self.task_model(query, context)
        loss = F.cross_entropy(logits, labels, reduction="none")
        return scores, mask, logits, loss

    def policy_terms(self, query, candidates, labels, scores):
        dist = Categorical(logits=scores)
        action = dist.sample()
        b = query.shape[0]
        chosen = candidates[torch.arange(b, device=query.device), action]

        with torch.no_grad():
            logits_base = self.task_model(query, None)
            loss_base = F.cross_entropy(logits_base, labels, reduction="none")

        logits_ctx = self.task_model(query, chosen)
        loss_ctx = F.cross_entropy(logits_ctx, labels, reduction="none")

        reward = task_aligned_reward(loss_base, loss_ctx)
        advantage = standardize_advantage(reward)
        p_loss = reinforce_loss(dist.log_prob(action), advantage)
        return p_loss, reward, action

    def forward(self, query, candidates, labels):
        scores, _, logits, item_loss = self.differentiable_path(query, candidates, labels)
        p_loss, reward, _ = self.policy_terms(query, candidates, labels, scores)
        return TACSOutput(
            logits=logits,
            task_loss=item_loss.mean(),
            policy_loss=p_loss,
            reward=reward,
            scores=scores,
        )
