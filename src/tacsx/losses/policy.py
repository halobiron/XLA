from __future__ import annotations
import torch


def task_aligned_reward(loss_without_context, loss_with_context):
    """Positive iff selected context decreases task loss."""
    return loss_without_context.detach() - loss_with_context.detach()


def standardize_advantage(reward: torch.Tensor, eps: float = 1e-6):
    reward = reward.reshape(-1)
    return (reward - reward.mean()) / (reward.std(unbiased=False) + eps)


def reinforce_loss(log_prob: torch.Tensor, advantage: torch.Tensor):
    return -(log_prob * advantage.detach()).mean()
