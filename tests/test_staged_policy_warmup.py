import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_staged_policy_warmup import policy_weight
from tacsx.losses.policy import reinforce_loss, standardize_advantage


def test_warmup_removes_only_policy_gradient_and_switches_after_epoch1():
    assert [policy_weight(i) for i in range(1, 5)] == [0., 1., 1., 1.]
    with pytest.raises(ValueError):
        policy_weight(0)
    with pytest.raises(ValueError):
        policy_weight(5)
    scores = torch.tensor([[.1, .3], [.2, -.1]], requires_grad=True)
    task = scores.square().mean()
    policy = reinforce_loss(scores.log_softmax(-1)[:, 0], standardize_advantage(torch.tensor([1., -1.])))
    task_grad = torch.autograd.grad(task, scores, retain_graph=True)[0]
    warm_grad = torch.autograd.grad(task + policy_weight(1) * policy, scores, retain_graph=True)[0]
    full_grad = torch.autograd.grad(task + policy_weight(2) * policy, scores)[0]
    assert torch.equal(task_grad, warm_grad)
    assert torch.isfinite(full_grad).all()
    assert not torch.equal(task_grad, full_grad)
