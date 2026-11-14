import copy
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_tacs_shortlist as runner
from tacsx.models.tacs import TACSClassifier


class Encoder(nn.Module):
    embedding_dim = 3

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.Linear(3, 3)

    def encode_global(self, images):
        return self.layer(images.mean((-2, -1)))


class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.head = Encoder(), nn.Linear(3, 2)

    def forward(self, query, context=None):
        x = self.backbone.encode_global(query)
        if context is not None:
            x = x + self.backbone.encode_global(context)
        return self.head(x)


def models():
    base = TACSClassifier(None, Task()).eval().requires_grad_(False)
    cfg = {"backbone": {"repo_dir": "unused", "weights": "unused"}}
    with patch.object(runner, "load_model", side_effect=lambda *args: (copy.deepcopy(base), "unused", cfg)), patch.object(runner, "DINOv3Adapter", Encoder):
        control = runner.build_task(None, "no_context", "cpu")
        full = runner.build_task(None, "full_tacs", "cpu")
    return control, full


def test_same_task_initialization_and_trainability_for_both_arms():
    control, full = models()
    assert all(p.requires_grad for p in control.task_model.parameters())
    assert all(p.requires_grad for p in full.task_model.parameters())
    for a, b in zip(control.task_model.parameters(), full.task_model.parameters()):
        assert torch.equal(a, b)
    assert not any(p.requires_grad for p in full.selector.encoder.parameters())
    assert full.selector.projector.weight.requires_grad
    assert torch.equal(full.selector.projector.weight, torch.eye(3))


def test_cached_candidate_features_preserve_scores_and_hybrid_gradient():
    torch.manual_seed(3)
    _, full = models()
    query, candidates = torch.randn(6, 3, 4, 4), torch.randn(6, 4, 3, 4, 4)
    with torch.no_grad():
        features = F.normalize(full.selector.encoder.encode_global(candidates.flatten(0, 1)), dim=-1).reshape(6, 4, 3)
    cached = runner.learned_scores(full, query, features, 1 / .07)
    direct = full.selector(query, candidates) / .07
    assert torch.allclose(cached, direct, atol=1e-5)
    out = full.run_mode("full_tacs", query, candidates, torch.tensor([0, 1, 0, 1, 0, 1]), cached)
    (out.task_loss + out.policy_loss).backward()
    grad = full.selector.projector.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert out.reward is not None and not out.reward.requires_grad
    assert all(p.grad is None for p in full.selector.encoder.parameters())


def test_shared_selector_reuses_the_trainable_task_vit_and_backpropagates_query_scores():
    base = TACSClassifier(None, Task()).eval().requires_grad_(False)
    cfg = {"backbone": {"repo_dir": "unused", "weights": "unused"}}
    with patch.object(runner, "load_model", side_effect=lambda *args: (copy.deepcopy(base), "unused", cfg)), patch.object(runner, "DINOv3Adapter", Encoder):
        full = runner.build_task(None, "full_tacs", "cpu", shared_selector_backbone=True)
    assert full.selector.encoder is full.task_model.backbone
    assert all(p.requires_grad for p in full.selector.encoder.parameters())
    query, candidates = torch.randn(4, 3, 4, 4), torch.randn(4, 3, 3, 4, 4)
    scores = runner.learned_scores(full, query, torch.empty(4, 3, 3), 1 / .07, candidates)
    scores.square().mean().backward()
    grads = [p.grad for p in full.task_model.backbone.parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
