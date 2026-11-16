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


def test_residual_projection_starts_at_identity_and_learns_with_policy_weight():
    base = TACSClassifier(None, Task()).eval().requires_grad_(False)
    cfg = {"backbone": {"repo_dir": "unused", "weights": "unused"}}
    with patch.object(runner, "load_model", side_effect=lambda *args: (copy.deepcopy(base), "unused", cfg)), patch.object(runner, "DINOv3Adapter", Encoder):
        full = runner.build_task(None, "full_tacs", "cpu", lambda_policy=.1, selector_projector="residual_mlp")
    x = torch.randn(5, 3)
    assert torch.equal(full.selector.projector(x), x)
    assert full.lambda_policy == .1
    full.selector.projector(x).square().mean().backward()
    assert full.selector.projector.output.weight.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in full.selector.encoder.parameters())


def test_same_task_initialization_and_trainability_for_both_arms():
    control, full = models()
    assert all(p.requires_grad for p in control.task_model.parameters())
    assert all(p.requires_grad for p in full.task_model.parameters())
    for a, b in zip(control.task_model.parameters(), full.task_model.parameters()):
        assert torch.equal(a, b)
    assert not any(p.requires_grad for p in full.selector.encoder.parameters())
    assert full.selector.projector.weight.requires_grad
    assert torch.equal(full.selector.projector.weight, torch.eye(3))


def test_separate_selector_unfreezes_only_last_two_blocks_and_refreshes_candidates():
    class BlocksEncoder(Encoder):
        def __init__(self,*args,**kwargs):
            super().__init__()
            self.model=nn.Module()
            self.model.blocks=nn.ModuleList([nn.Linear(3,3) for _ in range(4)])
            self.model.norm=nn.LayerNorm(3)

        def encode_global(self,x):
            x=super().encode_global(x)
            for block in self.model.blocks:
                x=block(x)
            return self.model.norm(x)

    base=TACSClassifier(None,Task()).eval().requires_grad_(False)
    cfg={"backbone":{"repo_dir":"unused","weights":"unused"}}
    with patch.object(runner,"load_model",side_effect=lambda *args:(copy.deepcopy(base),"unused",cfg)),patch.object(runner,"DINOv3Adapter",BlocksEncoder):
        full=runner.build_task(None,"full_tacs","cpu",selector_unfreeze_last_blocks=2)
    enc=full.selector.encoder
    assert enc is not full.task_model.backbone
    assert not any(p.requires_grad for block in enc.model.blocks[:2] for p in block.parameters())
    assert all(p.requires_grad for block in enc.model.blocks[2:] for p in block.parameters())
    q,c=torch.randn(3,3,4,4),torch.randn(3,4,3,4,4)
    a=runner.learned_scores(full,q,torch.zeros(3,4,3),1.,c)
    b=runner.learned_scores(full,q,torch.ones(3,4,3)*999,1.,c)
    assert torch.equal(a,b)  # stale cache is not used for a trainable selector
    a.square().sum().backward()
    assert enc.model.blocks[-1].weight.grad.abs().sum()>0
    assert enc.model.blocks[0].weight.grad is None


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
