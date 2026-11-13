import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tacsx.models.baselines import DINOEmbeddingCache
from tacsx.models.dinov3_adapter import DINOv3Adapter
from tacsx.models.selector import TACSSelector
from tacsx.models.tacs import TACSClassifier


class DummyEncoder(nn.Module):
    embedding_dim = 3

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def encode_global(self, images):
        return self.linear(images.mean((-2, -1)))


class DummyTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, query, context=None):
        x = query.mean((-2, -1))
        if context is not None:
            x = x + context.mean((-2, -1))
        return self.linear(x)


def inputs():
    torch.manual_seed(12)
    return torch.randn(4, 3, 4, 4), torch.randn(4, 4, 3, 4, 4), torch.tensor([0, 1, 0, 1])


@pytest.mark.parametrize("mode", ["gumbel_only", "policy_only", "full_tacs"])
def test_eval_uses_argmax_without_sampling(mode):
    model = TACSClassifier(TACSSelector(DummyEncoder()), DummyTask()).eval()
    q, c, y = inputs()
    scores = torch.tensor([[0., 1., 2., 3.]]).repeat(len(q), 1)
    with patch("torch.nn.functional.gumbel_softmax", side_effect=AssertionError("eval sampled")), \
         patch("torch.distributions.Categorical.sample", side_effect=AssertionError("eval sampled")):
        first = model.run_mode(mode, q, c, y, scores)
        torch.manual_seed(999)
        second = model.run_mode(mode, q, c, y, scores)
    assert torch.equal(first.action, torch.full((len(q),), 3))
    assert torch.equal(first.context, c[:, 3])
    assert torch.equal(first.logits, second.logits)


def test_full_task_reward_and_policy_use_the_same_action():
    model = TACSClassifier(TACSSelector(DummyEncoder()), DummyTask()).train()
    q, c, y = inputs()
    with patch.object(model, "policy_terms", wraps=model.policy_terms) as policy:
        out = model.run_mode("full_tacs", q, c, y)
    assert torch.equal(policy.call_args.kwargs["action"], out.action)
    ce = torch.nn.functional.cross_entropy
    expected = ce(model.task_model(q), y, reduction="none") - ce(out.logits, y, reduction="none")
    assert torch.allclose(out.reward, expected.detach())
    assert not out.reward.requires_grad
    # Policy reward must not create a second gradient route into the task model.
    task_grads = torch.autograd.grad(out.policy_loss, tuple(model.task_model.parameters()),
                                     allow_unused=True, retain_graph=True)
    assert all(grad is None for grad in task_grads)
    (out.task_loss + out.policy_loss).backward()
    assert model.selector.projector[0].weight.grad is not None
    assert torch.isfinite(model.selector.projector[0].weight.grad).all()


@pytest.mark.parametrize("threshold,expected_k", [(0.5, 2), (0.7, 3), (0.9, 4), (1., 4)])
def test_adaptive_topk_includes_threshold_crossing(threshold, expected_k):
    model = TACSClassifier(TACSSelector(DummyEncoder()), DummyTask(), adaptive_threshold=threshold)
    q, c, y = inputs()
    scores = torch.zeros(len(q), 4)
    out = model.run_mode("adaptive_topk_tacs", q, c, y, scores)
    assert torch.equal((out.topk_weights > 0).sum(-1), torch.full((len(q),), expected_k))
    assert torch.allclose(out.topk_weights.sum(-1), torch.ones(len(q)))


class IndexRope(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, H, W):
        self.calls += 1
        values = torch.arange(H * W).float().unsqueeze(-1)
        return values, values


class IdentityBlock(nn.Module):
    def forward(self, tokens, rope):
        return tokens


def fake_adapter(layout):
    # No official model construction or downloads in tests.
    adapter = DINOv3Adapter.__new__(DINOv3Adapter)
    nn.Module.__init__(adapter)
    adapter.pair_rope_layout = layout
    adapter.model = nn.Module()
    adapter.model.n_storage_tokens = 0
    adapter.model.rope_embed = IndexRope()
    adapter.model.blocks = nn.ModuleList([IdentityBlock(), IdentityBlock()])
    adapter.model.norm = nn.Identity()
    return adapter


def test_horizontal_rope_matches_image_major_patch_order():
    adapter = fake_adapter("horizontal")
    sin, _ = adapter._rope_for_tokens((2, 2), 2)
    # Grid rows are [q00 q01 c00 c01], [q10 q11 c10 c11],
    # but the token sequence is [q00 q01 q10 q11 c00 c01 c10 c11].
    assert sin.flatten().tolist() == [0, 1, 4, 5, 2, 3, 6, 7]


def test_legacy_and_per_image_rope_are_explicit_ablations():
    legacy, per_image = fake_adapter("legacy"), fake_adapter("per_image")
    assert legacy._rope_for_tokens((2, 2), 2)[0].flatten().tolist() == list(range(8))
    assert per_image._rope_for_tokens((2, 2), 2)[0].flatten().tolist() == [0, 1, 2, 3] * 2


def test_adapter_regenerates_positions_per_block_and_checks_token_count():
    adapter = fake_adapter("horizontal")
    x = torch.randn(1, 9, 3)
    assert torch.equal(adapter.forward_blocks(x, (2, 2), num_images=2), x)
    assert adapter.model.rope_embed.calls == len(adapter.model.blocks)
    with pytest.raises(ValueError, match="expected 9 tokens"):
        adapter.forward_blocks(x[:, :8], (2, 2), num_images=2)


def test_dino_retrieval_stays_fixed_when_task_backbone_changes():
    task = DummyEncoder()
    retriever = copy.deepcopy(task).eval().requires_grad_(False)
    q, _, _ = inputs()
    ids = torch.tensor([10, 11])
    cache = DINOEmbeddingCache(ids, torch.nn.functional.normalize(torch.randn(2, 3), dim=-1), retriever)
    chosen_ids = ids.repeat(len(q), 1)
    before = cache.scores(task, q, chosen_ids)
    with torch.no_grad():
        task.linear.weight.add_(10)
    assert torch.equal(before, cache.scores(task, q, chosen_ids))
    assert not before.requires_grad


def test_trainable_backbone_keeps_parameterless_training_augmentation():
    from tacsx.engine import _set_training_mode
    trainable = fake_adapter("horizontal")
    trainable.model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(3, 3), nn.Dropout())])
    frozen = fake_adapter("horizontal")
    model = nn.ModuleList([trainable, frozen])
    _set_training_mode(model)
    assert trainable.model.rope_embed.training
    assert trainable.model.blocks[0][1].training
    assert not frozen.training
    assert not frozen.model.rope_embed.training


def test_task_head_initialization_matches_across_baseline_and_learned_modes(monkeypatch):
    from tacsx import cli
    monkeypatch.setattr(cli, "DINOv3Adapter", lambda *args, **kwargs: DummyEncoder())
    cfg = {"backbone": {"weights": "dummy", "repo_dir": "dummy", "freeze": True},
           "selector": {}, "policy": {}}
    torch.manual_seed(7)
    baseline = cli.build_model(cfg, False, include_selector=False)
    torch.manual_seed(7)
    learned = cli.build_model(cfg, False, include_selector=True)
    assert torch.equal(baseline.task_model.head.weight, learned.task_model.head.weight)
    assert torch.equal(baseline.task_model.head.bias, learned.task_model.head.bias)


def test_data_order_is_independent_of_model_rng_and_context_mode(monkeypatch):
    from tacsx.data import cifar100

    class FakeCIFAR:
        def __init__(self, *args, **kwargs):
            self.targets = list(range(20))

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.full((3, 4, 4), float(index)), self.targets[index]

    monkeypatch.setattr(cifar100.datasets, "CIFAR100", FakeCIFAR)
    def order(include_context):
        train, _, _, _ = cifar100.make_cifar100_loaders(
            "dummy", 4, .2, 2, 4, 0, 42, include_context=include_context)
        return torch.cat([batch["query_id"] for batch in train])
    first = order(False)
    torch.randn(1000)  # Represents extra backbone construction or policy sampling.
    assert torch.equal(first, order(True))
