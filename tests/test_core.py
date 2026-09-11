import torch
from torch import nn

from tacsx.models.selector import TACSSelector, straight_through_select, topk_context_weights
from tacsx.losses.policy import task_aligned_reward, standardize_advantage, reinforce_loss
from tacsx.data.candidate_pool import FixedCandidatePool, PoolItem
from tacsx.models.dinov3_adapter import TokenBatch
from tacsx.models.dual_input_vit import DualInputViT
from tacsx.models.tacs import TACSClassifier


class DummyEncoder(nn.Module):
    embedding_dim = 8
    def encode_global(self, images):
        x = images.mean(dim=(-2, -1))
        reps = x.repeat(1, (8 + x.shape[1] - 1) // x.shape[1])
        return reps[:, :8]


def test_selector_shape():
    selector = TACSSelector(DummyEncoder(), projection=False)
    q = torch.randn(3, 3, 16, 16)
    c = torch.randn(3, 5, 3, 16, 16)
    assert selector(q, c).shape == (3, 5)


def test_gumbel_hard_selection_is_one_hot():
    scores = torch.randn(4, 6, requires_grad=True)
    candidates = torch.randn(4, 6, 3, 8, 8)
    selected, mask = straight_through_select(scores, candidates, tau=0.1)
    assert selected.shape == (4, 3, 8, 8)
    assert torch.allclose(mask.sum(dim=-1), torch.ones(4))
    assert torch.all((mask == 0) | (mask == 1))


def test_reward_sign():
    r = task_aligned_reward(torch.tensor([1.0, 0.5]), torch.tensor([0.6, 0.8]))
    assert r[0] > 0
    assert r[1] < 0


def test_policy_loss_is_finite():
    log_prob = torch.tensor([-0.2, -1.1], requires_grad=True)
    adv = standardize_advantage(torch.tensor([1.0, -1.0]))
    loss = reinforce_loss(log_prob, adv)
    assert torch.isfinite(loss)
    loss.backward()
    assert log_prob.grad is not None


def test_topk_weights_sum_to_one():
    scores = torch.randn(4, 8)
    idx, w = topk_context_weights(scores, k=3)
    assert idx.shape == (4, 3)
    assert w.shape == (4, 3)
    assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_candidate_pool_excludes_query():
    pool = FixedCandidatePool([PoolItem(i, i % 2) for i in range(10)])
    out = pool.sample_indices(query_index=3, n=5)
    assert 3 not in out
    assert len(out) == 5


class DummyTokenAdapter(nn.Module):
    embedding_dim = 8

    def prepare_tokens(self, images):
        b = images.shape[0]
        patches = images.mean(dim=1).reshape(b, 4, 4).repeat(1, 1, 2)
        return TokenBatch(torch.cat((torch.zeros(b, 1, 8), patches), dim=1), (2, 2))

    def forward_blocks(self, tokens, grid_size):
        assert tokens.shape[1] == 1 + grid_size[0] * grid_size[1]
        return tokens


def test_dual_input_accepts_none_and_context():
    model = DualInputViT(DummyTokenAdapter(), num_classes=3)
    q, c = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    assert model(q).shape == (2, 3)
    assert model(q, c).shape == (2, 3)


class DummyTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, query, context=None):
        x = query.mean(dim=(-2, -1))
        if context is not None:
            x = x + context.mean(dim=(-2, -1))
        return self.linear(x)


def test_all_required_model_modes_smoke():
    model = TACSClassifier(TACSSelector(DummyEncoder(), projection=False), DummyTask())
    q, c, labels = torch.randn(3, 3, 8, 8), torch.randn(3, 4, 3, 8, 8), torch.tensor([0, 1, 0])
    scores = model.selector(q, c)
    for mode in ("no_context", "random_context", "dino_similarity", "gumbel_only", "policy_only", "full_tacs", "topk_tacs", "adaptive_topk_tacs"):
        out = model.run_mode(mode, q, c, labels, None if mode == "no_context" else scores)
        assert torch.isfinite(out.task_loss + out.policy_loss)
