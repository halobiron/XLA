import torch
from torch import nn

from tacsx.models.selector import TACSSelector, straight_through_select, topk_context_weights
from tacsx.losses.policy import task_aligned_reward, standardize_advantage, reinforce_loss
from tacsx.data.candidate_pool import FixedCandidatePool, PoolItem


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
