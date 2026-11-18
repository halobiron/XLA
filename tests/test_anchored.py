"""TACS-X anchored extension: trained query-only CE and a per-query abstain gate."""
import torch
import torch.nn.functional as F
from torch import nn

from tacsx.engine import evaluate
from tacsx.models.selector import TACSSelector
from tacsx.models.tacs import TACSClassifier, gate_features, gate_loss_from_reward


class DummyEncoder(nn.Module):
    embedding_dim = 3

    def encode_global(self, images):
        return images.mean(dim=(-2, -1))


class DummyTask(nn.Module):
    """Separate parameters for the query-only and the context path."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(3, 2)
        self.context = nn.Linear(3, 2)

    def forward(self, query, context=None):
        out = self.query(query.mean(dim=(-2, -1)))
        if context is not None:
            out = out + self.context(context.mean(dim=(-2, -1)))
        return out


def _model(**kwargs):
    torch.manual_seed(0)
    return TACSClassifier(TACSSelector(DummyEncoder(), projection=False), DummyTask(), **kwargs)


def _batch():
    torch.manual_seed(1)
    return torch.randn(6, 3, 4, 4), torch.randn(6, 5, 3, 4, 4), torch.tensor([0, 1, 0, 1, 1, 0])


def _force_query_only(model):
    with torch.no_grad():
        model.gate.net[-1].bias.fill_(-10.0)


def test_default_configuration_is_unchanged_paper_path():
    model = _model().train()
    assert not model.anchored and model.gate is None
    q, c, y = _batch()
    out = model.run_mode("full_tacs", q, c, y)
    assert out.aux_loss is None and out.gate_loss is None and out.null_selected is None


def test_aux_only_keeps_paper_selection_and_loss():
    q, c, y = _batch()
    paper, aux = _model().train(), _model(aux_query_weight=1.0).train()
    torch.manual_seed(7); a = paper.run_mode("full_tacs", q, c, y)
    torch.manual_seed(7); b = aux.run_mode("full_tacs", q, c, y)
    assert torch.equal(a.action, b.action)
    assert torch.allclose(a.task_loss, b.task_loss) and torch.allclose(a.reward, b.reward)
    assert torch.allclose(a.policy_loss, b.policy_loss)


def test_aux_query_loss_trains_query_only_path_and_reward_stays_detached():
    model = _model(aux_query_weight=0.5).train()
    q, c, y = _batch()
    out = model.run_mode("full_tacs", q, c, y)
    assert torch.allclose(out.aux_loss, 0.5 * F.cross_entropy(model.task_model(q, None), y))
    assert not out.reward.requires_grad
    model.zero_grad()
    out.aux_loss.backward()
    assert model.task_model.query.weight.grad.abs().sum() > 0
    assert model.task_model.context.weight.grad is None


def test_gate_features_are_label_free_detached_and_finite():
    base = torch.randn(4, 5, requires_grad=True)
    ctx = torch.randn(4, 5, requires_grad=True)
    scores = torch.tensor([[1., 2., -float("inf")], [0., 0., 0.], [3., -float("inf"), -float("inf")], [1., 1., 2.]])
    f = gate_features(base, ctx, scores)
    assert f.shape == (4, 11) and torch.all(torch.isfinite(f)) and not f.requires_grad


def test_gate_loss_trains_only_the_gate():
    model = _model(aux_query_weight=1.0, abstain_gate=True).train()
    q, c, y = _batch()
    out = model.run_mode("full_tacs", q, c, y)
    model.zero_grad()
    out.gate_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gate.parameters())
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("gate."))


def test_gate_loss_weights_by_reward_magnitude():
    logit = torch.zeros(3, requires_grad=True)
    loss = gate_loss_from_reward(logit, torch.tensor([2.0, -1.0, 0.0]))
    loss.backward()
    assert logit.grad[0] < 0 < logit.grad[1]          # push toward context / query-only
    assert logit.grad[2] == 0                          # zero reward: no preference
    assert abs(logit.grad[0]) > abs(logit.grad[1])


def test_gate_at_init_uses_context_like_paper_tacs():
    q, c, y = _batch()
    gated, plain = _model(abstain_gate=True).eval(), _model().eval()
    a, b = gated.run_mode("full_tacs", q, c, y), plain.run_mode("full_tacs", q, c, y)
    assert not torch.any(a.null_selected)
    assert torch.equal(a.action, b.action) and torch.allclose(a.logits, b.logits)


def test_gate_fallback_predicts_query_only_and_context_still_trains():
    q, c, y = _batch()
    model = _model(abstain_gate=True)
    _force_query_only(model)
    model.eval()
    out = model.run_mode("full_tacs", q, c, y)
    assert torch.all(out.null_selected) and torch.all(out.reward == 0)
    assert torch.allclose(out.logits, model.task_model(q, None), atol=1e-6)
    model.train()
    out = model.run_mode("full_tacs", q, c, y)
    model.zero_grad()
    (out.task_loss + out.policy_loss).backward()
    assert model.task_model.context.weight.grad.abs().sum() > 0  # no starvation


def test_policy_only_and_gumbel_only_with_gate():
    q, c, y = _batch()
    model = _model(abstain_gate=True, aux_query_weight=1.0).train()
    for mode in ("policy_only", "gumbel_only"):
        out = model.run_mode(mode, q, c, y)
        assert torch.isfinite(out.task_loss + out.policy_loss + out.aux_loss + out.gate_loss)
        if mode == "gumbel_only":
            assert out.reward is None and out.policy_loss == 0


def test_pool_gather_matches_candidate_tensor_path():
    """Full-pool scripts pass a GEMM gather; it must equal per-query indexing."""
    torch.manual_seed(4)
    pool = torch.randn(7, 3, 4, 4)
    q, y = torch.randn(5, 3, 4, 4), torch.tensor([0, 1, 1, 0, 1])
    model = _model(abstain_gate=True, aux_query_weight=1.0).train()
    scores = torch.randn(5, 7)
    gather = lambda m: (m @ pool.flatten(1)).reshape(len(m), *pool.shape[1:])
    torch.manual_seed(5)
    a = model.anchored_step(q, y, scores, gather)
    torch.manual_seed(5)
    b = model.run_mode("full_tacs", q, pool.expand(5, -1, -1, -1, -1), y, scores)
    assert torch.equal(a.action, b.action) and torch.allclose(a.task_loss, b.task_loss, atol=1e-6)


def test_evaluator_records_query_only_rows():
    model = _model(abstain_gate=True)
    _force_query_only(model)
    batch = {"query": torch.randn(2, 3, 4, 4), "label": torch.tensor([0, 1]),
             "query_id": torch.tensor([7, 8]), "candidates": torch.randn(2, 4, 3, 4, 4),
             "candidate_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 9]]),
             "candidate_labels": torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])}

    class OneBatchLoader(list):
        dataset = None

    metrics, pairs = evaluate(model, OneBatchLoader([batch]), "full_tacs", torch.device("cpu"), collect_pairs=True)
    assert metrics["null_rate"] == 1.0
    assert all(p["null_context"] and p["candidate_id"] == -1 for p in pairs)


def test_param_groups_give_gate_its_own_group():
    from tacsx.cli import param_groups
    model = _model(abstain_gate=True)
    groups = {g["name"]: g for g in param_groups(model, {"lr_head": 3e-4})}
    assert {id(p) for p in groups["gate"]["params"]} == {id(p) for p in model.gate.parameters()}
    assert groups["gate"]["lr"] == 1e-3 and groups["gate"]["weight_decay"] == 0.0
    assert "gate" not in {g["name"] for g in param_groups(_model(), {"lr_head": 3e-4})}
