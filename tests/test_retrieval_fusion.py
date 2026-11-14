import pytest
import torch
import torch.nn.functional as F

from tacsx.models.retrieval_fusion import FusionConfig, NativeContextFusion, NativeResidualClassifier, retrieve


def inputs():
    torch.manual_seed(2)
    return torch.randn(8, 6), torch.arange(8), torch.randn(3, 6), torch.randn(3)


@pytest.mark.parametrize("method", ["probability_pool", "feature_pool", "logit_pool", "log_probability_pool", "pseudo_label_pool", "pseudo_log_pool"])
def test_zero_context_weight_preserves_query_classifier(method):
    pool, ids, weight, bias = inputs()
    query = torch.randn(4, 6)
    model = NativeContextFusion(pool, ids, weight, bias, FusionConfig(k=3, alpha=0, method=method))
    result = model(query)
    assert torch.allclose(result["logits"].softmax(-1), F.linear(query, weight, bias).softmax(-1), atol=1e-6)
    assert torch.allclose(result["context_weights"].sum(-1), torch.ones(4))


def test_retrieval_excludes_query_identity():
    pool = torch.eye(4)
    ids = torch.tensor([10, 20, 30, 40])
    _, indices = retrieve(pool[:2], pool, ids, 2, query_ids=ids[:2])
    assert not (ids[indices] == ids[:2, None]).any()


def test_fusion_gradients_are_finite_without_candidate_labels():
    pool, ids, weight, bias = inputs()
    query = torch.randn(4, 6, requires_grad=True)
    model = NativeContextFusion(pool, ids, weight, bias, FusionConfig(k=3, gate="uncertainty"))
    logits = model(query)["logits"]
    F.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_rejects_invalid_fusion_settings():
    with pytest.raises(ValueError):
        FusionConfig(alpha=1.5).validate()
    with pytest.raises(ValueError):
        FusionConfig(retrieval_temperature=0).validate()


def test_residual_context_and_control_start_at_the_same_frozen_baseline():
    pool, ids, weight, bias = inputs()
    q, c = torch.randn(4, 6), torch.randn(4, 6)
    context = NativeResidualClassifier(weight, bias, use_context=True, hidden=8).eval()
    control = NativeResidualClassifier(weight, bias, use_context=False, hidden=8).eval()
    baseline = F.linear(q, weight, bias)
    assert torch.equal(context(q, c), baseline)
    assert torch.equal(control(q, c), baseline)
    assert sum(p.numel() for p in context.parameters()) == sum(p.numel() for p in control.parameters())
    assert not context.weight.requires_grad


def test_query_control_ignores_context_after_an_update():
    pool, ids, weight, bias = inputs()
    q, c = torch.randn(4, 6), torch.randn(4, 6)
    model = NativeResidualClassifier(weight, bias, use_context=False, hidden=8, dropout=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    F.cross_entropy(model(q, c), torch.tensor([0, 1, 2, 0])).backward()
    optimizer.step()
    assert torch.equal(model(q, c), model(q, c * 100))
