import torch

from tacsx.models.utility_selector import UtilitySelector, expected_context_loss, prediction_features


def test_label_free_features_and_null_action_preserve_query_logits():
    p, q, sim = torch.randn(3, 4, 5), torch.randn(3, 5), torch.randn(3, 4)
    stats, logits = prediction_features(p, q, sim, allow_null=True)
    assert stats.shape == (3, 5, 26)
    assert torch.equal(logits[:, 0], q)
    model = UtilitySelector(embedding_dim=6, hidden=8, allow_null=True)
    scores, outputs = model(torch.randn(3, 6), torch.randn(3, 4, 6), p, q, sim)
    assert scores.shape == (3, 5)
    assert torch.equal(outputs[:, 1:], p)
    loss = expected_context_loss(scores, outputs, torch.tensor([0, 1, 2]))
    loss.backward()
    assert model.net[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


def test_oracle_targets_cannot_backpropagate_into_cached_classifier_predictions():
    scores = torch.randn(2, 3, requires_grad=True)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    expected_context_loss(scores, logits, torch.tensor([1, 2])).backward()
    assert logits.grad is None
    assert torch.isfinite(scores.grad).all()
