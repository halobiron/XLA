import importlib.util
from pathlib import Path
import sys

import pytest
import torch

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("cub_selector_learning", scripts / "learn_cub_selector.py")
learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning)


def test_selector_split_excludes_pool_aliases_and_groups_query_identities():
    ids = list(range(14))
    identities = torch.tensor([0, 0, 2, 3, 4, 5, 6, 7, 7, 9, 10, 11, 12, 13])
    targets = [0] * 7 + [1] * 7
    split = learning.make_splits(ids, [0, 7], identities, targets, fit_per_class=2, holdout_per_class=1)
    assert len(split["fit"]) == 4 and len(split["holdout"]) == 2
    fit = {int(identities[i]) for i in split["fit"]}
    holdout = {int(identities[i]) for i in split["holdout"]}
    assert not fit & holdout
    assert not (fit | holdout) & {0, 7}
    with pytest.raises(ValueError, match="Insufficient"):
        learning.make_splits(ids, [0, 7], identities, targets, fit_per_class=5, holdout_per_class=1)


def test_cosine_selector_ignores_candidate_predictions_and_has_finite_learning_signal():
    model = learning.build_head("cosine_metric")
    pool = torch.randn(6, 384)
    data = {"query_features": torch.randn(2, 384), "positions": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "pair_logits": torch.randn(2, 3, 5)}
    scores, logits = learning.head_forward("cosine_metric", model, data, pool, torch.arange(2))
    data["pair_logits"] = torch.randn_like(logits) * 100
    again, _ = learning.head_forward("cosine_metric", model, data, pool, torch.arange(2))
    assert torch.equal(scores, again)
    loss = learning.expected_context_loss(scores, logits, torch.tensor([0, 1]))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
