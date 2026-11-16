import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_warmup_oracle import nearest_same_class, oracle_views


def test_same_class_is_not_accuracy_bound_and_missing_class_has_declared_fallback():
    logits = torch.tensor([[[0., 2.], [3., 0.]], [[1., 0.], [0., 2.]]])
    labels = torch.tensor([0, 1])
    candidate_labels = torch.tensor([[0, 1], [0, 0]])
    result = oracle_views(logits, labels, candidate_labels, torch.tensor([[2., 1.], [1., 2.]]))
    assert result["same_class_available16"].tolist() == [True, False]
    assert result["actions"]["oracle_same_class_shortlist"].tolist() == [0, 0]
    assert result["any_correct16"].tolist() == [True, True]
    assert result["logits"]["oracle_same_class_shortlist"].argmax(-1).tolist() == [1, 0]
    assert result["logits"]["oracle_min_ce16"].argmax(-1).tolist() == labels.tolist()


def test_full_pool_label_oracle_masks_ids_and_pixel_duplicates_before_selecting():
    q = torch.tensor([[1., 0.]])
    c = torch.tensor([[1., 0.], [.99, .01], [.9, .1], [0., 1.]])
    args = (q, c, torch.tensor([5]), torch.tensor([5, 7, 8, 9]), torch.tensor([100]),
            torch.tensor([100, 100, 101, 102]), torch.tensor([0]), torch.tensor([0, 0, 0, 1]))
    assert nearest_same_class(*args).tolist() == [2]
    with pytest.raises(ValueError):
        nearest_same_class(*args[:-1], torch.tensor([1, 1, 1, 1]))
