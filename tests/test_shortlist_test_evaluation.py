import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_shortlist_test import summarize_clean, verify_checkpoint_lock


def test_paired_counts_and_accuracy_use_every_input_image():
    labels = torch.tensor([0, 0, 0, 0])
    baseline = torch.tensor([[2., 0.], [0., 2.], [2., 0.], [0., 2.]])
    full = torch.tensor([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
    result = summarize_clean({"no_context": baseline, "full_tacs": full}, labels)
    assert result["count"] == 4
    assert result["metrics"]["no_context"]["correct"] == 2
    assert result["metrics"]["full_tacs"]["clean_accuracy"] == .5
    assert result["paired_counts"] == {"fixed": 1, "broken": 1, "both_correct": 1, "both_wrong": 1}
    assert result["tacs_improvement"]["clean_accuracy_pp"]["difference"] == 0


def test_nll_reduction_can_be_negative_without_accuracy_changes():
    labels = torch.zeros(4, dtype=torch.long)
    result = summarize_clean({"no_context": torch.tensor([[2., 0.]]).repeat(4, 1),
                              "full_tacs": torch.tensor([[1., 0.]]).repeat(4, 1)}, labels)
    assert result["tacs_improvement"]["clean_accuracy_pp"]["difference"] == 0
    assert result["tacs_improvement"]["clean_nll_reduction"]["ci"][1] < 0


def test_test_evaluation_refuses_changed_validation_checkpoints():
    locked = {"checkpoint_sha256": {"no_context": "original", "full_tacs": "original"}}
    verify_checkpoint_lock(dict(locked["checkpoint_sha256"]), locked)
    with pytest.raises(ValueError, match="original validation lock"):
        verify_checkpoint_lock({"no_context": "new", "full_tacs": "original"}, locked)
