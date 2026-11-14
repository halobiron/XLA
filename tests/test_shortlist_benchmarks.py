import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_shortlist_benchmarks import summarize


def test_identical_models_cannot_establish_any_benchmark_gain():
    torch.manual_seed(11)
    logits = torch.randn(16, 9, 3)
    labels = torch.arange(16) % 3
    result = summarize({"full_tacs": logits, "no_context": logits.clone()}, labels)
    assert result["positive_familywise_metrics"] == []
    for value in result["tacs_improvement"].values():
        assert value["ci"] == [0, 0]
        assert value["confidence_level"] > .98
    assert len(result["completion_evidence_seven_comparisons"]) == 7
    for value in result["completion_evidence_seven_comparisons"].values():
        assert value["ci"] == [0, 0]
        assert value["confidence_level"] > .99


def test_nll_is_a_distinct_metric_and_uses_lower_is_better():
    labels = torch.zeros(16, dtype=torch.long)
    baseline = torch.tensor([1., 0.]).expand(16, 9, 2).clone()
    improved = torch.tensor([2., 0.]).expand(16, 9, 2).clone()
    result = summarize({"full_tacs": improved, "no_context": baseline}, labels)
    assert result["positive_familywise_metrics"] == ["clean_nll_reduction"]
    assert result["tacs_improvement"]["clean_accuracy_pp"]["difference"] == 0
    assert result["tacs_improvement"]["clean_nll_reduction"]["ci"][0] > 0
