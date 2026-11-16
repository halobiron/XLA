import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_query_noise_pilot import NoisyQueries, paired_seed_summary, spec


def test_paired_input_noise_and_augmentation_ignore_model_rng():
    source = SimpleNamespace(images=np.full((3, 32, 32, 3), 128, dtype=np.uint8), targets=[0, 1, 2])
    data = NoisyQueries(source, [2, 0, 1], seed=42, epoch=1)
    state = torch.random.get_rng_state()
    first = data[0]
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.rand(100)
    second = data[0]
    assert torch.equal(first["query"], second["query"])
    assert first["sigma_index"] == second["sigma_index"]
    assert np.all(source.images == 128)  # clean candidate source is immutable
    a, b = NoisyQueries(source, [2], 42, sigma_index=0)[0], NoisyQueries(source, [2], 44, sigma_index=0)[0]
    assert torch.equal(a["query"], b["query"])  # validation shared across seeds


def test_gate_uses_seed_uncertainty_and_requires_all_seeds_positive():
    correct = np.zeros((3, 2, 100, 2))
    # All seed means positive, but large between-seed variation excludes promotion.
    for seed, wins in enumerate([1, 2, 50]):
        correct[seed, 1, :wins] = 1
    summary = paired_seed_summary(correct)
    assert summary["all_three_positive"]
    assert summary["seed_t_ci95_pp"][0] < 0
    assert not summary["promotion_gate_passed"]
    correct[:, 1] = 0
    correct[:, 1, :10] = 1
    assert paired_seed_summary(correct)["promotion_gate_passed"]
    correct[0, 1] = 0
    assert not paired_seed_summary(correct)["promotion_gate_passed"]
    with pytest.raises(ValueError):
        paired_seed_summary(correct[:2])


def test_protocol_fixes_final_epoch_and_prevents_automatic_test():
    protocol = spec(SimpleNamespace(batch_size=32, workers=4))
    assert protocol["seeds"] == [42, 43, 44]
    assert protocol["epochs"] == 4
    assert protocol["noise"]["sigma"] == [.08, .16]
    assert protocol["test_access"] is False
    assert protocol["automatic_test_or_kaggle"] is False
    assert "fixed final epoch4" in protocol["checkpoint"]
