import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_query_noise_reduced import reduced_summary


def test_dropping_seed44_preserves_both_retained_seeds_but_disables_original_gate():
    correct = np.zeros((2, 2, 4, 2))
    correct[:, 1] = 1
    result = reduced_summary(correct)
    assert result["seed_differences_pp"] == [100., 100.]
    assert result["all_retained_seeds_positive"]
    assert result["omitted_seeds"] == [44]
    assert not result["promotion_gate_passed"]
    assert result["seed_t_ci95_pp"] is None
    with pytest.raises(ValueError):
        reduced_summary(np.zeros((3, 2, 4, 2)))
