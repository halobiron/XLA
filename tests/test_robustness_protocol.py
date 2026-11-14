"""Protocol checks without CIFAR data or model downloads."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_tacs_robustness import corrupt, paired_ci


def test_noise_is_independent_of_order_and_global_rng():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    first = np.asarray(corrupt(image, "gaussian_noise", .08, 12))
    np.random.seed(999)
    corrupt(image, "gaussian_noise", .08, 13)
    assert np.array_equal(first, np.asarray(corrupt(image, "gaussian_noise", .08, 12)))
    assert not np.array_equal(first, np.asarray(corrupt(image, "gaussian_noise", .08, 13)))
    assert np.array_equal(image, np.full_like(image, 128))


def test_clean_control_preserves_all_pixels():
    image = np.random.default_rng(12).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert np.array_equal(image, np.asarray(corrupt(image, "clean", 0, 12)))


def test_cluster_bootstrap_handles_a_zero_difference():
    result = paired_ci(np.zeros(12))
    assert result["difference"] == 0
    assert result["ci"] == [0, 0]
