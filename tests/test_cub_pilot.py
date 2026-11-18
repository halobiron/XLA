import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("cub_pilot", Path(__file__).parents[1] / "scripts/run_cub_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_full_pool_selection_matches_reference_values_and_score_gradients():
    scores = torch.randn(3, 7, requires_grad=True)
    other = scores.detach().clone().requires_grad_(True)
    pixels = torch.randn(7, 3, 4, 4)
    weights = torch.randn(3, 3, 4, 4)
    torch.manual_seed(91)
    selected, actions = pilot.select_context(scores, pixels)
    torch.manual_seed(91)
    mask = F.gumbel_softmax(other, tau=.1, hard=True)
    reference = torch.einsum("bn,nchw->bchw", mask, pixels)
    assert torch.equal(selected, pixels[actions])
    assert torch.equal(selected, reference)
    a = torch.autograd.grad((selected * weights).sum(), scores)[0]
    b = torch.autograd.grad((reference * weights).sum(), other)[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_futility_gate_waits_until_epoch3_and_requires_both_conditions():
    assert not pilot.futile(2, {"difference_pp": -10, "ci95_pp": [-12, -8]})
    assert pilot.futile(3, {"difference_pp": -3, "ci95_pp": [-5, -1]})
    assert not pilot.futile(3, {"difference_pp": -3, "ci95_pp": [-6, .1]})
    assert not pilot.futile(3, {"difference_pp": -1, "ci95_pp": [-1.5, -.5]})


def test_paired_comparison_preserves_sign_and_percentage_point_units():
    stats = pilot.paired_stats([True] * 40, [False] * 40)
    assert stats["difference_pp"] == -100
    assert stats["ci95_pp"] == [-100, -100]
