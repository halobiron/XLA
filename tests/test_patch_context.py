import pytest
import torch

from tacsx.models.patch_context import PatchContextClassifier


@pytest.mark.parametrize("use_context", [False, True])
def test_patch_correction_starts_exactly_at_baseline_and_learns(use_context):
    model = PatchContextClassifier(torch.randn(7, 12), use_context=use_context, key_dim=4, value_dim=8)
    q, patches, context, logits = torch.randn(3, 12), torch.randn(3, 5, 12), torch.randn(3, 5, 12), torch.randn(3, 7)
    output = model(q, patches, logits, context)
    assert torch.equal(output, logits)
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 2])).backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_context_path_uses_patch_content_after_correction_learns():
    model = PatchContextClassifier(torch.randn(7, 12), key_dim=4, value_dim=8).eval()
    torch.nn.init.normal_(model.output.weight, std=.1)
    q, patches, logits = torch.randn(3, 12), torch.randn(3, 5, 12), torch.randn(3, 7)
    a = model(q, patches, logits, torch.randn(3, 5, 12))
    b = model(q, patches, logits, torch.randn(3, 5, 12))
    assert not torch.allclose(a, b)
    with pytest.raises(ValueError, match="Context patches"):
        model(q, patches, logits)
