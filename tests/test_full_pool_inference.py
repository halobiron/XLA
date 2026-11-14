import torch
from torch import nn
import torch.nn.functional as F

from tacsx.models.selector import TACSSelector


class DummyEncoder(nn.Module):
    embedding_dim = 3

    def encode_global(self, images):
        return images.mean((-2, -1))


def test_cached_full_pool_matches_native_selector_scores_and_actions():
    torch.manual_seed(41)
    selector = TACSSelector(DummyEncoder()).eval()
    query, candidates = torch.randn(5, 3, 4, 4), torch.randn(13, 3, 4, 4)
    native = selector(query, candidates[None].expand(5, -1, -1, -1, -1))
    q = F.normalize(selector.projector(selector.encoder.encode_global(query)), dim=-1)
    c = F.normalize(selector.projector(selector.encoder.encode_global(candidates)), dim=-1)
    cached = q @ c.T
    assert torch.allclose(native, cached, atol=1e-6)
    assert torch.equal(native.argmax(-1), cached.argmax(-1))
