import numpy as np
import pytest
import torch

from tacsx.data.identity import pixel_identity_ids, heldout_without_identity_overlap
from tacsx.models.retrieval_fusion import retrieve


def test_pixel_duplicates_share_identity_even_when_source_ids_differ():
    a, b = np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)
    groups = pixel_identity_ids([a, b, a.copy(), b.copy()])
    assert groups.tolist() == [0, 1, 0, 1]
    keep, excluded = heldout_without_identity_overlap([0], [2, 1, 3], groups)
    assert keep == [1]
    assert excluded == [2, 3]


def test_candidate_identity_exclusion_is_stricter_than_row_exclusion():
    q = torch.tensor([[1., 0.]])
    pool = torch.tensor([[1., 0.], [.9, .1], [0., 1.]])
    _, positions = retrieve(q, pool, torch.tensor([50, 60, 70]), 1,
                            query_ids=torch.tensor([99]), query_identity_ids=torch.tensor([5]),
                            pool_identity_ids=torch.tensor([5, 6, 7]))
    assert positions.item() == 1
    with pytest.raises(ValueError, match="Both query"):
        retrieve(q, pool, torch.tensor([50, 60, 70]), 1, query_identity_ids=torch.tensor([5]))
