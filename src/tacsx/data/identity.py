"""Exact pixel identities, independent of class labels and source row IDs."""
from __future__ import annotations

import hashlib
import torch


def pixel_identity_ids(images):
    groups, canonical = [], {}
    for index, image in enumerate(images):
        digest = hashlib.sha256(image.tobytes()).digest()
        groups.append(canonical.setdefault(digest, index))
    return torch.tensor(groups, dtype=torch.long)


def heldout_without_identity_overlap(training_ids, heldout_ids, identity_ids):
    """Exclude seen pixels and repeated heldout pixels without consulting labels."""
    seen = {int(identity_ids[i]) for i in training_ids}
    keep, excluded = [], []
    for index in heldout_ids:
        group = int(identity_ids[index])
        if group in seen:
            excluded.append(index)
        else:
            keep.append(index)
            seen.add(group)
    return keep, excluded
