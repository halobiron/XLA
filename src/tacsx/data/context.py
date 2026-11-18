from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .candidate_pool import FixedCandidatePool


class ContextDataset(Dataset):
    """Dataset-independent query wrapper with a fixed training-only pool."""

    def __init__(self, base, candidate_base=None, pool: FixedCandidatePool | None = None,
                 candidates_per_query=0, seed=0, include_context=True, query_ids=None,
                 return_candidate_images=True):
        self.base, self.candidate_base, self.pool = base, candidate_base, pool
        self.candidates_per_query, self.seed, self.epoch = candidates_per_query, seed, 0
        self.include_context, self.query_ids = include_context, query_ids
        self.return_candidate_images = return_candidate_images

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        query_id = index if self.query_ids is None else self.query_ids[index]
        # Match crops across modes without changing the model's CPU/CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + self.epoch * 10000019 + query_id)
            image, label = self.base[index]
        result = {"query": image, "label": label, "query_id": query_id}
        if not self.include_context:
            return result
        ids = self.pool.sample_indices(query_id, self.candidates_per_query, self.epoch * 10000019)
        result.update(candidate_ids=torch.tensor(ids),
                      candidate_labels=torch.tensor([self.candidate_base.targets[i] for i in ids]))
        if self.return_candidate_images:
            result["candidates"] = self.load_candidate_images(torch.tensor(ids))
        return result

    def load_candidate_images(self, ids):
        return torch.stack([self.candidate_base[int(i)][0] for i in ids.tolist()])
