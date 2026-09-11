from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class PoolItem:
    index: int
    label: int


class FixedCandidatePool:
    """Fixed train-derived pool; sampling excludes the query source index."""

    def __init__(self, items: list[PoolItem], seed: int = 0):
        if len(items) < 2:
            raise ValueError("candidate pool needs at least two items")
        self.items, self.seed = list(items), seed

    def sample_indices(self, query_index: int, n: int, seed_offset: int = 0) -> list[int]:
        valid = [item.index for item in self.items if item.index != query_index]
        if not valid:
            raise ValueError("candidate pool contains only the query")
        rng = random.Random(self.seed + seed_offset + query_index * 1000003)
        return rng.sample(valid, n) if n <= len(valid) else [rng.choice(valid) for _ in range(n)]
