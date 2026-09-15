from __future__ import annotations

import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .candidate_pool import FixedCandidatePool, PoolItem


def _transform(image_size: int, train: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomCrop(image_size, padding=max(1, image_size // 16)), transforms.RandomHorizontalFlip()]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


class CIFAR100ContextDataset(Dataset):
    def __init__(self, base, candidate_base=None, pool: FixedCandidatePool | None = None,
                 candidates_per_query: int = 0, seed: int = 0, include_context: bool = True,
                 query_ids: list[int] | None = None, return_candidate_images: bool = True):
        self.base, self.candidate_base, self.pool = base, candidate_base, pool
        self.candidates_per_query, self.seed, self.epoch = candidates_per_query, seed, 0
        self.include_context = include_context
        self.query_ids = query_ids
        self.return_candidate_images = return_candidate_images

    def set_epoch(self, epoch: int): self.epoch = epoch
    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        query_id = index if self.query_ids is None else self.query_ids[index]
        if not self.include_context:
            return {"query": image, "label": label, "query_id": query_id}
        ids = self.pool.sample_indices(query_id, self.candidates_per_query, self.epoch * 10000019)
        candidate_labels = ([int(self.candidate_base.targets[i]) for i in ids]
                            if hasattr(self.candidate_base, "targets")
                            else [int(self.candidate_base[i][1]) for i in ids])
        result = {"query": image, "label": label, "query_id": query_id,
                  "candidate_ids": torch.tensor(ids), "candidate_labels": torch.tensor(candidate_labels)}
        if self.return_candidate_images:
            result["candidates"] = torch.stack([self.candidate_base[i][0] for i in ids])
        return result

    def load_candidate_images(self, ids: torch.Tensor) -> torch.Tensor:
        """Load only contexts selected after cached retrieval scores are known."""
        return torch.stack([self.candidate_base[int(i)][0] for i in ids.tolist()])


def make_cifar100_loaders(root: str, image_size: int, pool_ratio: float, candidates_per_query: int,
                           batch_size: int, num_workers: int, seed: int, include_context: bool = True,
                           validation_ratio: float = 0.1, return_candidate_images: bool = True):
    root = str(Path(root))
    train_aug = datasets.CIFAR100(root, train=True, download=True, transform=_transform(image_size, True))
    train_eval = datasets.CIFAR100(root, train=True, download=True, transform=_transform(image_size, False))
    test = datasets.CIFAR100(root, train=False, download=True, transform=_transform(image_size, False))
    ids = list(range(len(train_eval)))
    random.Random(seed).shuffle(ids)
    val_count = max(1, int(len(ids) * validation_ratio))
    val_ids, train_ids = ids[:val_count], ids[val_count:]
    train_base, val_base = Subset(train_aug, train_ids), Subset(train_eval, val_ids)
    # Test images are a separate dataset; offset their IDs so they never cause a
    # spurious exclusion from the training-derived candidate pool.
    test_ids = list(range(len(train_eval), len(train_eval) + len(test)))
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if not include_context:
        train = CIFAR100ContextDataset(train_base, include_context=False, query_ids=train_ids)
        valid = CIFAR100ContextDataset(val_base, include_context=False, query_ids=val_ids)
        test_set = CIFAR100ContextDataset(test, include_context=False, query_ids=test_ids)
        return (DataLoader(train, shuffle=True, **kwargs), DataLoader(valid, shuffle=False, **kwargs),
                DataLoader(test_set, shuffle=False, **kwargs), None)

    pool_ids = train_ids[:max(2, int(len(train_ids) * pool_ratio))]
    pool = FixedCandidatePool([PoolItem(i, int(train_eval.targets[i])) for i in pool_ids], seed)
    train = CIFAR100ContextDataset(train_base, train_eval, pool, candidates_per_query, seed, query_ids=train_ids, return_candidate_images=return_candidate_images)
    valid = CIFAR100ContextDataset(val_base, train_eval, pool, candidates_per_query, seed + 1, query_ids=val_ids, return_candidate_images=return_candidate_images)
    test_set = CIFAR100ContextDataset(test, train_eval, pool, candidates_per_query, seed + 2, query_ids=test_ids, return_candidate_images=return_candidate_images)
    return (DataLoader(train, shuffle=True, **kwargs), DataLoader(valid, shuffle=False, **kwargs),
            DataLoader(test_set, shuffle=False, **kwargs), pool)
