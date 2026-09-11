from __future__ import annotations

import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .candidate_pool import FixedCandidatePool, PoolItem


def _transform(image_size: int, train: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomCrop(image_size, padding=max(1, image_size // 16)), transforms.RandomHorizontalFlip()]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


class CIFAR100ContextDataset(Dataset):
    def __init__(self, base, candidate_base, pool: FixedCandidatePool, candidates_per_query: int, seed: int = 0):
        self.base, self.candidate_base, self.pool = base, candidate_base, pool
        self.candidates_per_query, self.seed, self.epoch = candidates_per_query, seed, 0

    def set_epoch(self, epoch: int): self.epoch = epoch
    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        ids = self.pool.sample_indices(index, self.candidates_per_query, self.epoch * 10000019)
        candidates, candidate_labels = zip(*(self.candidate_base[i] for i in ids))
        return {"query": image, "label": label, "query_id": index, "candidates": torch.stack(candidates),
                "candidate_ids": torch.tensor(ids), "candidate_labels": torch.tensor(candidate_labels)}


def make_cifar100_loaders(root: str, image_size: int, pool_ratio: float, candidates_per_query: int,
                           batch_size: int, num_workers: int, seed: int):
    root = str(Path(root))
    train_aug = datasets.CIFAR100(root, train=True, download=True, transform=_transform(image_size, True))
    train_eval = datasets.CIFAR100(root, train=True, download=True, transform=_transform(image_size, False))
    test = datasets.CIFAR100(root, train=False, download=True, transform=_transform(image_size, False))
    ids = list(range(len(train_eval))); random.Random(seed).shuffle(ids)
    pool = FixedCandidatePool([PoolItem(i, int(train_eval.targets[i])) for i in ids[:max(2, int(len(ids) * pool_ratio))]], seed)
    train = CIFAR100ContextDataset(train_aug, train_eval, pool, candidates_per_query, seed)
    valid = CIFAR100ContextDataset(test, train_eval, pool, candidates_per_query, seed + 1)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(valid, shuffle=False, **kwargs), pool
