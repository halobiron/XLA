from __future__ import annotations

from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .candidate_pool import FixedCandidatePool, PoolItem
from .context import ContextDataset


def image_transform(image_size: int, training: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if training:
        ops += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()]
    return transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


def _read_table(path):
    table = {}
    for line in path.read_text(encoding="utf8").splitlines():
        index, value = line.split(maxsplit=1)
        index = int(index)
        if index in table:
            raise ValueError(f"Duplicate ID {index} in {path.name}")
        table[index] = value
    return table


class CUB200Images(Dataset):
    """Official CUB-200-2011 metadata; global row ID = official image ID - 1.

    Construction reads metadata only. Image pixels are opened on demand, so
    ordinary validation-only runs never open the official test images.
    """

    def __init__(self, root, transform=None):
        root = Path(root)
        if (root / "CUB_200_2011").is_dir():
            root = root / "CUB_200_2011"
        required = ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")
        missing = [name for name in required if not (root / name).is_file()]
        if missing or not (root / "images").is_dir():
            raise FileNotFoundError(
                f"Extract CUB_200_2011.tgz into data/ first; expected metadata and images/ at {root}. "
                "Official download: https://data.caltech.edu/records/65de6-vp158")
        paths, labels, split, classes = [_read_table(root / name) for name in required]
        if set(paths) != set(labels) or set(paths) != set(split) or sorted(paths) != list(range(1, len(paths) + 1)):
            raise ValueError("CUB image IDs must be contiguous and match across metadata files")
        self.root, self.transform = root, transform
        self.classes = [classes[i] for i in sorted(classes)]
        self.paths = [paths[i] for i in sorted(paths)]
        self.targets = [int(labels[i]) - 1 for i in sorted(paths)]
        self.is_train = [int(split[i]) for i in sorted(paths)]
        if set(self.is_train) - {0, 1} or any(y < 0 or y >= len(self.classes) for y in self.targets):
            raise ValueError("Invalid CUB label or official split flag")
        image_root = (root / "images").resolve()
        if any(not (image_root / name).resolve().is_relative_to(image_root) for name in self.paths):
            raise ValueError("Image path escapes CUB images directory")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / "images" / self.paths[index]) as image:
            image = image.convert("RGB")
        return (self.transform(image) if self.transform else image), self.targets[index]


def make_cub200_loaders(root, image_size, pool_ratio, candidates_per_query, batch_size,
                        num_workers, seed, include_context=True, validation_ratio=.1,
                        return_candidate_images=True):
    if not 0 < validation_ratio < 1 or not 0 < pool_ratio <= 1:
        raise ValueError("validation_ratio must be in (0,1), pool_ratio in (0,1]")
    if include_context and candidates_per_query < 1:
        raise ValueError("Context modes need at least one candidate per query")
    train_aug = CUB200Images(root, image_transform(image_size, True))
    clean = CUB200Images(root, image_transform(image_size, False))
    rng = random.Random(seed)
    train_ids, val_ids = [], []
    for label in range(len(clean.classes)):
        ids = [i for i, (y, flag) in enumerate(zip(clean.targets, clean.is_train)) if flag and y == label]
        if len(ids) < 2:
            raise ValueError(f"Class {label} needs at least two official training images")
        rng.shuffle(ids)
        n = min(len(ids) - 1, max(1, int(len(ids) * validation_ratio)))
        val_ids.extend(ids[:n])
        train_ids.extend(ids[n:])
    rng.shuffle(train_ids)
    test_ids = [i for i, flag in enumerate(clean.is_train) if not flag]
    pool = None
    if include_context:
        if len(train_ids) < 2:
            raise ValueError("Insufficient training images for candidate pool")
        pool_ids = train_ids[:max(2, int(len(train_ids) * pool_ratio))]
        pool = FixedCandidatePool([PoolItem(i, clean.targets[i]) for i in pool_ids], seed)
    datasets = [ContextDataset(Subset(base, ids), clean, pool, candidates_per_query, seed,
                              include_context, ids, return_candidate_images)
                for base, ids in ((train_aug, train_ids), (clean, val_ids), (clean, test_ids))]
    loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=(i == 0), num_workers=num_workers,
                          pin_memory=True, generator=torch.Generator().manual_seed(seed + i))
               for i, dataset in enumerate(datasets)]
    return (*loaders, pool)
