"""Dataset loading. Classnames and prompt templates come from clip_benchmark,
which is what produced the published OpenCLIP zero-shot table, so our numbers are comparable.

Note on EuroSAT: clip_benchmark's plain "eurosat" ignores the split argument (torchvision has
no official split). We make our own seeded, stratified train/test split. The published
OpenCLIP EuroSAT number may use a different split, so treat that comparison as approximate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# our short name -> clip_benchmark dataset name
CLIP_BENCHMARK_NAMES = {
    "flowers": "flowers",
    "eurosat": "eurosat",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "dtd": "dtd",
    "food101": "food101",
    "pets": "pets",
}

# datasets whose clip_benchmark loader has no real split and needs our own
NEEDS_OWN_SPLIT = {"eurosat"}


class LabeledSubset(Subset):
    """Subset that keeps .classes and .templates so zero-shot code can use it like the full set."""

    def __init__(self, dataset: Dataset, indices, classes, templates):
        super().__init__(dataset, list(indices))
        self.classes = classes
        self.templates = templates


def get_targets(ds) -> np.ndarray:
    """Integer labels without loading images when the dataset exposes them."""
    for attr in ("targets", "_labels", "labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    return np.asarray([ds[i][1] for i in range(len(ds))])  # slow fallback


def stratified_split(targets, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class shuffled split. Returns (train_idx, test_idx), disjoint and covering all indices."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(targets):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(test_idx))


def subsample(ds, max_samples: int | None, seed: int):
    """Random subset of at most max_samples items, for quick smoke runs. None keeps everything."""
    if max_samples is None or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=max_samples, replace=False))
    return LabeledSubset(ds, idx, ds.classes, ds.templates)


def build_dataset(name: str, split: str, root: str, transform, split_seed: int = 0,
                  eurosat_test_fraction: float = 0.2, download: bool = True):
    """Return a dataset with .classes and .templates. split is 'train' or 'test'.

    split_seed only affects our own EuroSAT split. Keep it fixed across training seeds so the
    test set never changes between runs.
    """
    from clip_benchmark.datasets.builder import build_dataset as cb_build

    if name not in CLIP_BENCHMARK_NAMES:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(CLIP_BENCHMARK_NAMES)}")
    cb_split = "test" if name in NEEDS_OWN_SPLIT else ("trainval" if (name == "pets" and split == "train") else split)
    ds = cb_build(CLIP_BENCHMARK_NAMES[name], root=f"{root}/{name}", transform=transform,
                  split=cb_split, download=download)
    if name in NEEDS_OWN_SPLIT:
        train_idx, test_idx = stratified_split(get_targets(ds), eurosat_test_fraction, split_seed)
        ds = LabeledSubset(ds, train_idx if split == "train" else test_idx, ds.classes, ds.templates)
    return ds


def make_loader(ds, batch_size: int, num_workers: int, shuffle: bool, seed: int = 0,
                drop_last: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last, generator=g)
