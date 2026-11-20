"""Dataset loading. Classnames and prompt templates come from clip_benchmark,
which is what produced the published OpenCLIP zero-shot table, so our numbers are comparable.

Splits follow Liao et al. (TMLR 2026), Sec. 4 and Appendix A:
* Flowers102: torchvision train + val for training, test for evaluation.
* EuroSAT (no official split): "a seed-0 random permutation followed by an 80/20 train/test division".
  The paper does not name the RNG or which end is train; we use numpy default_rng(seed) and take the
  first 80% as train. So our indices will not equal theirs, only the protocol.
* Target evaluation: "a seed-0 uniform random sample of 2,000 test examples" (see subsample()).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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
# datasets whose training set is the union of several official splits (Liao et al.)
TRAIN_SPLITS = {"flowers": ["train", "val"], "pets": ["trainval"]}


class LabeledSubset(Subset):
    """Subset that keeps .classes and .templates so zero-shot code can use it like the full set."""

    def __init__(self, dataset: Dataset, indices, classes, templates):
        super().__init__(dataset, list(indices))
        self.classes = classes
        self.templates = templates


class LabeledConcat(ConcatDataset):
    """ConcatDataset that keeps .classes and .templates (all parts must share them)."""

    def __init__(self, datasets):
        super().__init__(datasets)
        self.classes = datasets[0].classes
        self.templates = datasets[0].templates


def get_targets(ds) -> np.ndarray:
    """Integer labels without loading images when the dataset exposes them."""
    for attr in ("targets", "_labels", "labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    return np.asarray([ds[i][1] for i in range(len(ds))])  # slow fallback


def permutation_split(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random permutation, first (1 - test_fraction) is train, rest is test (Liao et al. EuroSAT protocol)."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(n * (1.0 - test_fraction)))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def stratified_split(targets, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class shuffled split. Returns (train_idx, test_idx), disjoint and covering all indices.

    Kept as an option (data.split_mode: stratified); the default follows Liao et al. (permutation).
    """
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
    """Uniform random subset of at most max_samples items (no replacement). None keeps everything.

    Used for Liao et al.'s 2,000-example target evaluation and for quick smoke runs.
    """
    if max_samples is None or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=max_samples, replace=False))
    return LabeledSubset(ds, idx, ds.classes, ds.templates)


def build_dataset(name: str, split: str, root: str, transform, split_seed: int = 0,
                  eurosat_test_fraction: float = 0.2, split_mode: str = "permutation",
                  download: bool = True):
    """Return a dataset with .classes and .templates. split is 'train' or 'test'.

    split_seed only affects our own EuroSAT split. Keep it fixed across training seeds so the
    test set never changes between runs.
    """
    from clip_benchmark.datasets.builder import build_dataset as cb_build

    if name not in CLIP_BENCHMARK_NAMES:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(CLIP_BENCHMARK_NAMES)}")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    def _cb(s):
        return cb_build(CLIP_BENCHMARK_NAMES[name], root=f"{root}/{name}", transform=transform,
                        split=s, download=download)

    if name in NEEDS_OWN_SPLIT:
        ds = _cb("test")  # split argument is ignored by clip_benchmark for these
        if split_mode == "permutation":
            train_idx, test_idx = permutation_split(len(ds), eurosat_test_fraction, split_seed)
        elif split_mode == "stratified":
            train_idx, test_idx = stratified_split(get_targets(ds), eurosat_test_fraction, split_seed)
        else:
            raise ValueError(f"split_mode must be permutation or stratified, got {split_mode!r}")
        return LabeledSubset(ds, train_idx if split == "train" else test_idx, ds.classes, ds.templates)

    if split == "train":
        parts = [_cb(s) for s in TRAIN_SPLITS.get(name, ["train"])]
        return parts[0] if len(parts) == 1 else LabeledConcat(parts)
    return _cb("test")


def make_loader(ds, batch_size: int, num_workers: int, shuffle: bool, seed: int = 0,
                drop_last: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last, generator=g)
