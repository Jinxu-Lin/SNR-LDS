"""CIFAR-2/5/10 dataset loaders (registered; HuggingFace import is lazy)."""
from __future__ import annotations

from balds.schema.registry import DATASETS
from .base import ImageDataset

CIFAR_CLASSES = {
    "cifar2": [1, 7],                 # automobile + horse (D-TRAK/DAS standard)
    "cifar5": [0, 1, 2, 7, 8],
    "cifar10": list(range(10)),
}


def _convert(split):
    import numpy as np
    import torch
    images, labels = [], []
    for item in split:
        img = item["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img), dtype=torch.float32)
        if img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        images.append(img / 127.5 - 1.0)             # -> [-1, 1]
        labels.append(item["label"])
    return ImageDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


def _load_cifar(classes, data_dir):
    from datasets import load_dataset
    ds = load_dataset("cifar10", cache_dir=data_dir)
    class_set = set(classes)
    out = {}
    for split in ("train", "test"):
        data = ds[split].filter(lambda x: x["label"] in class_set) if len(classes) < 10 else ds[split]
        out[split] = _convert(data)
    return out["train"], out["test"]


def load_cifar_by_index(data_dir, train_index, test_index, *, classes):
    """``(train, test)`` selected by ORIGINAL cifar10 split index, order preserved.

    The other loaders here define a dataset by a class filter plus (optionally) a
    seeded subsample, so a position in the result means nothing outside this
    code. An imported dataset is the opposite case: some other project's index
    list IS the definition, and row ``i`` of its feature matrices is
    ``train_index[i]``. Hence no sorting, no filtering, no dedup — the given
    order is the contract, and ``classes`` is only checked, never applied.
    """
    from datasets import load_dataset
    ds = load_dataset("cifar10", cache_dir=data_dir)
    allowed, out = set(int(c) for c in classes), []
    for split, index in (("train", train_index), ("test", test_index)):
        data = _convert(ds[split].select([int(i) for i in index]))
        seen = set(int(v) for v in data.labels.tolist())
        if not seen <= allowed:
            raise ValueError(f"{split} index selects classes {sorted(seen)}, "
                             f"outside the declared {sorted(allowed)}")
        out.append(data)
    return out[0], out[1]




# --- cifar2_5k: the ICLR-formal CIFAR-2 (adjudicated 2026-08-07) -------------
# N=5000 = 2500/class balanced subsample of the 10k cifar2 train split, drawn
# with a FIXED sampling seed (independent of any run seed) so the training set
# is one deterministic object across seeds/machines. Registered as its own
# dataset key so every artifact path (masks/GT/featurize/scores) is isolated
# from the legacy 10k tree by construction. Test split = full filtered 2k.
CIFAR2_5K_PER_CLASS = 2500
CIFAR2_5K_SAMPLING_SEED = 42


def _load_cifar2_5k(data_dir, **_):
    import numpy as np
    train, test = _load_cifar(CIFAR_CLASSES["cifar2"], data_dir)
    rng = np.random.RandomState(CIFAR2_5K_SAMPLING_SEED)
    keep = []
    labels = train.labels.numpy()
    for cls in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == cls)          # HF order: deterministic
        assert len(idx) >= CIFAR2_5K_PER_CLASS, f"class {cls}: {len(idx)} < {CIFAR2_5K_PER_CLASS}"
        keep.append(rng.choice(idx, CIFAR2_5K_PER_CLASS, replace=False))
    keep = np.sort(np.concatenate(keep))             # original order preserved
    return ImageDataset(train.images[keep], train.labels[keep]), test


DATASETS.add("cifar2_5k", _load_cifar2_5k)

# cifar10_v2: same full 50k CIFAR-10 data, NEW product-tree key for the ICLR
# formal rerun (adjudicated 2026-08-07: old cifar10 products are archived-only;
# reusing the "cifar10" key would collide with the archived tree on the master
# and idempotent skips would silently reuse弃用 checkpoints — same isolation
# rationale as cifar2_5k, dispatcher 定案 2026-08-09).
DATASETS.add("cifar10_v2", lambda data_dir, **_: _load_cifar(CIFAR_CLASSES["cifar10"], data_dir))
