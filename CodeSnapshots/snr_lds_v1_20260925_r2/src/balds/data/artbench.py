"""ArtBench-2/5 dataset loaders (registered; torchvision import is lazy)."""
from __future__ import annotations

from balds.schema.registry import DATASETS
from .base import ImageDataset

# ImageFolder alphabetical indices: baroque=1, post_impressionism=4, renaissance=6,
# romanticism=7, ukiyo_e=9
# --------------------------------------------------------------------------- #
# 256x256 lazy loaders — the SD3.5+LoRA latent platform ([o], ICLR formal)
# --------------------------------------------------------------------------- #

#: Alphabetical ImageFolder order — the canonical style index space. Labels
#: everywhere in the platform are THESE indices (a 2-style subset keeps its
#: original ids, like cifar2 keeps 1/7).
ARTBENCH_STYLES = (
    "art_nouveau", "baroque", "expressionism", "impressionism",
    "post_impressionism", "realism", "renaissance", "romanticism",
    "surrealism", "ukiyo_e",
)


class LazyImageFolder:
    """Path-backed 256x256 dataset: images decoded on access, never resident.

    Consumed only by the ``balds latents`` encode stage (and future visualisation)
    — 50000x3x256x256 floats would be ~39 GB, so unlike the CIFAR loaders this
    one exposes ``.labels`` eagerly but loads pixels per item. Everything
    downstream of the encode stage operates on the cached latents instead.
    """

    def __init__(self, paths: list, labels: list, *, center_crop_square=None) -> None:
        self.paths = paths
        import torch
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.center_crop_square = (list(center_crop_square)
                                   if center_crop_square is not None
                                   else [False] * len(paths))
        if len(self.center_crop_square) != len(self.paths):
            raise ValueError("center-crop flags must align with paths")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.center_crop_square[idx]:
            side = min(img.size)
            left = (img.width - side) // 2
            top = (img.height - side) // 2
            img = img.crop((left, top, left + side, top + side))
        if img.size != (256, 256):                    # ArtBench ships 256x256
            img = img.resize((256, 256), resample=Image.Resampling.BICUBIC)
        x = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)
        return x / 127.5 - 1.0, int(self.labels[idx])


def _artbench_256_split(data_dir: str, split: str, class_ids) -> LazyImageFolder:
    import os
    keep = set(class_ids)
    paths, labels = [], []
    for idx, style in enumerate(ARTBENCH_STYLES):
        if idx not in keep:
            continue
        folder = os.path.join(data_dir, split, style)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"{folder} not found — point datasets.raw_dirs.artbench at the "
                f"artbench-10-imagefolder-split root (train/<style>/, test/<style>/)")
        for fname in sorted(os.listdir(folder)):
            paths.append(os.path.join(folder, fname))
            labels.append(idx)
    return LazyImageFolder(paths, labels)


def _load_artbench10(data_dir: str, **_):
    ids = list(range(10))
    return (_artbench_256_split(data_dir, "train", ids),
            _artbench_256_split(data_dir, "test", ids))


def _load_artbench2_256(data_dir: str, **_):
    """AN-2 counterfactual platform: post_impressionism + ukiyo_e, N=5000
    (2500/style, fixed sampling seed, deterministic — the cifar2_5k pattern)."""
    import numpy as np
    ids = [4, 9]
    train = _artbench_256_split(data_dir, "train", ids)
    rng = np.random.RandomState(42)
    labels = train.labels.numpy()
    keep = []
    for cls in ids:
        idx = np.flatnonzero(labels == cls)           # sorted-filename order
        assert len(idx) >= 2500, f"style {cls}: only {len(idx)} images"
        keep.append(rng.choice(idx, 2500, replace=False))
    keep = np.sort(np.concatenate(keep))
    sub = LazyImageFolder([train.paths[i] for i in keep], labels[keep].tolist())
    return sub, _artbench_256_split(data_dir, "test", ids)


DATASETS.add("artbench2_256", _load_artbench2_256)
