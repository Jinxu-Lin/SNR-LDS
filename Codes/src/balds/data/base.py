"""Dataset value types (Data-Access layer)."""
from __future__ import annotations

import numpy as np


class ImageDataset:
    """Images in ``[-1, 1]`` with original class labels; exposes ``.images``/``.labels``.

    The featurizer and ground-truth code index ``.images[i]`` / ``.labels[i]``.
    """

    def __init__(self, images, labels) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class SubsetView(ImageDataset):
    """A boolean-mask view over a dataset (the training subset for an LDS model)."""

    def __init__(self, dataset: ImageDataset, mask) -> None:
        idx = np.where(np.asarray(mask))[0]
        super().__init__(dataset.images[idx], dataset.labels[idx])


class LatentPairDataset(ImageDataset):
    """Training dataset for the latent platform: random-horizontal-flip via a
    coin flip between two PRE-ENCODED orientations per item.

    Flipping a latent tensor spatially is NOT the latent of the flipped image
    (the VAE is only approximately equivariant), so the encode stage caches
    both orientations and augmentation becomes an index choice. ``.images``
    stays the unflipped tensor — every non-training consumer (masks length,
    featurization, GT) reads the canonical orientation, exactly like the CIFAR
    pipeline featurizes unaugmented images. ``__getitem__`` uses the global
    torch RNG (as does the training loop's own noise), so it belongs in
    training only.
    """

    def __init__(self, images, images_flipped, labels) -> None:
        super().__init__(images, labels)
        if images_flipped.shape != images.shape:
            raise ValueError("orientation tensors must be congruent")
        self.images_flipped = images_flipped

    def __getitem__(self, idx):
        import torch
        flip = bool(torch.rand(()) < 0.5)
        return (self.images_flipped[idx] if flip else self.images[idx]), self.labels[idx]

    def masked(self, mask) -> "LatentPairDataset":
        idx = np.where(np.asarray(mask))[0]
        return LatentPairDataset(self.images[idx], self.images_flipped[idx],
                                 self.labels[idx])


def balanced_query_indices(labels, Q: int) -> list[int]:
    """Fixed, class-balanced indices for the ``val`` query track.

    Deterministic by construction, no RNG: classes in ascending order, and within
    each class the first ``Q/num_classes`` items in the split's natural order.

    Exists because the val track must be the *same* Q images everywhere it is
    consumed — GT loss computation, query featurization and per-protocol scoring
    each selected ``test.images[:Q]`` independently. That is neither class
    balanced (it takes whatever the split happens to order first) nor guaranteed
    to stay in agreement if any one call site is edited. The formal ICLR spec
    (experiment.qmd §1.3) requires a balanced fixed index set, so all three now
    route through here.
    """
    import torch

    lab = [int(v) for v in labels.tolist()]
    classes = sorted(set(lab))
    per, rem = divmod(Q, len(classes))
    if rem:
        raise ValueError(f"Q={Q} is not divisible by {len(classes)} classes — "
                         f"the val track must be class balanced")
    idx: list[int] = []
    for c in classes:
        hits = [i for i, v in enumerate(lab) if v == c][:per]
        if len(hits) < per:
            raise ValueError(f"class {c}: only {len(hits)} test images available, need {per}")
        idx.extend(hits)
    return sorted(idx)


def select_val_queries(test_ds, Q: int):
    """The val query set: ``ImageDataset`` of the balanced fixed indices."""
    import torch

    idx = torch.tensor(balanced_query_indices(test_ds.labels, Q), dtype=torch.long)
    return ImageDataset(test_ds.images[idx], test_ds.labels[idx])
