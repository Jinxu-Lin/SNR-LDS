"""Image-space baselines (pixel / CLIP) on the SD3.5 latent platform — §6.2-28 ①.

The latent platform's pipeline datasets are VAE latents; the embedders are
defined on IMAGES. Wiring them therefore means giving ``_embed`` a second image
source, and the one thing that can go silently wrong is ORDER: if the raw folder
enumerates the training set differently from the latent cache, the baseline
attributes a permutation of the training set and every number downstream is
wrong without anything raising.

T1 the three splits produce features of the right shape on a stub latent platform
T2 the train side is in the LATENT CACHE's order, proven with a fixture whose
   images encode their own index (and a deliberately permuted cache must raise)
T3 the val track takes the balanced fixed indices of the raw test split
T4 the gen track uses the generation's stored decoded previews (images_u8)
T5 pixel is downsampled per ``featurize.pixel_resize_by_dataset`` and the sidecar
   meta records it; CIFAR keeps the literal pixels (resize=None), byte-identical
T6 CLIP refuses a second resize rather than ignoring it
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config, resolve_pixel_resize  # noqa: E402
from balds.workflows.features import FeaturizeUseCase  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.attribution.embed import embed  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_emb_toy"
H = 8                     # stand-in for 256; resized to R, like 256 -> 64
R = 4
NTRAIN, NTEST, Q = 6, 4, 2
CLASSES = [4, 9]

#: train label order: interleaved, so a within-class reordering of the raw
#: folder relative to the latent cache would still change the label vector.
TRAIN_LABELS = torch.tensor([4, 9, 4, 9, 4, 9])
TEST_LABELS = torch.tensor([4, 4, 9, 9])


def _value(i: int, n: int) -> float:
    """Image i is a constant image; its value identifies the index."""
    return -1.0 + 2.0 * i / n


class _LazyConstImages:
    """Path-free stand-in for ``LazyImageFolder``: constant images that encode
    their own position, decoded on access like the real 256px loader."""

    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (torch.full((3, H, H), _value(int(idx), len(self.labels))),
                int(self.labels[idx]))


def _toy_loader(data_dir, **_):
    return _LazyConstImages(TRAIN_LABELS), _LazyConstImages(TEST_LABELS)


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


def _env(*, permute_cache: bool = False):
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                            code_version="test")
    cfg = load_config({"storage.data_root": f"{root}/data"})
    cfg["datasets"][_DS] = {"classes": CLASSES, "num_classes": 10, "platform": "sd35_lora"}
    cfg["featurize"]["pixel_resize_by_dataset"] = {_DS: R}
    cfg["featurize"]["batch_size"] = 2
    cfg["lds"]["Q"] = Q
    spec = RunSpec(dataset=_DS)
    train_lab = TRAIN_LABELS.clone()
    if permute_cache:                       # the cache disagrees with the folder
        train_lab = train_lab.flip(0)
    store.save(K.LATENTS, spec, {"latents": torch.zeros(NTRAIN, 16, 2, 2),
                                 "labels": train_lab}, split="train")
    store.save(K.LATENTS, spec, {"latents": torch.zeros(NTEST, 16, 2, 2),
                                 "labels": TEST_LABELS}, split="test")
    base = RunSpec(dataset=_DS, process="cfm", seed=42, conditional=True)
    # the generation: latents + the decoded previews the gen track embeds
    gen_imgs = torch.stack([torch.full((3, H, H), _value(i, Q)) for i in range(Q)])
    store.save(K.GENERATION, base, {
        "samples": torch.zeros(Q, 16, 2, 2),
        "images_u8": ((gen_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8),
        "labels": torch.tensor(CLASSES), "Q": Q, "gen_seed": 165, "ode_steps": 3,
        "conditional": True, "platform": "sd35_lora"})
    return store, cfg, base


def test_t1_t2_t5_train_side_is_in_the_latent_cache_order():
    store, cfg, base = _env()
    uc = FeaturizeUseCase(store, cfg, device="cpu")
    out = uc.run(_DS, 42, feat="pixel", split="train")
    assert out["train_features"] == [NTRAIN, 3 * R * R], out
    assert out["pixel_resize"] == R

    F = store.load(K.TRAIN_FEATURES, base, feat="pixel")
    # row i must be the constant image at position i of the raw folder, i.e. the
    # same position the latent cache holds -- a permutation would shuffle these
    for i in range(NTRAIN):
        want = (_value(i, NTRAIN) / 2 + 0.5)
        assert torch.allclose(F[i], torch.full((3 * R * R,), want), atol=1e-6), i

    meta = store.load(K.FEATURE_META, base, feat="pixel")
    assert meta["pixel_resize"] == R and meta["kind"] == "embedding"
    assert meta["dim"] == 3 * R * R and meta["platform"] == "sd35_lora"


def test_t2b_a_permuted_latent_cache_is_refused():
    store, cfg, base = _env(permute_cache=True)
    uc = FeaturizeUseCase(store, cfg, device="cpu")
    with pytest.raises(ValueError, match="do not line up"):
        uc.run(_DS, 42, feat="pixel", split="train")
    assert not store.exists(K.TRAIN_FEATURES, base, feat="pixel")


def test_t3_val_track_takes_the_balanced_fixed_indices():
    store, cfg, base = _env()
    uc = FeaturizeUseCase(store, cfg, device="cpu")
    out = uc.run(_DS, 42, feat="pixel", split="query", query_type="val")
    assert out["query_val"] == [Q, 3 * R * R]
    qspec = base.with_(query_type="val")
    F = store.load(K.QUERY_FEATURES, qspec, feat="pixel")
    # labels [4,4,9,9], Q=2 -> one per class, in the split's natural order: 0 and 2
    for row, i in enumerate((0, 2)):
        want = (_value(i, NTEST) / 2 + 0.5)
        assert torch.allclose(F[row], torch.full((3 * R * R,), want), atol=1e-6)


def test_t4_gen_track_uses_the_stored_decoded_previews():
    store, cfg, base = _env()
    uc = FeaturizeUseCase(store, cfg, device="cpu")
    out = uc.run(_DS, 42, feat="pixel", split="query", query_type="gen")
    assert out["query_gen"] == [Q, 3 * R * R]
    F = store.load(K.QUERY_FEATURES, base.with_(query_type="gen"), feat="pixel")
    gen = store.load(K.GENERATION, base)
    want = embed("pixel", gen["images_u8"].float() / 127.5 - 1.0, device="cpu",
                 batch_size=2, resize=R)
    assert torch.equal(F, want)


def test_t1b_clip_runs_on_the_latent_platform_and_is_chunk_invariant():
    store, cfg, base = _env()
    uc = FeaturizeUseCase(store, cfg, device="cpu")
    out = uc.run(_DS, 42, feat="clip", split="both", query_type="gen")
    assert out["train_features"][0] == NTRAIN and out["train_features"][1] == 512
    assert out["query_gen"] == [Q, 512] and out["pixel_resize"] is None
    F = store.load(K.TRAIN_FEATURES, base, feat="clip")
    # chunking must not change the numbers: the same images embedded in one call
    imgs = torch.stack([_LazyConstImages(TRAIN_LABELS)[i][0] for i in range(NTRAIN)])
    assert torch.equal(F, embed("clip", imgs, device="cpu", batch_size=2))


def test_t5b_cifar_keeps_the_literal_pixels():
    cfg = load_config({})
    assert resolve_pixel_resize(cfg, "cifar2_5k") is None
    assert resolve_pixel_resize(cfg, "cifar10_v2") is None
    assert resolve_pixel_resize(cfg, "artbench2_256") == 64
    imgs = torch.rand(3, 3, 32, 32) * 2 - 1
    assert embed("pixel", imgs, device="cpu").shape == (3, 3072)
    # resize=None is bit-identical to the pre-seam behaviour (flatten of [0,1])
    assert torch.equal(embed("pixel", imgs, device="cpu"),
                       (imgs.float() / 2 + 0.5).clamp(0, 1).reshape(3, -1))


def test_t6_clip_refuses_a_second_resize():
    with pytest.raises(ValueError, match="own input resize"):
        embed("clip", torch.zeros(1, 3, 32, 32), device="cpu", resize=64)


def test_t5c_area_downsample_is_the_exact_box_filter():
    """256 -> 64 is an integer factor, so the resize is an exact 4x4 mean; it
    must not depend on anti-aliasing flags or backend."""
    x = torch.rand(2, 3, 8, 8) * 2 - 1
    got = embed("pixel", x, device="cpu", resize=4).reshape(2, 3, 4, 4)
    unit = (x.float() / 2 + 0.5).clamp(0, 1)
    want = unit.reshape(2, 3, 4, 2, 4, 2).mean(dim=(3, 5))
    assert torch.allclose(got, want, atol=1e-6)
    assert np.isfinite(got.numpy()).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
