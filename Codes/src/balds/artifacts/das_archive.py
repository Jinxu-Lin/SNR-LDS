"""Readers for the DAS release archive (Diffusion Attribution Score, CIFAR-2).

The archive is an EXTERNAL artifact tree produced by someone else's code, so its
layout is a fact to be read, not a convention we own. This module is the single
place that knows that layout; everything above it receives numpy arrays and
python lists and never sees a path.

Layout (``root`` = the directory holding ``CIFAR2/`` and ``_code_DAS/``)::

    CIFAR2/saved/5000-0.5/
      grad/<readout>/<T>/ddpm-{train-keys-{0..4},gen-keys,val-keys}-<p>.npy
                          HEADERLESS float32 dumps (np.memmap of the producing
                          script), NOT .npy containers despite the extension.
                          Rows are (N, p) projected+timestep-averaged features
                          for every readout except `loss` (see `read_features`).
      error/error-<T>-train.pkl     list of (T, chunk) float32 -> concat axis 1
      lds-val/ddpm-sub-<m>-<j>/e-<e>-<track>.pkl
                          list of 4 (T=1000, chunk_q) float32 batches of the GT
                          loss; concat on axis 1 -> (timestep, query). `m` is the
                          subset, `j` the retrain replica, `e` the noise seed.
      gen/gen/<i>.png     the generated query images, i = 0..999
    _code_DAS/CIFAR2/data/indices/5000-0.5/
      idx-train.pkl       5000 ORIGINAL cifar10 train-split indices, in the order
                          the features were computed in (train-keys-0 = [0:1000])
      idx-val.pkl         1000 original cifar10 TEST-split indices, same role for
                          the val query track
      lds-val/sub-idx-<m>.pkl   2500 original train indices kept by subset m
      lds-test/sub-idx-<m>.pkl  the wrong-family control (a second draw of the
                          same generator; d1_migrate 2026-09-02 used it to prove
                          the regenerated lds-val family is the right one)

Provenance of the reading conventions: ``_code_DAS/CIFAR2/03_grad.py`` (feature
row order = dataloader order over ``dataset.select(idx)``, shuffle=False) and
``_Data/reports/d2_das_linear_2026-09-02/`` (GT axis convention: the four
pickled blocks are the BATCH dimension; averaging over queries instead of
timesteps drops LDS to .0046, indistinguishable from the control).
"""
from __future__ import annotations

import os
import pickle

import numpy as np

#: sub-tree holding the produced artifacts, relative to the archive root
SAVED_SUBDIR = os.path.join("CIFAR2", "saved", "5000-0.5")
#: sub-tree holding the index files that DEFINE the split
INDEX_SUBDIR = os.path.join("_code_DAS", "CIFAR2", "data", "indices", "5000-0.5")

#: repo feature name -> the archive's readout directory. The name on the left is
#: this project's readout vocabulary (balds.workflows.common._READOUTS); the name on
#: the right is DAS's own ``--loss_function_type``.
READOUT_DIRS: dict[str, str] = {
    "das": "mean",                          # mean(eps)
    "dtrak": "mean-squared-l2-norm",        # mean(eps^2) = L_Square (the real D-TRAK)
    "trak": "loss",                         # L_Simple (mse)
    "l1norm": "l1norm",                     # sum|eps|
}
#: how many 1000-row shards the train features are split into
TRAIN_SHARDS = 5


def resolve_root(path: str) -> str:
    """Normalise ``path`` to the archive root (the parent of ``CIFAR2/``).

    Accepts either the root itself or the ``CIFAR2/saved/5000-0.5`` sub-tree,
    because both are natural things to point at and the two index trees live on
    opposite sides of that boundary. Raises with the expected layout rather than
    failing later on a missing file.
    """
    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, "_code_DAS")) and os.path.isdir(os.path.join(path, "CIFAR2")):
        return path
    if os.path.isdir(os.path.join(path, "grad")) and os.path.isdir(os.path.join(path, "lds-val")):
        # pointed at CIFAR2/saved/5000-0.5 -> the root is three levels up
        cand = os.path.abspath(os.path.join(path, "..", "..", ".."))
        if os.path.isdir(os.path.join(cand, "_code_DAS")):
            return cand
    raise FileNotFoundError(
        f"{path} is not a DAS archive: expected either the root (holding "
        f"CIFAR2/ and _code_DAS/) or its CIFAR2/saved/5000-0.5 sub-tree")


def saved_dir(root: str) -> str:
    return os.path.join(root, SAVED_SUBDIR)


def index_dir(root: str) -> str:
    return os.path.join(root, INDEX_SUBDIR)


def _unpickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --- the split definition ----------------------------------------------------

def read_split_indices(root: str) -> tuple[list[int], list[int]]:
    """``(idx_train, idx_val)`` — ORIGINAL cifar10 split indices, in archive order.

    Order matters and is not sortable: feature row ``i`` is ``idx_train[i]``, so
    re-sorting these would silently permute every feature matrix against its
    labels and masks.
    """
    idx = index_dir(root)
    train = [int(i) for i in _unpickle(os.path.join(idx, "idx-train.pkl"))]
    val = [int(i) for i in _unpickle(os.path.join(idx, "idx-val.pkl"))]
    return train, val


def read_subset_indices(root: str, m: int, *, family: str = "lds-val") -> list[int]:
    """Original train indices kept by LDS subset ``m`` of ``family``."""
    path = os.path.join(index_dir(root), family, f"sub-idx-{int(m)}.pkl")
    return [int(i) for i in _unpickle(path)]


# --- features ----------------------------------------------------------------

def feature_path(root: str, readout: str, T: int, key: str, proj_dim: int) -> str:
    """Path of one headerless feature dump; ``key`` is ``train-0``.. or ``gen``/``val``."""
    stem = f"train-keys-{key.split('-', 1)[1]}" if key.startswith("train-") else f"{key}-keys"
    return os.path.join(saved_dir(root), "grad", READOUT_DIRS[readout], str(int(T)),
                        f"ddpm-{stem}-{int(proj_dim)}.npy")


def has_features(root: str, readout: str, T: int, key: str, proj_dim: int) -> bool:
    return os.path.isfile(feature_path(root, readout, T, key, proj_dim))


def read_feature_dump(root: str, readout: str, T: int, key: str, proj_dim: int) -> np.ndarray:
    """One feature dump as ``(rows, proj_dim)`` float32.

    ``rows`` is the sample count for every readout EXCEPT ``trak``/``loss``,
    whose dumps were produced by a modified script that keeps one row per
    (sample, timestep) — ``rows = n_samples * T`` there. Nothing here averages
    it: the caller has to say what the extra axis means, because guessing wrong
    silently mixes different images into one feature.
    """
    path = feature_path(root, readout, T, key, proj_dim)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DAS feature dump missing: {path}")
    a = np.fromfile(path, dtype=np.float32)
    p = int(proj_dim)
    if a.size % p:
        raise ValueError(f"{path}: {a.size} floats is not a multiple of p={p}")
    return a.reshape(-1, p)


# --- the per-timestep training error (DAS's e_n input) -----------------------

def read_error_matrix(root: str, T: int) -> np.ndarray:
    """``(T, N)`` float32 per-timestep per-sample training error.

    The pickle is a list of ``(T, chunk)`` blocks over the SAMPLE axis (DAS's own
    ``score.py`` does ``np.concatenate(..., axis=-1)`` then swaps axes), so the
    concatenation is on axis 1. Values are the raw per-timestep MSE; DAS takes
    the square root only at the very end of its own e_n chain.
    """
    blocks = _unpickle(os.path.join(saved_dir(root), "error", f"error-{int(T)}-train.pkl"))
    return np.concatenate([np.asarray(b, dtype=np.float32) for b in blocks], axis=1)


# --- ground-truth losses -----------------------------------------------------

def gt_loss_path(root: str, m: int, replica: int, e_seed: int, track: str) -> str:
    return os.path.join(saved_dir(root), "lds-val", f"ddpm-sub-{int(m)}-{int(replica)}",
                        f"e-{int(e_seed)}-{track}.pkl")


def read_gt_loss_block(root: str, m: int, replica: int, e_seed: int, track: str) -> np.ndarray:
    """``(n_timesteps, n_queries)`` float64 losses of one (subset, replica, seed).

    The pickle holds four blocks that are the BATCH dimension of the query loop,
    hence the axis-1 concatenation; axis 0 is the timestep grid the loss is
    averaged over. Getting this backwards is not loud — it produces a plausible
    matrix whose LDS is ~0 (d2 report §5).
    """
    blocks = _unpickle(gt_loss_path(root, m, replica, e_seed, track))
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], axis=1)


# --- generated query images --------------------------------------------------

def read_generated_images(root: str, n: int, *, folder: str = "gen") -> np.ndarray:
    """``(n, 3, 32, 32)`` uint8 CHW images ``gen/<folder>/{0..n-1}.png``.

    Indexed by filename, not by directory listing order: ``sorted(os.listdir)``
    puts ``10.png`` before ``2.png``, which would permute the query set against
    the query features.
    """
    from PIL import Image
    base = os.path.join(saved_dir(root), "gen", folder)
    out = np.zeros((int(n), 3, 32, 32), dtype=np.uint8)
    for i in range(int(n)):
        path = os.path.join(base, f"{i}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"generated query image missing: {path}")
        img = Image.open(path).convert("RGB")
        if img.size != (32, 32):
            raise ValueError(f"{path}: expected 32x32, got {img.size}")
        out[i] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    return out


def count_generated_images(root: str, *, folder: str = "gen") -> int:
    """How many consecutively-numbered PNGs ``gen/<folder>/`` holds from 0."""
    base = os.path.join(saved_dir(root), "gen", folder)
    n = 0
    while os.path.isfile(os.path.join(base, f"{n}.png")):
        n += 1
    return n


# --- the trained main model --------------------------------------------------

def model_paths(root: str, seed: int) -> dict[str, str]:
    """The three files of ``ddpm/ddpm_<seed>`` (a diffusers ``DDPMPipeline`` directory)."""
    base = os.path.join(saved_dir(root), "ddpm", f"ddpm_{int(seed)}")
    return {"unet_config": os.path.join(base, "unet", "config.json"),
            "unet_weights": os.path.join(base, "unet", "diffusion_pytorch_model.bin"),
            "scheduler_config": os.path.join(base, "scheduler", "scheduler_config.json")}


def read_ddpm_model(root: str, seed: int) -> dict:
    """``{"unet_config", "scheduler_config", "unet_state", "paths"}`` of the archive model.

    ``unet_state`` is the state dict exactly as saved (CPU; diffusers-0.16 key
    names, see ``balds.models.das_ddpm``).
    """
    import json

    import torch

    paths = model_paths(root, seed)
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"DAS archive model files missing: {missing}")
    with open(paths["unet_config"]) as fh:
        unet_config = json.load(fh)
    with open(paths["scheduler_config"]) as fh:
        scheduler_config = json.load(fh)
    state = torch.load(paths["unet_weights"], map_location="cpu", weights_only=True)
    return {"unet_config": unet_config, "scheduler_config": scheduler_config,
            "unet_state": state, "paths": paths}
