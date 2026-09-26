"""Assimilate the DAS release archive into this library (§6.2-33 ⑦).

The DAS CIFAR-2 release ships everything an LDS evaluation needs — projected
gradient features for four readouts, 128 subset masks, and 128 x 3 retrains x 3
noise seeds of ground-truth losses — computed by *their* code on *their* DDPM.
This module registers that as the dataset key ``cifar2_das`` and copies the
archive into this library's own artifact addresses, so every downstream command
(``balds score`` / ``evaluate`` / the mechanism batteries) works on it unchanged.

**It is an import, not a re-derivation.** Nothing here recomputes a gradient or
a loss: the numbers are DAS's, only their addresses and container formats become
ours. Two consequences worth stating once:

* The archive's process is a DDPM (linear beta 1e-4..0.02, T=1000, epsilon
  prediction) trained UNCONDITIONALLY. Artifacts are filed under
  ``process="ddpm"``, and — per TASK_DAS_IMPORT — under the default conditional
  identity, so ``generations/cifar2_das/ddpm_cond/...`` reads "cond" for a model
  that has no classes. That segment is the only place ``conditional`` shows up
  for this platform (featurize/scores/GT paths do not encode it), and taking the
  default keeps every CLI invocation flag-free. The stored labels are zeros, the
  unconditional convention, so no reader mistakes them for class information.
* Q is whatever the archive holds (1000 per track), via ``resolve_Q``. See the
  ``lds.Q_by_dataset`` note in defaults.yaml for why the global 100 is wrong
  here. At any smaller Q the gen track takes the first Q rows and the val track
  takes ``balanced_query_indices`` — the same selection ``select_val_queries``
  would make — so the two stay consistent with this repo's own caliber.

Verified on import day (2026-09-04): the identity-row LDS computed from these
artifacts reproduces ``_Data/reports/d2_das_linear_2026-09-02`` to |delta| <= 1e-4
on all six (readout x track) cells, with the same argmax lambda.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.logging import get_logger, set_context
from balds.schema.registry import DATASETS
from balds.schema.runspec import RunSpec
from balds.data.base import balanced_query_indices
from balds.data.cifar import load_cifar_by_index
from balds.models.das_ddpm import PLATFORM, build_das_ddpm, verify_scheduler_config
from balds.artifacts import das_archive as da
from balds.artifacts import sha256_file

from .config import resolve_Q

log = get_logger("balds.das_import")

#: The dataset key. Its own product tree, isolated from cifar2/cifar2_5k by
#: construction (different N, different split, someone else's model).
DATASET = "cifar2_das"
CLASSES = (1, 7)
PROCESS = "ddpm"
SEED = 42
#: The archive caliber this importer reads. All four are archive facts, not
#: knobs: p=4096 is the only projection every readout has, T=100 is the paper's
#: projected-family grid, and 128 x 3 x 3 is what ``lds-val/`` contains for
#: EVERY subset (replicas 3 and 4 exist for m < 64 only, so including them would
#: average unevenly across subsets — d2 used 0..2 for the same reason).
PROJ_DIM = 4096
T_FEAT = 100
T_ERROR = 100
M_SUBSETS = 128
REPLICAS = (0, 1, 2)
E_SEEDS = (0, 1, 2)

#: Readouts imported by default. ``trak`` (DAS's ``loss`` = L_Simple) is NOT in
#: this list: its dumps are (n_samples * T, p) rather than (n_samples, p), and
#: the shipped ``03_grad.py`` cannot have produced them, so the meaning of the
#: extra axis is undocumented. See :data:`TRAK_CAVEAT`.
DEFAULT_FEATS: tuple[str, ...] = ("das", "dtrak", "l1norm")
IMPORTABLE_FEATS: tuple[str, ...] = ("das", "dtrak", "trak", "l1norm")

TRAK_CAVEAT = (
    "the `loss` (L_Simple/TRAK) dumps carry n_samples*T rows, an axis the shipped "
    "03_grad.py never writes; reading it as sample-major and averaging over T gives "
    "gen LDS .0312 (vs .0068 for the time-major reading and .0015-.0131 for the "
    "wrong-family control), which is suggestive but an order of magnitude below this "
    "project's own trak_T100 (.2265). Row norms also do not match per-timestep "
    "normalised gradients. Imported only on explicit request, and the meta records "
    "the assumption."
)


# --------------------------------------------------------------------------- #
# the dataset key
# --------------------------------------------------------------------------- #
def _load_cifar2_das(data_dir: str, *, raw_dirs: Optional[dict] = None, **_):
    """``(train, test)`` for ``cifar2_das``: DAS's own 5000 / 1000 index lists.

    ``data_dir`` is the CIFAR cache (the pixels); the split itself lives in the
    DAS archive, so this loader needs ``raw_dirs`` — the app layer's ``_load_ds``
    passes it. Order is the archive's, never sorted: feature row ``i`` IS
    ``idx_train[i]``.
    """
    root = (raw_dirs or {}).get("das_archive")
    if not root:
        raise ValueError(
            f"{DATASET} is defined by index files inside the DAS archive, so it needs "
            f"datasets.raw_dirs.das_archive alongside the CIFAR cache. Load it through "
            f"balds.workflows.common._load_ds (which passes raw_dirs), not get_dataset(name, dir).")
    idx_train, idx_val = da.read_split_indices(da.resolve_root(root))
    return load_cifar_by_index(data_dir, idx_train, idx_val, classes=list(CLASSES))


DATASETS.add(DATASET, _load_cifar2_das)


# --------------------------------------------------------------------------- #
# the importer
# --------------------------------------------------------------------------- #
class DasImportUseCase:
    """Write the DAS archive into this library's artifact addresses (idempotent)."""

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg = store, cfg          # no compute: device is unused

    # -- helpers ------------------------------------------------------------
    def _spec(self, **changes) -> RunSpec:
        return RunSpec(dataset=DATASET, process=PROCESS, seed=SEED,
                       conditional=True, **changes)

    def _skip(self, kind, spec, force: bool, **key) -> bool:
        return self.store.exists(kind, spec, **key) and not force

    def _reduce_rows(self, feat: str, a: np.ndarray, *, n_expected: int) -> np.ndarray:
        """Collapse a raw dump to one row per sample."""
        if a.shape[0] == n_expected:
            return a
        if feat == "trak" and a.shape[0] == n_expected * T_FEAT:
            # sample-major (row = sample*T + t); see TRAK_CAVEAT
            return a.reshape(n_expected, T_FEAT, a.shape[1]).mean(axis=1)
        raise ValueError(
            f"{feat}: dump has {a.shape[0]} rows, expected {n_expected} "
            f"(or {n_expected * T_FEAT} for the per-timestep `loss` layout)")

    # -- stages -------------------------------------------------------------
    def _import_masks(self, root: str, idx_train, spec, force: bool):
        if self._skip(K.SUBSET_MASKS, spec, force):
            return "skipped"
        pos = {g: i for i, g in enumerate(idx_train)}
        masks = []
        for m in range(M_SUBSETS):
            z = np.zeros(len(idx_train), dtype=np.bool_)
            keep = da.read_subset_indices(root, m)
            missing = [g for g in keep if g not in pos]
            if missing:
                raise ValueError(f"subset {m} keeps {len(missing)} indices absent from "
                                 f"idx-train.pkl (first: {missing[:3]}) — the index "
                                 f"trees do not belong to the same run")
            z[[pos[g] for g in keep]] = True
            masks.append(z)
        self.store.save(K.SUBSET_MASKS, spec, masks)
        log.info("masks: %d x %d, keep/row %d", len(masks), masks[0].size, int(masks[0].sum()))
        return [len(masks), int(masks[0].size), int(masks[0].sum())]

    def _import_features(self, root: str, feat: str, spec, n_train: int,
                         qsel: dict, q_rows: dict, force: bool) -> dict:
        """Train + per-track query features of one readout, under ``<feat>_T100``."""
        name = f"{feat}_T{T_FEAT}"
        out: dict = {}
        tracks = [t for t in qsel if da.has_features(root, feat, T_FEAT, t, PROJ_DIM)]

        if self._skip(K.TRAIN_FEATURES, spec, force, feat=name):
            out["train"] = "skipped"
        else:
            per_shard = n_train // da.TRAIN_SHARDS
            rows = np.concatenate(
                [self._reduce_rows(feat, da.read_feature_dump(root, feat, T_FEAT,
                                                              f"train-{i}", PROJ_DIM),
                                   n_expected=per_shard)
                 for i in range(da.TRAIN_SHARDS)], axis=0)
            if rows.shape[0] != n_train:
                raise ValueError(f"{name}: {rows.shape[0]} train rows, expected {n_train}")
            self.store.save(K.TRAIN_FEATURES, spec, torch.from_numpy(rows).float(), feat=name)
            out["train"] = list(rows.shape)

        for track in qsel:
            qspec = spec.with_(query_type=track)
            if track not in tracks:
                out[f"query_{track}"] = "absent from archive"
                continue
            if self._skip(K.QUERY_FEATURES, qspec, force, feat=name):
                out[f"query_{track}"] = "skipped"
                continue
            raw = da.read_feature_dump(root, feat, T_FEAT, track, PROJ_DIM)
            rows = self._reduce_rows(feat, raw, n_expected=q_rows[track])[qsel[track]]
            self.store.save(K.QUERY_FEATURES, qspec, torch.from_numpy(rows).float(), feat=name)
            out[f"query_{track}"] = list(rows.shape)

        if self._skip(K.FEATURE_META, spec, force, feat=name):
            out["meta"] = "skipped"
            return out
        # provenance is hashed only when the sidecar is actually (re)written —
        # hashing the `loss` dumps costs ~10 GB of reads
        keys = [f"train-{i}" for i in range(da.TRAIN_SHARDS)] + tracks
        meta = {"source": "DAS release archive", "archive_root": root, "readout": feat,
                "archive_readout_dir": da.READOUT_DIRS[feat], "T": T_FEAT,
                "proj_dim": PROJ_DIM, "imported_by": "balds import-das",
                "row_order": "idx-train.pkl (train) / idx-val.pkl (val) / gen png index (gen)",
                "query_index": {t: [int(i) for i in qsel[t]] for t in tracks},
                "source_sha256": {k: sha256_file(da.feature_path(root, feat, T_FEAT,
                                                                 k, PROJ_DIM))
                                  for k in keys}}
        if feat == "trak":
            meta["row_layout"] = "sample_major(n,T,p).mean(T)  [UNVERIFIED]"
            meta["caveat"] = TRAK_CAVEAT
        self.store.save(K.FEATURE_META, spec, meta, feat=name)
        out["meta"] = list(meta["source_sha256"])
        return out

    def _import_error_weight(self, root: str, spec, n_train: int, force: bool):
        # e_n is readout-independent and every consumer loads it from feat="das"
        # (pipeline._EN_FEAT), so it goes there, NOT into the _T100 directories.
        if self._skip(K.ERROR_WEIGHT, spec, force, feat="das"):
            return "skipped"
        errors = da.read_error_matrix(root, T_ERROR).T           # (N, T) raw per-timestep MSE
        if errors.shape[0] != n_train:
            raise ValueError(f"error matrix has {errors.shape[0]} samples, expected {n_train}")
        # This repo's e_n caliber (horizontal/error_weight.compute_error_weight):
        # sqrt -> L2-normalise across timesteps -> mean. DAS's own score.py does
        # the sqrt LAST instead; the artifact kind's caliber is this repo's, so
        # an imported e_n means the same thing as every other error_train.npy.
        errors = np.sqrt(errors)
        norms = np.linalg.norm(errors, axis=1, keepdims=True)
        e_n = (errors / (norms + 1e-8)).mean(axis=1).astype(np.float32)
        self.store.save(K.ERROR_WEIGHT, spec, e_n, feat="das")
        return list(e_n.shape)

    def _import_ground_truth(self, root: str, spec, qsel: dict, force: bool) -> dict:
        """Per-noise-seed GT_LOSSES (replica-averaged) + the GT_MATRIX over them.

        Splitting it this way is what makes the import consistent with the rest
        of the pipeline: ``balds subsets gt --e-seeds 0,1,2`` re-derives exactly
        this GT_MATRIX from the stored GT_LOSSES, because both are unweighted
        means over the same 3 x 3 cells.
        """
        out: dict = {}
        for track, sel in qsel.items():
            tspec = spec.with_(query_type=track)
            mats = []
            for e in E_SEEDS:
                if self._skip(K.GT_LOSSES, tspec, force, eseed=e):
                    mats.append(self.store.load(K.GT_LOSSES, tspec, eseed=e))
                    out[f"{track}_e{e}"] = "skipped"
                    continue
                mat = np.zeros((M_SUBSETS, len(sel)))
                for m in range(M_SUBSETS):
                    per_replica = [da.read_gt_loss_block(root, m, j, e, track).mean(axis=0)
                                   for j in REPLICAS]
                    mat[m] = np.mean(per_replica, axis=0)[sel]
                self.store.save(K.GT_LOSSES, tspec, mat, eseed=e)
                mats.append(mat)
                out[f"{track}_e{e}"] = list(mat.shape)
                log.info("gt_losses %s eseed=%d: %s (mean over %d replicas)",
                         track, e, mat.shape, len(REPLICAS))
            if self._skip(K.GT_MATRIX, tspec, force):
                out[f"{track}_matrix"] = "skipped"
            else:
                gt = np.mean(mats, axis=0)
                self.store.save(K.GT_MATRIX, tspec, gt)
                out[f"{track}_matrix"] = list(gt.shape)
        return out

    def _import_generations(self, root: str, spec, Q: int, force: bool):
        if self._skip(K.GENERATION, spec, force):
            return "skipped"
        u8 = da.read_generated_images(root, Q)                   # (Q, 3, 32, 32) uint8
        images_u8 = torch.from_numpy(u8)
        samples = images_u8.float() / 127.5 - 1.0                # -> [-1, 1]
        self.store.save(K.GENERATION, spec, {
            "samples": samples, "images_u8": images_u8,
            "labels": torch.zeros(Q, dtype=torch.long),          # unconditional model
            "Q": Q, "gen_seed": None, "ode_steps": None, "conditional": True,
            "source": "DAS release archive gen/gen/{i}.png",
            # gen_seed/ode_steps are None on purpose: the archive keeps the PNGs
            # but not the sampler arguments that made them, and inventing a
            # plausible value is how a replay would silently attribute the wrong
            # images. Anything that needs them (trajectory capture, the
            # counterfactual chain) must fail loudly rather than guess.
        })
        return [Q, list(samples.shape)]

    def import_model(self, root: Optional[str] = None, *, force: bool = False):
        """File ``ddpm/ddpm_42`` as the CHECKPOINT the curvature path loads.

        Same identity as every other artifact of this platform
        (``checkpoints/cifar2_das/ddpm_cond/seed_42/final.pt``); the content says
        ``conditional: False`` because the model has no classes. The weights are
        the archive's, unconverted — ``build_das_ddpm`` renames the legacy
        attention keys at load time.
        """
        root = da.resolve_root(root or self.cfg["datasets"]["raw_dirs"]["das_archive"])
        spec = self._spec()
        if self._skip(K.CHECKPOINT, spec, force):
            return "skipped"
        src = da.read_ddpm_model(root, SEED)
        scheduler = verify_scheduler_config(src["scheduler_config"])
        ckpt = {
            "platform": PLATFORM, "model_type": PROCESS, "conditional": False,
            "unet_config": src["unet_config"],
            "unet_state": {k: (v.float() if v.is_floating_point() else v).cpu()
                           for k, v in src["unet_state"].items()},
            "scheduler_config": src["scheduler_config"],
            "source": {"archive_root": root,
                       "paths": {k: p[len(root):].lstrip("/") for k, p in src["paths"].items()},
                       "sha256": {k: sha256_file(p) for k, p in src["paths"].items()}},
            "step": None,
        }
        # strict load under the installed diffusers before anything is filed, so a
        # key/shape mismatch fails here rather than at the first GPU fit
        n_params = sum(p.numel() for p in build_das_ddpm(ckpt, device="cpu").parameters())
        self.store.save(K.CHECKPOINT, spec, ckpt)
        log.info("model: %d tensors, %d parameters, scheduler %s",
                 len(ckpt["unet_state"]), n_params, scheduler)
        return {"tensors": len(ckpt["unet_state"]), "n_params": n_params,
                "scheduler": scheduler}

    # -- entry point --------------------------------------------------------
    def run(self, root: Optional[str] = None, *, force: bool = False,
            feats: Optional[Sequence[str]] = None, model: bool = False) -> dict:
        root = da.resolve_root(root or self.cfg["datasets"]["raw_dirs"]["das_archive"])
        feats = tuple(feats) if feats else DEFAULT_FEATS
        bad = [f for f in feats if f not in IMPORTABLE_FEATS]
        if bad:
            raise ValueError(f"unknown readout(s) {bad}; choose from {list(IMPORTABLE_FEATS)}")
        if "trak" in feats:
            log.warning("importing `trak` on request: %s", TRAK_CAVEAT)

        spec = self._spec()
        set_context(run_id=spec.digest()[:6], trace_id=f"import-das:{DATASET}")
        idx_train, idx_val = da.read_split_indices(root)
        n_train, n_val = len(idx_train), len(idx_val)
        Q = resolve_Q(self.cfg, DATASET)

        # Which archive rows become this platform's queries. gen mirrors
        # GenerateUseCase (the first Q samples); val mirrors select_val_queries
        # (class-balanced fixed indices) so a later re-featurization of the val
        # track would land on the SAME images these features describe. At the
        # archive's own Q both are the identity.
        n_gen = da.count_generated_images(root)
        if Q > min(n_gen, n_val):
            raise ValueError(f"lds.Q_by_dataset.{DATASET}={Q} exceeds the archive "
                             f"({n_gen} gen / {n_val} val queries)")
        if Q == n_val:
            val_sel = np.arange(n_val)
        else:
            _, test_ds = _load_cifar2_das(self.cfg["datasets"]["raw_dirs"]["cifar"],
                                          raw_dirs=self.cfg["datasets"]["raw_dirs"])
            val_sel = np.asarray(balanced_query_indices(test_ds.labels, Q))
        qsel = {"gen": np.arange(Q), "val": val_sel}
        q_rows = {"gen": n_gen, "val": n_val}
        log.info("importing DAS archive %s: N=%d, Q=%d/track, feats=%s",
                 root, n_train, Q, list(feats))

        out: dict = {"root": root, "dataset": DATASET, "N": n_train, "Q": Q,
                     "feats": list(feats)}
        out["masks"] = self._import_masks(root, idx_train, spec, force)
        out["features"] = {f: self._import_features(root, f, spec, n_train, qsel,
                                                    q_rows, force)
                           for f in feats}
        out["error_weight"] = self._import_error_weight(root, spec, n_train, force)
        out["ground_truth"] = self._import_ground_truth(root, spec, qsel, force)
        out["generation"] = self._import_generations(root, spec, Q, force)
        if model:
            out["model"] = self.import_model(root, force=force)
        return out
