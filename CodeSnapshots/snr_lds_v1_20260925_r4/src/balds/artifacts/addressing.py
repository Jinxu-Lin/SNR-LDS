"""Convention-over-configuration path derivation.

Turns ``(ArtifactKind, RunSpec, **key)`` into the *human-mirror* relative path,
**identical to the legacy ``_Data/`` layout** so existing artifacts on the
remote tree are addressable with zero migration. The content-addressed blob
path (``cas/<hash[:2]>/<hash>``) is also defined here.

This module is the only place that knows the on-disk naming convention; nothing
else in the codebase constructs a path.
"""
from __future__ import annotations

import re

from balds.schema.artifact import ArtifactKind
from balds.schema.runspec import RunSpec
from balds.schema.identity import validate_run_token


def model_segment(spec: RunSpec) -> str:
    """Full-model checkpoint/generation sub-directory, e.g. ``"cfm_cond"``."""
    return f"{spec.process}_{'cond' if spec.conditional else 'uncond'}"


def model_key(spec: RunSpec) -> str:
    """Short process key used in subset/GT artifact names: ``fm`` (CFM) or ``ddpm``."""
    return "fm" if spec.process == "cfm" else "ddpm"


def _ds_qt(spec: RunSpec) -> str:
    """Dataset segment with the query-track suffix when one is required."""
    if spec.query_type == "inject":
        return f"{spec.dataset}_inject"
    return f"{spec.dataset}_val" if spec.is_val else spec.dataset


def _proj_segment(key: dict) -> str:
    """Projection-identity sub-path for attribution artifacts.

    Features (and therefore the scores computed from them) produced at a
    different projection dimension or seed are DIFFERENT artifacts, not
    interchangeable ones — but the path used to carry no trace of either. That
    was invisible while only one p existed; the moment HP-p sweeps
    p in {1024, 4096, 16384, 32768}, all four write to the same files, the last
    one wins, and whatever is left on disk is silently consumed downstream as if
    it were the chosen p.

    Empty unless a ``proj`` key is passed, so every existing path (and the
    post-decision production run, which uses the configured default) is
    byte-identical. HP-p passes it explicitly for each rung.
    """
    return f"/{key['proj']}" if key.get("proj") else ""


def _attr_process_suffix(spec: RunSpec) -> str:
    """Process sub-path for attribution artifacts (featurize/scores).

    Empty for CFM so every legacy path stays byte-identical (zero migration);
    non-CFM processes get their own segment so e.g. matched-DDPM features can
    never overwrite the CFM ones (audit 2026-07-07 gap #2).
    """
    return "" if spec.process == "cfm" else f"/{model_key(spec)}"


#: Block-partial filenames under ``blocks/``: ``<stem>_q{q0}_{q1}[_r{r0}_{r1}].npz``.
BLOCK_STEM: dict[ArtifactKind, str] = {
    ArtifactKind.SCORES_BLOCK: "scores",
    ArtifactKind.REPEAT_SCORES_BLOCK: "repeat_scores",
}


def _row_segment(key: dict) -> str:
    """Train-row range of a row-sharded block partial (EKFAC-ROWSHARD, 2026-09-09).

    A block partial is addressed by TWO half-open ranges: the query range
    ``[q0, q1)`` (EKFAC-SHARD) and, one level below it, the train-row range
    ``[r0, r1)`` so that a single query block's N-row train stream can itself be
    split across cards. Absent = the whole ``[0, N)`` row range, i.e. exactly
    the filename EKFAC-SHARD wrote — every partial already on disk keeps a
    byte-identical path and stays readable (zero migration).
    """
    if key.get("r0") is None:
        return ""
    return f"_r{int(key['r0'])}_{int(key['r1'])}"


def _inject_cf_segment(key: dict) -> str:
    if not key.get("cf"):
        return ""
    value = key["cf"]
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("INJECT counterfactual selector must match '[A-Za-z0-9_-]+'")
    return f"/cf_{value}"


def _config_segment(key: dict) -> str:
    """Optional method-configuration identity for direct E4 baselines."""
    value = key.get("config")
    if value is None:
        return ""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("method config tag must match '[A-Za-z0-9_-]+'")
    return f"/cfg_{value}"


def block_dir(kind: ArtifactKind, spec: RunSpec, **key) -> str:
    """Directory holding one cell's block partials of ``kind``.

    Range selectors are dropped, so a full block key can be handed straight in:
    every partial of the cell shares this one directory regardless of how the
    query and row axes were tiled.
    """
    if kind not in BLOCK_STEM:
        raise ValueError(f"{kind} has no per-query-block partials")
    cell = {k: v for k, v in key.items() if k not in ("q0", "q1", "r0", "r1")}
    return relpath(kind, spec, q0=0, q1=0, **cell).rsplit("/", 1)[0]


def parse_block_key(kind: ArtifactKind, filename: str) -> dict | None:
    """``{"q0", "q1"[, "r0", "r1"]}`` for a block filename, ``None`` if not one.

    The inverse of :func:`relpath` for block partials — the only sanctioned way
    to read a query/row range back off the disk (nothing else parses a path).
    A whole-rows partial yields no ``r0``/``r1``, which is exactly the key that
    reproduces its path.
    """
    if kind not in BLOCK_STEM:
        raise ValueError(f"{kind} has no per-query-block partials")
    m = re.fullmatch(rf"{re.escape(BLOCK_STEM[kind])}_q(\d+)_(\d+)"
                     rf"(?:_r(\d+)_(\d+))?\.npz", filename)
    if m is None:
        return None
    out = {"q0": int(m.group(1)), "q1": int(m.group(2))}
    if m.group(3) is not None:
        out.update(r0=int(m.group(3)), r1=int(m.group(4)))
    return out


def relpath(kind: ArtifactKind, spec: RunSpec, **key) -> str:
    """Return the human-mirror path of an artifact, relative to ``data_root``.

    ``key`` carries kind-specific selectors:
    ``feat`` (featurize dirs), ``lam`` (per-lambda scores), ``step`` (checkpoint
    step), ``eseed`` (ground-truth ensemble seed), ``name`` (override filename),
    ``replica`` (noise-floor duplicate subset run, [m] 2026-08-09), ``proj``
    (projection identity for HP-p, e.g. ``"p1024s0"``), ``chain`` (GT losses of
    a subset chain whose seed differs from the query seed, §6.2-14 2026-08-17),
    ``q0``/``q1`` (half-open query range of a per-query-block score partial),
    ``r0``/``r1`` (half-open train-row range of a row-sharded block partial).
    ``replica``/``proj``/``chain``/``r0``/``r1`` absent keeps every path
    byte-identical to before they existed.
    """
    K = ArtifactKind
    seed = f"seed_{spec.seed}"
    model = model_segment(spec)
    rep = f"_r{key['replica']}" if key.get("replica") else ""   # replica suffix
    if spec.query_type == "inject" and kind in {
            K.GT_MATRIX, K.GT_LOSSES, K.GT_LOSSROW, K.LDS_RESULT}:
        raise ValueError(f"{kind.value} has no artifact on the inject query track")

    if kind == K.CHECKPOINT:
        name = key.get("name") or (f"step_{key['step']}" if "step" in key else "final")
        return f"checkpoints/{spec.dataset}/{model}/{seed}/{name}.pt"
    if kind == K.SUBSET_CHECKPOINT:
        return (f"checkpoints/subsets/{model_key(spec)}/{spec.dataset}/{seed}{rep}/"
                f"subset_{key['m']}/final.pt")
    if kind == K.LOSS_HISTORY:
        return f"checkpoints/{spec.dataset}/{model}/{seed}/loss_history.json"
    if kind == K.GENERATION:
        name = key.get("name", "samples")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("generation artifact name must match '[A-Za-z0-9_-]+'")
        return f"generations/{spec.dataset}/{model}/{seed}/{name}.pt"
    if kind == K.GEN_POOL:
        tag = key["tag"]
        if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", tag) is None:
            raise ValueError("pool tag must match '[A-Za-z0-9_-]+'")
        cf = _inject_cf_segment(key)
        return f"generations/{spec.dataset}/{model}/{seed}{cf}/pool_{tag}.pt"
    if kind == K.INJECT_CANDIDATES:
        tag = key["tag"]
        if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", tag) is None:
            raise ValueError("pool tag must match '[A-Za-z0-9_-]+'")
        cf = _inject_cf_segment(key)
        # a detector other than the pool's own (or a legacy cross-screen, see
        # InjectUseCase.screen) gets its own file, so two detectors can screen one pool
        detector = key.get("detector")
        if detector is None:
            det = ""
        elif isinstance(detector, str) and re.fullmatch(r"[A-Za-z0-9_-]+", detector):
            det = f".det-{detector}"
        else:
            raise ValueError("candidates detector must match '[A-Za-z0-9_-]+'")
        # a tagged detector (inject detector-train --detector-tag) gets its own file too
        detector_tag = key.get("detector_tag")
        if detector_tag is not None:
            if not isinstance(detector_tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", detector_tag) is None:
                raise ValueError("candidates detector_tag must match '[A-Za-z0-9_-]+'")
            det += f".dtag-{detector_tag}"
        return f"generations/{spec.dataset}/{model}/{seed}{cf}/pool_{tag}_candidates{det}.json"
    if kind == K.INJECT_QUERIES:
        return f"generations/{spec.dataset}/{model}/{seed}/inject_queries.pt"
    if kind == K.INJECT_MINED:
        tag = key["tag"]
        if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", tag) is None:
            raise ValueError("mining tag must match '[A-Za-z0-9_-]+'")
        return f"generations/{spec.dataset}/{model}/{seed}/mine_{tag}.pt"
    if kind == K.GEN_TRAJECTORY:
        # a sibling of samples.pt rather than a field inside it: every existing
        # generation predates the trajectory, and folding it in would either
        # invalidate them or make "has a generation" ambiguous
        return f"generations/{spec.dataset}/{model}/{seed}/trajectory.pt"
    if kind == K.SUBSET_MASKS:
        return f"subsets/{spec.dataset}_masks.pkl"
    if kind == K.LATENTS:
        # dataset-level (VAE is pinned per platform): no seed, no process, no
        # query segment. ``flip`` selects the pre-encoded mirrored orientation.
        flip = "_flip" if key.get("flip") else ""
        return f"latents/{spec.dataset}/{key['split']}{flip}.pt"
    if kind == K.PROMPT_EMBEDS:
        return f"latents/{spec.dataset}/prompt_embeds.pt"

    if kind == K.INJECT_META:
        return f"subsets/{spec.dataset}_inject_meta.json"
    if kind == K.INJECT_DETECTOR:
        tag = key.get("tag")
        if tag is None:                     # the legacy (filed) detector address
            return f"results/inject/{spec.dataset}/detector.pt"
        if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", tag) is None:
            raise ValueError("detector tag must match '[A-Za-z0-9_-]+'")
        return f"results/inject/{spec.dataset}/detector_{tag}.pt"
    if kind in (K.INJECT_REVIEW_SHEET, K.INJECT_REVIEW_TABLE):
        tag = key["tag"]
        if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", tag) is None:
            raise ValueError("review tag must match '[A-Za-z0-9_-]+'")
        root = f"results/inject/{spec.dataset}/{model}/{seed}/review_{tag}"
        if kind == K.INJECT_REVIEW_TABLE:
            return f"{root}/review.csv"
        host = str(key["host"])
        if re.fullmatch(r"(?:\d+|random)", host) is None:
            raise ValueError("review sheet host must be an integer or 'random'")
        return f"{root}/sheet_{host}_{int(key['sheet'])}.png"

    if kind == K.GT_MATRIX:
        return f"results/gt_matrix_{model_key(spec)}_{_ds_qt(spec)}_{seed}{rep}.npy"
    if kind == K.GT_LOSSES:
        # ``chain`` (§6.2-14 decoupling): the subset-chain seed when it differs
        # from the query seed — "chain R's subset models scored on identity S's
        # queries". Absent = the legacy diagonal (chain == seed), byte-identical.
        ch = f"_chain_{key['chain']}" if key.get("chain") is not None else ""
        es = f"_eseed_{key['eseed']}" if "eseed" in key else ""
        return f"results/gt_losses_{model_key(spec)}_{_ds_qt(spec)}_{seed}{rep}{ch}{es}.npy"
    if kind == K.GT_LOSSROW:
        # one row of the (M, Q) gt_losses matrix, grouped in a per-eseed dir so
        # sharded `subsets losses --rank/--world` runs write disjoint per-subset
        # files (resumable per subset, mergeable into GT_LOSSES).
        ch = f"_chain_{key['chain']}" if key.get("chain") is not None else ""
        return (f"results/gt_lossrows_{model_key(spec)}_{_ds_qt(spec)}_{seed}{rep}{ch}"
                f"_eseed_{key['eseed']}/subset_{key['m']}.npy")
    if kind == K.LDS_RESULT:
        return f"results/lds_results_{_ds_qt(spec)}.json"
    if kind == K.INJECT_RESULT:
        base = (f"{spec.method}/{_ds_qt(spec)}{_attr_process_suffix(spec)}"
                f"/{seed}")
        return f"scores/{base}/inject_result.json"

    if kind in (K.PARAM_CONTRIBUTIONS, K.PARAM_WEIGHTS, K.ABU_PREP):
        if not key.get("config"):
            raise ValueError(f"{kind.value} requires a method config tag")
        root = (f"baselines/{spec.method}/{_ds_qt(spec)}{_attr_process_suffix(spec)}"
                f"/{seed}{_config_segment(key)}")
        filename = {
            K.PARAM_CONTRIBUTIONS: "learning_contributions.pt",
            K.PARAM_WEIGHTS: "weights.pt",
            K.ABU_PREP: "prepare.pt",
        }[kind]
        if kind == K.PARAM_CONTRIBUTIONS and ("q0" in key or "q1" in key):
            q0, q1 = int(key["q0"]), int(key["q1"])
            if q0 < 0 or q1 <= q0:
                raise ValueError("invalid contribution query range")
            filename = f"blocks/contributions_q{q0}_{q1}.pt"
        return f"{root}/{filename}"

    if kind in (K.INJECT_CF_MASK, K.INJECT_CF_META, K.INJECT_CF_CHECKPOINT):
        drop_tag = key["drop_tag"]
        if not isinstance(drop_tag, str) or re.fullmatch(r"[A-Za-z0-9_-]+", drop_tag) is None:
            raise ValueError("INJECT counterfactual drop_tag must match '[A-Za-z0-9_-]+'")
        if kind == K.INJECT_CF_MASK:
            return f"counterfactual/{spec.dataset}/inject_{drop_tag}/{seed}/mask.npy"
        if kind == K.INJECT_CF_META:
            return f"counterfactual/{spec.dataset}/inject_{drop_tag}/{seed}/meta.json"
        return (f"checkpoints/inject_cf/{model_key(spec)}/{spec.dataset}/{drop_tag}/"
                f"{seed}/final.pt")

    if kind in (K.CF_MASK, K.CF_META, K.CF_CHECKPOINT, K.CF_GENERATION):
        # AN-2 counterfactual chain: ``arm`` = source method name or "random";
        # ``qi`` = query index (method arms) or replicate index (random arm).
        arm = key["arm"]
        if kind == K.CF_META:
            return f"counterfactual/{spec.dataset}/{arm}/{seed}/meta.json"
        if kind == K.CF_MASK:
            return f"counterfactual/{spec.dataset}/{arm}/{seed}/q{key['qi']}.npy"
        if kind == K.CF_CHECKPOINT:
            return (f"checkpoints/counterfactual/{model_key(spec)}/{spec.dataset}/{arm}/"
                    f"{seed}/q{key['qi']}/final.pt")
        return f"counterfactual/{spec.dataset}/{arm}/{seed}/q{key['qi']}/regen.pt"
    if kind in (K.CF_ANALYSIS, K.CF_FIGURE):
        ext = "json" if kind == K.CF_ANALYSIS else "png"
        return f"counterfactual/{spec.dataset}/analysis/{seed}/{key['name']}.{ext}"

    if kind in (K.TRAIN_FEATURES, K.ERROR_WEIGHT, K.QUERY_FEATURES, K.FEATURE_META,
                K.REPEAT_FEATURES):
        feat = key["feat"]                       # e.g. "das", "dtrak", "trak", "l1norm"
        # e_n is a no-grad residual norm — no gradient, no projection — so it is
        # bit-identical across every rung of an HP-p sweep. Enforced here rather
        # than left to each caller: segmenting it would silently recompute the
        # same N x T_error forward passes once per rung.
        pseg = "" if kind == K.ERROR_WEIGHT else _proj_segment(key)
        # TracInCP/GAS featurize the SAME recipe at several training checkpoints
        # and average. Without a step segment every checkpoint writes to the final
        # model's path, so the average silently degenerates to four copies of one
        # checkpoint — a number that looks plausible and is not TracIn at all.
        sseg = f"/step_{key['step']}" if key.get("step") is not None else ""
        base = (f"featurize/{feat}/{spec.dataset}{_attr_process_suffix(spec)}"
                f"{pseg}/{seed}{sseg}")
        if kind == K.TRAIN_FEATURES:
            return f"{base}/train_features.pt"
        if kind == K.REPEAT_FEATURES:
            return f"{base}/repeat_features.pt"
        if kind == K.ERROR_WEIGHT:
            return f"{base}/error_train.npy"
        if kind == K.QUERY_FEATURES:
            return f"{base}/query_features_{spec.query_type}.pt"
        return f"{base}/meta.json"

    if kind == K.EKFAC_FACTORS:
        # Query-independent (no _val segment): one curvature package serves both
        # tracks and every damping. Process segment mirrors featurize/scores.
        return f"curvature/ekfac/{spec.dataset}{_attr_process_suffix(spec)}/{seed}/factors.pt"

    if kind in (K.SCORES, K.SCORES_LAMBDA, K.REPEAT_SCORES, K.SCORES_META,
                K.SCORES_BLOCK, K.REPEAT_SCORES_BLOCK):
        base = (f"{spec.method}/{_ds_qt(spec)}{_attr_process_suffix(spec)}"
                f"{_proj_segment(key)}/{seed}{_config_segment(key)}")
        if kind in BLOCK_STEM:
            # Block partial of a curvature stream: a sibling ``blocks/``
            # sub-directory, so the cell's whole-matrix artifacts keep their
            # exact legacy names and a shard's partials travel as ordinary
            # artifacts (`balds sync push --kind scores_block`). Two-level key:
            # the query range, then optionally the train-row range.
            return (f"scores/{base}/blocks/"
                    f"{BLOCK_STEM[kind]}_q{int(key['q0'])}_{int(key['q1'])}"
                    f"{_row_segment(key)}.npz")
        if kind == K.SCORES:
            return f"scores/{base}/scores.npy"
        if kind == K.SCORES_LAMBDA:
            return f"scores/{base}/scores_lambda_{float(key['lam']):.2e}.npy"
        if kind == K.SCORES_META:
            return f"scores/{base}/meta.json"
        if kind == K.REPEAT_SCORES:
            # (D, R, M, Q) repeat-scored training subset per damping — the
            # curvature path's sigma-hat measurement (task brief B1)
            return f"scores/{base}/repeat_scores.npz"

    raise ValueError(f"no addressing rule for {kind}")


def cas_relpath(blob_hash: str) -> str:
    """Content-addressed blob path: ``cas/<hash[:2]>/<hash>``."""
    return f"cas/{blob_hash[:2]}/{blob_hash}"
