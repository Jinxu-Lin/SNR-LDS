"""Appendix C.3 / §3.1 Diffusion-side table: the DAS archive (DDPM, CIFAR-2), 64 subsets.

Protocol (DAS, ICLR 2025, Appendix E.2 — verified against the PDF 2026-09-14): M = 64
subsets of 50 % of the training set, three retrained models per subset, 1,000
validation + 1,000 generated queries, Simple-Loss output averaged over 1,000
evenly spaced timesteps with three noise instances each; projected features at
T = 100, p = 4096. The imported archive (``cifar2_das``) holds 128 subsets; this
script uses subsets 0–63 (the paper's M). The 128-subset value goes to the JSON
only as a reference and is BATCH-CONFOUNDED: the archive's two halves were
produced separately (5 vs 3 retrain replicas) and the second half's losses sit
~14 within-batch sd above the first on every query, so pooling all 128 rows
halves the LDS (ledger 6.2-51). reference_M128_batch_corrected removes the
per-query batch means before pooling; it is the only 128-row number comparable
with the 64-row table.

Rows
  D-TRAK operator = L_square readout (``dtrak_T100``), DAS operator = L1-norm
  readout (``l1norm_T100``); the mean readout (``das_T100``) goes to the JSON only.
  Part B (when ``cfa ekfac score`` has filed them on the archived DDPM):
  ``ekfac_if`` and ``fmas_raw``, read from their per-damping ``scores_lambda_*``
  files and selected on the same 64-subset GT.
Columns
  linear term / folded square τ² / signed square sign(τ)τ², per track, with the
  per-query bootstrap 95 % CI (one archived model, no seeds).
λ
  the production grid ``score.lambda_sweep`` recomputed through the registered
  method's own kernel (``METHODS[m].score``), exactly what ``cfa score`` calls.
  Primary caliber (= §3.1 table 1): λ selected on the LINEAR term by grid max on
  the 64-subset GT, the two squares applied to those same scores. Secondary
  caliber (= the 2026-09-02 D2 report): λ re-selected per transform.
Query sets
  (i) the DAS protocol, Q = 1000 per track; (ii) the Q = 100 common subset the
  curvature rows are scored on: gen rows 0–99, val = ``balanced_query_indices``
  of the test labels (50 per class) — the same selections ``cfa ekfac score``
  makes for this dataset at ``lds.Q_by_dataset.cifar2_das=100``.

Run:  PYTHONPATH=Codes/src OPENBLAS_NUM_THREADS=8 python Codes/Figure/tab_C3_ddpm.py
"""
import argparse
import glob
import re
import time
import numpy as np
import torch
from _common import DATA, JSONS, TABLES, bootstrap_ci, fmt, lds_per_query, md_table, write_json, write_table

DS, PROC, SEED = "cifar2_das", "ddpm", 42
M_PAPER, Q_SUB = 64, 100
TRACKS = ("gen", "val")
ROWS = (("dtrak_T100", "D-TRAK 算子（L_square 读出）"), ("l1norm_T100", "DAS 算子（L1 读出）"),
        ("das_T100", "均值读出（JSON only）"))
CURV = (("ekfac_if", "EK-FAC IF"), ("fmas_raw", "FMAS"))
TRANSFORMS = (("identity", lambda a: a), ("fold", lambda a: a ** 2), ("signed", lambda a: np.sign(a) * a ** 2))


def ds_track(track):
    return DS if track == "gen" else f"{DS}_val"


def load_gt(track):
    """(gt64 (64,Q), masks64, gt128 (128,Q)) from the imported per-ζ losses."""
    import pickle
    mats = [np.load(DATA / "results" / f"gt_losses_{PROC}_{ds_track(track)}_seed_{SEED}_eseed_{e}.npy") for e in (0, 1, 2)]
    gt128 = np.mean(mats, axis=0)
    filed = np.load(DATA / "results" / f"gt_matrix_{PROC}_{ds_track(track)}_seed_{SEED}.npy")
    assert np.allclose(gt128, filed), "imported gt_matrix is not the mean of the three ζ losses"
    masks = pickle.load(open(DATA / "subsets" / f"{DS}_masks.pkl", "rb"))
    return gt128[:M_PAPER], masks[:M_PAPER], gt128, masks


def query_subset(track, test_labels):
    from balds.data.base import balanced_query_indices
    return list(range(Q_SUB)) if track == "gen" else balanced_query_indices(test_labels, Q_SUB)


def lambda_scores(container, method, track):
    """{λ: (N,Q) float32} through the registered method's kernel (the `cfa score` path)."""
    from balds.schema.artifact import ArtifactKind as K
    from balds.schema.estimator import FeatureSet
    from balds.schema.registry import METHODS
    from balds.schema.runspec import RunSpec
    meth = METHODS.get(method)
    spec = RunSpec(dataset=DS, seed=SEED, method=method, query_type=track, process=PROC, conditional=True)
    feats = FeatureSet(grads=container.store.load(K.TRAIN_FEATURES, spec, feat=meth.feat_method), error=None,
                       feat_method=meth.feat_method)
    q = container.store.load(K.QUERY_FEATURES, spec, feat=meth.feat_method)
    q = q if isinstance(q, torch.Tensor) else torch.as_tensor(q)
    for lam in [float(x) for x in container.cfg["score"]["lambda_sweep"]]:
        yield lam, np.asarray(meth.score(feats, q, lam, device="cpu"), dtype=np.float32)


def curvature_scores(method, track):
    """{damping: (N,Q') float} from the filed per-damping files of `cfa ekfac score` (Part B)."""
    d = DATA / "scores" / method / ds_track(track) / PROC / f"seed_{SEED}"
    out = {}
    for p in sorted(glob.glob(str(d / "scores_lambda_*.npy"))):
        m = re.search(r"scores_lambda_(.+)\.npy$", p)
        out[float(m.group(1))] = np.load(p).astype(np.float32)
    return out


def evaluate(sc_by_key, gt, masks, cols=None):
    """{key: {transform: per-query LDS array}} on GT/masks, optionally on a query subset."""
    res = {}
    for key, sc in sc_by_key.items():
        a = sc if cols is None else sc[:, cols]
        g = gt if cols is None else gt[:, cols]
        res[key] = {name: lds_per_query(fn(a.astype(np.float64)), g, masks)[0] for name, fn in TRANSFORMS}
    return res


def select(res):
    """primary: key chosen on identity, transforms at that key; secondary: per-transform argmax."""
    keys = list(res)
    k_id = max(keys, key=lambda k: res[k]["identity"].mean())
    primary = {name: (k_id, res[k_id][name]) for name, _ in TRANSFORMS}
    secondary = {name: max(((k, res[k][name]) for k in keys), key=lambda kv: kv[1].mean()) for name, _ in TRANSFORMS}
    return primary, secondary


def cell(pq):
    m, lo, hi = bootstrap_ci(pq)
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def table_rows(sel, label, track, base_pq=None):
    k_id, pq_id = sel["identity"]; k_f, pq_f = sel["fold"]; k_s, pq_s = sel["signed"]
    dm, dlo, dhi = bootstrap_ci(pq_f - pq_id)
    return [label, track, f"{cell(pq_id)}（{k_id:g}）", f"{cell(pq_f)}（{k_f:g}）", f"{cell(pq_s)}（{k_s:g}）",
            f"{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]", f"{100 * (pq_f.mean() / pq_id.mean() - 1):+.1f}%"]


def main():
    from balds.workflows import build_container
    from balds.workflows.common import _load_ds
    c = build_container(device="cpu")
    _, test_ds = _load_ds(c.cfg, DS)
    hdr = ["方法", "轨", "一次项（λ/ϱ）", "折叠平方", "保号平方", "Δ折叠−一次项 [CI]", "相对增益"]
    out = {"protocol": "DAS App. E.2: M=64 subsets, 3 retrains, 3 ζ, 1000 timesteps; features T=100 p=4096",
           "rows": {}, "reference_M128": {}}
    tabs = {"q1000_primary": [], "q1000_reselect": [], "q100_primary": [], "q100_reselect": []}
    t0 = time.time()
    for track in TRACKS:
        gt64, masks64, gt128, masks128 = load_gt(track)
        cols = query_subset(track, test_ds.labels)
        for method, label in ROWS:
            sc = {}
            for lam, s in lambda_scores(c, method, track):
                sc[lam] = s
            r1000 = evaluate(sc, gt64, masks64)
            r100 = evaluate(sc, gt64, masks64, cols)
            p1000, s1000 = select(r1000); p100, s100 = select(r100)
            ref128 = select(evaluate(sc, gt128, masks128))[0]
            gt128c = gt128.copy()
            gt128c[:M_PAPER] -= gt128[:M_PAPER].mean(axis=0, keepdims=True)
            gt128c[M_PAPER:] -= gt128[M_PAPER:].mean(axis=0, keepdims=True)
            ref128c = select(evaluate(sc, gt128c, masks128))[0]
            out["rows"][f"{method}|{track}"] = {
                "q1000_primary": {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in p1000.items()},
                "q1000_reselect": {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in s1000.items()},
                "q100_primary": {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in p100.items()},
                "q100_reselect": {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in s100.items()},
                "per_lambda_identity_q1000": {f"{k:g}": float(v["identity"].mean()) for k, v in r1000.items()}}
            out["reference_M128"][f"{method}|{track}"] = {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in ref128.items()}
            out.setdefault("reference_M128_batch_corrected", {})[f"{method}|{track}"] = {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in ref128c.items()}
            if method != "das_T100":
                tabs["q1000_primary"].append(table_rows(p1000, label, track))
                tabs["q1000_reselect"].append(table_rows(s1000, label, track))
                tabs["q100_primary"].append(table_rows(p100, label, track))
                tabs["q100_reselect"].append(table_rows(s100, label, track))
            print(f"{method:11s} {track}: identity {p1000['identity'][1].mean():.4f} (λ={p1000['identity'][0]:g}) "
                  f"fold {p1000['fold'][1].mean():.4f} signed {p1000['signed'][1].mean():.4f} | Q100 identity "
                  f"{p100['identity'][1].mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
        # Part B: curvature rows on the archived DDPM (present only after `cfa ekfac score`)
        for method, label in CURV:
            sc = curvature_scores(method, track)
            if not sc:
                out["rows"][f"{method}|{track}"] = "pending: no per-damping scores filed for cifar2_das/ddpm"
                continue
            Qp = next(iter(sc.values())).shape[1]
            sub = cols[:Qp] if Qp <= len(cols) else None
            gsub = gt64[:, sub] if sub is not None else gt64
            r = {k: {n: lds_per_query(fn(v.astype(np.float64)), gsub, masks64)[0] for n, fn in TRANSFORMS} for k, v in sc.items()}
            p, s = select(r)
            out["rows"][f"{method}|{track}"] = {"Q": int(Qp),
                                                "primary": {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in p.items()},
                                                "reselect": {n: {"key": k, "mean": float(v.mean())} for n, (k, v) in s.items()}}
            tabs["q100_primary"].append(table_rows(p, label + f"（Q={Qp}）", track))
            tabs["q100_reselect"].append(table_rows(s, label + f"（Q={Qp}）", track))
    align = ["---", "---", "---:", "---:", "---:", "---:", "---:"]
    write_table("tab_C3_ddpm_q1000", md_table(hdr, tabs["q1000_primary"], align),
                "附录 C.3 Diffusion 侧（DAS 存档 DDPM，64 子集 × 3 重训 × 3 ζ × 1000 时步；T=100，p=4096；Q=1000）：λ 按一次项选定，平方沿用同一分数")
    write_table("tab_C3_ddpm_q1000_reselect", md_table(hdr, tabs["q1000_reselect"], align),
                "同上，λ 逐变换重选（D2 口径）")
    write_table("tab_C3_ddpm_q100", md_table(hdr, tabs["q100_primary"], align),
                f"附录 C.3 共同查询子集 Q={Q_SUB}（gen 前 100 / val 类均衡 100）：投影行 + 曲率行（曲率行由 `cfa ekfac score` 落盘后填入）")
    write_table("tab_C3_ddpm_q100_reselect", md_table(hdr, tabs["q100_reselect"], align),
                f"同上，λ/ϱ 逐变换重选")
    write_json("tab_C3_ddpm", out)


if __name__ == "__main__":
    main()
