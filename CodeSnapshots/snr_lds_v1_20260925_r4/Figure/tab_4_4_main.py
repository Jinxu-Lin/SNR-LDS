"""§4.4 main benchmark tables: CIFAR-2 / CIFAR-10 (full LDS) and ArtBench-2.

Every cell is the full LDS of the method's FILED native scores (``scores.npy``,
λ/ϱ already selected by the production pipeline) against the platform GT.
 * three seeds → mean±std; identical values across seeds (model-independent
   pixel/CLIP on the shared val queries, the constant control on val) → single
   value tagged n=1; one seed → value with per-query bootstrap 95 % CI.
 * constant control: τ ≡ 1 (the prediction is the number of deleted rows).
 * exact-retraining reference (CIFAR-2): one retrain chain's subset losses as
   the predictor, the mean of the other two chains as the truth; gen std over
   the three query identities, val a single value (shared queries).
 * ArtBench-2 DAS native output: the DAS linear-term scores squared with λ
   RE-selected on the squared scores over the production λ grid (the AB2
   battery rule), computed through the registered method's own kernel; the
   same-λ square is reported next to it for transparency.
"""
import itertools
import numpy as np
from scipy import stats
from _common import (DATA, SEEDS, TRACKS, Cell, fmt, fmt_ci, fmt_ms, has_scores, load_gt, load_masks, lds,
                     lds_per_query, md_table, write_json, write_table)

METHODS = (("pixel_dot", "Raw pixel dot"), ("pixel_cos", "Raw pixel cos"), ("clip_dot", "CLIP dot"),
           ("clip_cos", "CLIP cos"), ("grad_dot_T100", "Gradient dot"), ("grad_cos_T100", "Gradient cos"),
           ("tracincp_T100", "TracInCP"), ("gas_T100", "GAS"),
           ("journey_trak_T100", "Journey-TRAK（均值读出类比版；仅 gen）"),
           ("relative_if_T100", "Relative IF"), ("renorm_if_T100", "Renorm. IF"),
           ("trak_T100", "TRAK（损失读出）"), ("dtrak_T100", "D-TRAK（平方范数读出）"),
           ("ekfac_if", "EK-FAC IF"), ("fmas_raw", "FMAS"))


def cell_text(method, ds, track, seeds):
    vals, pq = [], []
    for s in seeds:
        if has_scores(method, ds, track, s):
            c = Cell(method, ds, track, s)
            p, m = lds_per_query(c.a, c.gt, c.masks)
            vals.append(m); pq.append(p)
    if not vals:
        return "—", vals
    if len(vals) == 1:
        return fmt_ci(pq[0]) + "（n=1）", vals
    return fmt_ms(vals), vals


def constant_control(ds, track, seeds):
    vals, pq = [], []
    for s in seeds:
        gt = load_gt(ds, track, s); masks = load_masks(ds)
        ones = np.ones((len(masks[0]), gt.shape[1]))
        p, m = lds_per_query(ones, gt, masks)
        vals.append(m); pq.append(p)
    if len(vals) == 1 or np.all(np.asarray(vals) == vals[0]):
        return fmt_ci(pq[0]) + "（n=1）", vals
    return fmt_ms(vals), vals


def exact_retraining_c2():
    """CIFAR-2: chain r's subset losses (mean over 3 ζ) vs the mean of the other two chains."""
    ds, chains, eseeds = "cifar2_5k", (42, 123, 456), (0, 1, 2)

    def losses(seed, track, chain):
        if track == "val":
            stem = f"gt_losses_fm_{ds}_val_seed_{chain}_eseed_"
        elif chain == seed:
            stem = f"gt_losses_fm_{ds}_seed_{seed}_eseed_"
        else:
            stem = f"gt_losses_fm_{ds}_seed_{seed}_chain_{chain}_eseed_"
        return np.mean([np.load(DATA / "results" / f"{stem}{e}.npy") for e in eseeds], axis=0)

    def sp(A, B):
        return float(np.mean([stats.spearmanr(A[:, q], B[:, q]).statistic for q in range(A.shape[1])]))

    out = {}
    for track in TRACKS:
        per_identity = []
        for s in SEEDS if track == "gen" else (42,):
            P = {r: losses(s, track, r) for r in chains}
            M = min(v.shape[0] for v in P.values())
            P = {r: v[:M] for r, v in P.items()}
            per_identity.append(np.mean([sp(P[r], np.mean([P[o] for o in chains if o != r], axis=0)) for r in chains]))
        out[track] = per_identity
    return out


def artbench_das_native(track):
    """DAS linear scores squared, λ re-selected per transform through the method's kernel."""
    import torch
    from balds.workflows import build_container
    from balds.schema.artifact import ArtifactKind as K
    from balds.schema.estimator import FeatureSet
    from balds.schema.registry import METHODS as REG
    from balds.schema.runspec import RunSpec
    ds, seed, m = "artbench2_256", 42, "das_T100"
    c = build_container(device="cpu"); meth = REG.get(m)
    spec = RunSpec(dataset=ds, seed=seed, method=m, query_type=track, process="cfm", conditional=True)
    feats = FeatureSet(grads=c.store.load(K.TRAIN_FEATURES, spec, feat=meth.feat_method), error=None,
                       feat_method=meth.feat_method)
    q = c.store.load(K.QUERY_FEATURES, spec, feat=meth.feat_method)
    q = q if isinstance(q, torch.Tensor) else torch.as_tensor(q)
    gt, masks = load_gt(ds, track, seed), load_masks(ds)
    best = {"identity": None, "square": None}
    for lam in [float(x) for x in c.cfg["score"]["lambda_sweep"]]:
        sc = np.asarray(meth.score(feats, q, lam, device="cpu"), dtype=np.float64)
        for name, mat in (("identity", sc), ("square", sc ** 2)):
            p, mean = lds_per_query(mat, gt, masks)
            if best[name] is None or mean > best[name][1]:
                best[name] = (lam, mean, p)
    filed = Cell(m, ds, track, seed)
    return {"identity_regrid": best["identity"], "square_regrid": best["square"],
            "square_same_lambda": lds_per_query(filed.a ** 2, filed.gt, filed.masks)}


def main():
    res = {}
    # --- CIFAR-2 / CIFAR-10 ---
    rows = []
    for name, ds, track in (("参考：subset-size control", None, None),):
        pass
    ctrl = {ds: {t: constant_control(ds, t, SEEDS) for t in TRACKS} for ds in ("cifar2_5k", "cifar10_v2")}
    ex = exact_retraining_c2()
    rows.append(["*参考：subset-size control*"] + [f"*{ctrl[ds][t][0]}*" for ds in ("cifar2_5k", "cifar10_v2") for t in TRACKS])
    rows.append(["*参考：exact retraining（单链预测 vs 另两链均值真值）*", f"*{fmt_ms(ex['gen'])}*",
                 f"*{ex['val'][0]:.4f}（n=1）*", "—", "—"])
    for m, label in METHODS:
        r = [label]
        for ds in ("cifar2_5k", "cifar10_v2"):
            for t in TRACKS:
                txt, vals = cell_text(m, ds, t, SEEDS)
                res[f"{m}|{ds}|{t}"] = vals
                r.append(txt)
        rows.append(r)
    write_table("tab_4_4_main_cifar", md_table(["方法", "CIFAR-2 gen", "CIFAR-2 val", "CIFAR-10 gen", "CIFAR-10 val"], rows),
                "§4.4 CIFAR 两平台全量 LDS（原生分数；三种子 mean±std，单种子带逐查询自举 95% CI）")
    res["constant_control"] = {ds: {t: ctrl[ds][t][1] for t in TRACKS} for ds in ctrl}
    res["exact_retraining_c2"] = ex
    # --- ArtBench-2 ---
    ds, seed = "artbench2_256", 42
    rows = [["*参考：subset-size control*"] + [f"*{constant_control(ds, t, (seed,))[0]}*" for t in TRACKS]]
    for m, label in METHODS:
        rows.append([label] + [cell_text(m, ds, t, (seed,))[0] for t in TRACKS])
    das = {t: artbench_das_native(t) for t in TRACKS}
    rows.append(["DAS 一次项（诊断参照，已归档 λ）"] + [cell_text("das_T100", ds, t, (seed,))[0] for t in TRACKS])
    rows.append(["DAS（FM 适配版，原生平方输出，逐变换重选 λ）"]
                + [f"{fmt_ci(das[t]['square_regrid'][2])}（λ={das[t]['square_regrid'][0]:g}）" for t in TRACKS])
    rows.append(["（对照）DAS 一次项分数直接平方，沿用一次项 λ"] + [fmt_ci(das[t]["square_same_lambda"][0]) for t in TRACKS])
    rows.append(["（对照）DAS 一次项按重算网格重选 λ"]
                + [f"{fmt_ci(das[t]['identity_regrid'][2])}（λ={das[t]['identity_regrid'][0]:g}）" for t in TRACKS])
    write_table("tab_4_4_main_artbench2", md_table(["方法", "ArtBench-2 gen", "ArtBench-2 val"], rows),
                "§4.4 ArtBench-2（单种子 42，M=32；逐查询自举 95% CI）")
    res["artbench2_das"] = {t: {k: (v[0], v[1]) if k != "square_same_lambda" else v[1] for k, v in das[t].items()} for t in TRACKS}
    write_json("tab_4_4_main", res)


if __name__ == "__main__":
    main()
