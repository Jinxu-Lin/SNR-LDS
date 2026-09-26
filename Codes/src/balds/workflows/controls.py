"""Task-scoped CPU analyses. Historical inputs are read-only; no model execution."""
from __future__ import annotations
import csv
import gzip
import json
import pickle
from pathlib import Path
import numpy as np

from balds.evaluation.lds import compute_lds
from balds.evaluation.background import SNR_RULE, fit_snr_background

GRID = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)

def read_table(path):
    # The filed DAS300 exporter used escaped tab separators. Preserve its source.
    text = path.read_text()
    if "\t" not in text.splitlines()[0] and "\\t" in text.splitlines()[0]:
        text = text.replace("\\t", "\t")
    delim = "\t" if "\t" in text.splitlines()[0] else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delim))

def ci(a):
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError("invalid query-level bootstrap input")
    bank = np.random.default_rng(20260920).integers(0, len(a), (2000, len(a)))
    lo, hi = np.quantile(a[bank].mean(axis=1), [.025, .975])
    return dict(mean=float(a.mean()), ci_low=float(lo), ci_high=float(hi))

def head(s, k, absolute=False):
    order = np.argsort(-np.abs(s) if absolute else -s, axis=0, kind="stable")
    mask = np.zeros(s.shape, dtype=bool)
    np.put_along_axis(mask, order[:k], True, axis=0)
    return mask

def pair_loss(p, y):
    i, j = np.triu_indices(len(y), 1)
    dy, dp = y[i] - y[j], p[i] - p[j]
    valid = dy != 0
    loss = ((dp * dy < 0) + .5 * (dp == 0))
    denom = valid.sum(axis=0)
    value = np.divide((loss * valid).sum(axis=0), denom,
                      out=np.full(y.shape[1], np.nan), where=denom > 0)
    return value, (dy == 0).mean(axis=0), (dp == 0).mean(axis=0)

def nullable(x):
    return float(x) if np.isfinite(x) else None

def metrics(s, d, y):
    p = d @ s
    lds = compute_lds(y, p)[0]
    loss, yt, pt = pair_loss(p, y)
    return p, lds, loss, yt, pt


def fixed_snr_controls(raw, transformed, d, response, *, head_percentages=GRID,
                       zeta=3.0):
    """Evaluate transforms/heads on one SNR mask fitted only from raw scores.

    ``transformed`` maps filed transform names to readout matrices.  Head masks
    are intersected with the frozen raw-score SNR mask; no transformed value is
    ever refitted. Heads use absolute pre-square scores, as in Figure 3.
    """
    raw = np.asarray(raw, dtype=np.float64)
    d, response = np.asarray(d), np.asarray(response, dtype=np.float64)
    if raw.ndim != 2 or d.shape[1] != raw.shape[0] or response.shape != (len(d), raw.shape[1]):
        raise ValueError("raw scores, deletion matrix and responses are not aligned")
    selected = np.zeros(raw.shape, dtype=bool)
    fits = []
    for query in range(raw.shape[1]):
        fit = fit_snr_background(raw[:, query], zetas=(zeta,))
        fits.append(fit)
        if fit['status'] != 'fit_failed':
            selected[:, query] = fit['selected'][float(zeta)]
    rows = []
    for name, values in transformed.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != raw.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid transformed readout: {name}")
        variants = {'full': np.ones(raw.shape, dtype=bool), 'snr': selected}
        for pct in head_percentages:
            count = int(np.ceil(len(raw) * float(pct) / 100.0))
            variants[f'head_{pct}'] = head(raw, count, absolute=True)
            variants[f'snr_head_{pct}'] = selected & head(raw, count, absolute=True)
        for rule, support in variants.items():
            prediction, lds, *_ = metrics(np.where(support, values, 0.0), d, response)
            for query in range(raw.shape[1]):
                rows.append(dict(transform=name, rule=rule, query_id=query,
                                 lds=float(lds[query]), n_selected=int(support[:, query].sum()),
                                 prediction=prediction[:, query].tolist(),
                                 fit_status=fits[query]['status']))
    return dict(rule=SNR_RULE, zeta=float(zeta), selected=selected, fits=fits, rows=rows)

def data_pair(data, ds, track, seed):
    key = ds + ("_val" if track == "val" else "")
    y = np.load(data / f"results/gt_matrix_fm_{key}_seed_{seed}.npy").astype(float)
    with (data / f"subsets/{ds}_masks.pkl").open("rb") as f:
        keep = np.asarray(pickle.load(f), dtype=float)
    d = 1 - keep[:len(y)]
    return key, d, y



def das_linear(data, ds, track, seed):
    key=ds+("_val" if track=="val" else "")
    if ds != "artbench2_256":
        return np.load(data/f"scores/das_T100/{key}/seed_{seed}/scores.npy").astype(float)
    accepted = data / f"results/snr_lds_20260925/a3/inputs/artbench2_256/{track}/das_presquare_lambda1.npy"
    if accepted.is_file():
        return np.load(accepted).astype(float)
    import torch
    import balds.attribution
    from balds.schema.estimator import FeatureSet
    from balds.schema.registry import METHODS
    base=data/"featurize/das_T100/artbench2_256/seed_42"
    # Accepted AB2 same-configuration DAS diagnostic uses native lambda=1.0.
    method=METHODS.get("das_T100")
    return np.asarray(method.score(FeatureSet(
        grads=torch.load(base/"train_features.pt",map_location="cpu",weights_only=True),
        error=np.load(data/"featurize/das/artbench2_256/seed_42/error_train.npy"),
        feat_method=method.feat_method),
        torch.load(base/f"query_features_{track}.pt",map_location="cpu",weights_only=True),
        1.0,device="cpu"),dtype=float)

def transforms(data, out):
    rows, missing, groups = [], [], {}
    for ds in ["cifar2_5k","cifar10_v2","artbench2_256"]:
      for seed in ([42] if ds=="artbench2_256" else [42,123,456]):
       for track in ["gen","val"]:
        key,d,y=data_pair(data,ds,track,seed)
        for method in ["das_native_sq"]:
          try:
            raw=das_linear(data,ds,track,seed) if method=="das_native_sq" else np.load(data/f"scores/{method}/{key}/seed_{seed}/scores.npy").astype(float)
          except FileNotFoundError as exc:
            missing.append(dict(dataset=ds,seed=seed,track=track,method=method,reason=str(exc)));continue
          if raw.shape!=(d.shape[1],y.shape[1]) or not np.isfinite(raw).all():
            raise ValueError(f"invalid score {ds}/{method}/{track}/{seed}")
          versions={"linear":raw,"square":raw**2,"signed_square":np.sign(raw)*raw**2} if method=="das_native_sq" else {"native":raw}
          for transform,s in versions.items():
            variants={"full":s}
            for pct in [5]:
                k=int(np.ceil(len(s)*pct/100));h=head(s,k)
                variants[f"head_{pct}"]=s*h
                if transform!="native":
                    variants[f"fixed_linear_head_{pct}"]=s*head(raw,k)
            for rule,z in variants.items():
                _,lds,loss,yt,pt=metrics(z,d,y)
                groups.setdefault((ds,track,method,transform,rule),[]).append((seed,lds))
                for q in range(y.shape[1]):
                    rows.append(dict(dataset=ds,seed=seed,track=track,method=method,
                        transform=transform,rule=rule,query=q,lds=float(lds[q]),
                        pair_loss=nullable(loss[q]),response_ties=float(yt[q]),prediction_ties=float(pt[q]),
                        score_mean=float(z[:,q].mean()),nonzero=int(np.count_nonzero(z[:,q]))))
          print(ds,seed,track,method,"complete",flush=True)
          # Save per panel to keep restart losses bounded without touching old outputs.
          panel=[r for r in rows if r["dataset"]==ds and r["seed"]==seed and r["track"]==track and r["method"]==method]
          table(out/f"panels/{ds}_{seed}_{track}_{method}.tsv.gz",panel)
          rows=[]
    summaries=[]
    for (ds,track,method,transform,rule),items in groups.items():
        items=sorted(items)
        values=np.stack([v for _,v in items])
        rng=np.random.default_rng(20260920)
        if track=="val":
            ix=rng.integers(0,values.shape[1],(2000,values.shape[1]))
            boot=values[:,ix].mean(axis=(0,2))
        else:
            boot=np.zeros(2000)
            for v in values:
                ix=rng.integers(0,len(v),(2000,len(v)))
                boot+=v[ix].mean(axis=1)/len(values)
        lo,hi=np.quantile(boot,[.025,.975])
        summaries.append(dict(dataset=ds,track=track,method=method,transform=transform,rule=rule,
            mean=float(values.mean()),seed_std=float(values.mean(axis=1).std(ddof=1)) if len(values)>1 else None,
            ci_low=float(lo),ci_high=float(hi),seeds=[s for s,_ in items]))
    # Same draws and same query IDs for each pair; never bootstrap scalar seed means.
    differences=[]
    for key,items in groups.items():
        ds,track,method,transform,rule=key
        reference=(ds,track,method,"linear",rule) if method=="das_native_sq" and transform!="linear" else (ds,track,method,transform,"full")
        if key==reference or reference not in groups:
            continue
        a,b=sorted(items),sorted(groups[reference])
        if [s for s,_ in a] != [s for s,_ in b]:
            raise ValueError("paired model coverage differs")
        delta=np.stack([x[1]-y[1] for x,y in zip(a,b)])
        rng=np.random.default_rng(20260920)
        if track=="val":
            ix=rng.integers(0,delta.shape[1],(2000,delta.shape[1]))
            boot=delta[:,ix].mean(axis=(0,2))
        else:
            boot=np.zeros(2000)
            for v in delta:
                ix=rng.integers(0,len(v),(2000,len(v)))
                boot+=v[ix].mean(axis=1)/len(delta)
        lo,hi=np.quantile(boot,[.025,.975])
        differences.append(dict(dataset=ds,track=track,method=method,transform=transform,rule=rule,
            reference_transform=reference[3],reference_rule=reference[4],mean=float(delta.mean()),
            ci_low=float(lo),ci_high=float(hi)))
    dump(out/"summaries.json",summaries)
    dump(out/"paired_differences.json",differences)
    dump(out/"missing.json",missing)
    dump(out/"completion.json",dict(status="produced",scope="Final Tables 8 and 9: matched DAS readouts",
        missing=len(missing),note="No new model/score selection; AB2 DAS restored from features at native lambda=1.0. Paired query intervals retain the filed bootstrap protocol."))


