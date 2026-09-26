"""Pure exact-probe response decomposition for prediction pilot P3."""
from __future__ import annotations

import itertools
import numpy as np


def probe_terms(reference, target, chain_outputs, *, output_ndim: int | None = None) -> dict[str, np.ndarray]:
    """T6 scalars for one or many probes.

    ``reference`` and ``target`` have shape ``(..., D...)`` and
    ``chain_outputs`` has a leading chain axis.  Output dimensions are flattened
    after the shared prefix; callers normally pass ``(chain, probe, C,H,W)``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    sub = np.asarray(chain_outputs, dtype=np.float64)
    if ref.shape != tgt.shape or sub.ndim != ref.ndim + 1 or sub.shape[1:] != ref.shape:
        raise ValueError("chain_outputs must have shape (chain, *reference.shape)")
    if sub.shape[0] < 2:
        raise ValueError("at least two chains are required")
    if not all(np.isfinite(v).all() for v in (ref, tgt, sub)):
        raise ValueError("probe tensors must be finite")
    # The final three axes are the model output for image models.  For vectors,
    # the final axis is used.  A caller can insert a singleton probe prefix.
    if output_ndim is None:
        output_ndim = 3 if ref.ndim >= 3 else 1
    if output_ndim < 1 or output_ndim > ref.ndim:
        raise ValueError("output_ndim must select one or more trailing reference axes")
    axes_ref = tuple(range(ref.ndim-output_ndim, ref.ndim))
    axes_sub = tuple(axis+1 for axis in axes_ref)
    residual = ref - tgt
    delta = sub - ref[None, ...]
    j_ref = np.mean(residual*residual, axis=axes_ref)
    j_sub = np.mean((sub-tgt[None, ...])**2, axis=axes_sub)
    y = j_sub - j_ref[None, ...]
    linear = 2*np.mean(residual[None, ...]*delta, axis=axes_sub)
    energy = np.mean(delta*delta, axis=axes_sub)
    pairs = list(itertools.combinations(range(sub.shape[0]), 2))
    cross = np.stack([
        np.mean(delta[a]*delta[b], axis=axes_ref) for a, b in pairs
    ])
    scale = 1.0 + j_sub + j_ref[None, ...]
    error = np.abs(y-linear-energy)
    if np.any(error > 1e-10*scale):
        raise ArithmeticError("Y=L+E probe identity exceeded float64 tolerance")
    return {"J_ref": j_ref, "J_sub": j_sub, "Y": y, "L": linear,
            "E": energy, "cross": cross,
            "chain_pairs": np.asarray(pairs, dtype=np.int64),
            "identity_error": error}


def chain_decomposition(linear, energy, cross, chain_pairs=None) -> dict[str, np.ndarray]:
    """T6 three-or-more-chain finite-sample decomposition from saved scalars."""
    l = np.asarray(linear, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    x = np.asarray(cross, dtype=np.float64)
    if l.shape != e.shape or l.ndim < 1:
        raise ValueError("linear/energy must share a leading chain axis")
    chains = l.shape[0]
    pairs = (np.asarray(list(itertools.combinations(range(chains), 2)), dtype=np.int64)
             if chain_pairs is None else np.asarray(chain_pairs, dtype=np.int64))
    if x.shape[0] != len(pairs) or x.shape[1:] != l.shape[1:]:
        raise ValueError("cross values do not match chain pairs/output axes")
    expected = {tuple(v) for v in itertools.combinations(range(chains), 2)}
    if {tuple(map(int, v)) for v in pairs.tolist()} != expected:
        raise ValueError("chain_pairs must cover every unordered pair exactly once")
    e_total = e.mean(axis=0)
    u_mean = (2*x.sum(axis=0))/(chains*(chains-1))
    v_train = e_total-u_mean
    tol = 1e-12*(1+np.abs(e_total)+np.abs(u_mean))
    if np.any(v_train < -tol):
        raise ArithmeticError("V_train is negative beyond float64 roundoff")
    l_bar = l.mean(axis=0)
    y_bar = l_bar+e_total
    reconstructed = l_bar+u_mean+v_train
    if not np.allclose(y_bar, reconstructed, rtol=1e-12, atol=1e-14):
        raise ArithmeticError("chain response decomposition failed")
    return {"L_bar": l_bar, "E_total": e_total, "U_mean": u_mean,
            "V_train": v_train, "Y_bar": y_bar,
            "reconstructed": reconstructed}


def leave_one_chain(linear, energy, cross, chain_pairs=None) -> np.ndarray:
    """Stack ``(L_bar,U_mean,V_train,Y_bar)`` for each omitted chain."""
    l = np.asarray(linear, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    chains = l.shape[0]
    pairs = (np.asarray(list(itertools.combinations(range(chains), 2)), dtype=np.int64)
             if chain_pairs is None else np.asarray(chain_pairs, dtype=np.int64))
    x = np.asarray(cross, dtype=np.float64)
    if chains < 3:
        raise ValueError("leave-one-chain requires at least three chains")
    results = []
    for omitted in range(chains):
        keep = [j for j in range(chains) if j != omitted]
        remap = {old: new for new, old in enumerate(keep)}
        pair_idx = [j for j, (a,b) in enumerate(pairs) if a in remap and b in remap]
        new_pairs = np.asarray([(remap[int(pairs[j,0])], remap[int(pairs[j,1])])
                                for j in pair_idx], dtype=np.int64)
        out = chain_decomposition(l[keep], e[keep], x[pair_idx], new_pairs)
        results.append(np.stack([out["L_bar"], out["U_mean"],
                                 out["V_train"], out["Y_bar"]]))
    return np.stack(results)


def paired_component_bootstrap(components, plan) -> dict[str, np.ndarray]:
    """Resample all ``(mask,query)`` response components with one index plan."""
    values = {name: np.asarray(value, dtype=np.float64)
              for name, value in components.items()}
    if not values:
        raise ValueError("at least one response component is required")
    shape = next(iter(values.values())).shape
    if len(shape) != 2 or any(value.shape != shape for value in values.values()):
        raise ValueError("response components must share (mask,query) axes")
    modes = {
        "query_only": (plan["query_only_mask"], plan["query_only_query"]),
        "mask_only": (plan["mask_only_mask"], plan["mask_only_query"]),
        "both": (plan["both_mask"], plan["both_query"]),
    }
    out = {}
    for mode, (mask_draws, query_draws) in modes.items():
        mi, qi = np.asarray(mask_draws), np.asarray(query_draws)
        if mi.ndim != 2 or qi.ndim != 2 or mi.shape[0] != qi.shape[0]:
            raise ValueError("paired component bootstrap plan has invalid axes")
        for name, value in values.items():
            estimates = np.empty(mi.shape[0], dtype=np.float64)
            for b in range(mi.shape[0]):
                estimates[b] = value[np.ix_(mi[b], qi[b])].mean()
            out[f"{mode}__{name}"] = estimates
    return out
