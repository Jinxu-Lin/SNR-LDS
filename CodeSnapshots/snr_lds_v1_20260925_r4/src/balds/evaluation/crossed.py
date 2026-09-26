"""Pure P2 crossed-repeat estimators and the frozen M1-v1/M2 candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _crossed(z) -> np.ndarray:
    out = np.asarray(z, dtype=np.float64)
    if out.ndim < 2 or out.shape[0] < 2 or out.shape[1] < 2:
        raise ValueError("Z must have train/query replica axes of size at least two")
    if not np.isfinite(out).all():
        raise ValueError("Z contains NaN or infinity")
    return out


def crossed_anova(z) -> dict[str, np.ndarray]:
    """T4 mean squares and signed variance-component estimates."""
    value = _crossed(z)
    r, s = value.shape[:2]
    grand = value.mean(axis=(0, 1))
    row = value.mean(axis=1)
    col = value.mean(axis=0)
    row_expand = row[:, None, ...]
    col_expand = col[None, :, ...]
    grand_expand = grand[None, None, ...]
    ms_t = s * np.sum((row - grand) ** 2, axis=0) / (r - 1)
    ms_q = r * np.sum((col - grand) ** 2, axis=0) / (s - 1)
    residual = value - row_expand - col_expand + grand_expand
    ms_i = np.sum(residual * residual, axis=(0, 1)) / ((r - 1) * (s - 1))
    a = (ms_t - ms_i) / s
    b = (ms_q - ms_i) / r
    d = ms_i
    return {
        "MS_T": ms_t, "MS_Q": ms_q, "MS_I": ms_i,
        "a_hat": a, "b_hat": b, "d_hat": d,
        "single_score_variance": a + b + d,
        "crossed_mean_variance": a / r + b / s + d / (r * s),
        "grand_mean": grand,
    }


def u_cross(z) -> dict[str, np.ndarray]:
    """M2 full crossed square through both exactly equivalent formulas."""
    value = _crossed(z)
    r, s = value.shape[:2]
    total = value.sum(axis=(0, 1))
    row_sums = value.sum(axis=1)
    col_sums = value.sum(axis=0)
    numerator = (total * total - np.sum(row_sums * row_sums, axis=0)
                 - np.sum(col_sums * col_sums, axis=0)
                 + np.sum(value * value, axis=(0, 1)))
    direct = numerator / (r * (r - 1) * s * (s - 1))
    anova = crossed_anova(value)
    v_hat = (anova["MS_T"] + anova["MS_Q"] - anova["MS_I"]) / (r * s)
    equivalent = anova["grand_mean"] ** 2 - v_hat
    if not np.allclose(direct, equivalent, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("U_cross formulas disagree")
    return {"U_cross": direct, "V_hat_grand": v_hat,
            "equivalent": equivalent, "absolute_error": np.abs(direct-equivalent)}


def m0_summary(z) -> dict[str, np.ndarray]:
    value = _crossed(z)
    r, s = value.shape[:2]
    k = min(r, s)
    anova = crossed_anova(value)
    diagonal = np.mean(np.stack([value[j, j] for j in range(k)]), axis=0)
    leave_train = np.stack([np.delete(value, j, axis=0).mean(axis=(0, 1)) for j in range(r)])
    leave_query = np.stack([np.delete(value, j, axis=1).mean(axis=(0, 1)) for j in range(s)])
    return {
        **anova,
        "diagonal_mean": diagonal,
        "theoretical_variance_reduction": anova["d_hat"] * (k - 1) / (k * k),
        "leave_one_train": leave_train, "leave_one_query": leave_query,
    }


@dataclass(frozen=True)
class IsotonicFit:
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray


def pava_fit(x, y) -> IsotonicFit:
    """Count-weighted duplicate merge followed by nondecreasing PAVA."""
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    if xv.shape != yv.shape or xv.size == 0:
        raise ValueError("PAVA x/y must be non-empty vectors with equal length")
    if not np.isfinite(xv).all() or not np.isfinite(yv).all():
        raise ValueError("PAVA inputs must be finite")
    order = np.argsort(xv, kind="stable")
    sx, sy = xv[order], yv[order]
    ux, first, counts = np.unique(sx, return_index=True, return_counts=True)
    uy = np.add.reduceat(sy, first) / counts
    blocks: list[list[float]] = []
    for index, (mean, weight) in enumerate(zip(uy, counts)):
        blocks.append([float(index), float(index), float(mean), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            weight_sum = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / weight_sum
            blocks.append([left[0], right[1], mean, weight_sum])
    fitted = np.empty_like(uy)
    for start, end, mean, _ in blocks:
        fitted[int(start):int(end)+1] = mean
    return IsotonicFit(ux, fitted, counts.astype(np.float64))


def pava_predict(fit: IsotonicFit, x) -> np.ndarray:
    """Linear interpolation with continuous slope-one extrapolation."""
    value = np.asarray(x, dtype=np.float64)
    out = np.interp(value, fit.x, fit.y)
    left = value < fit.x[0]
    right = value > fit.x[-1]
    out[left] = fit.y[0] + value[left] - fit.x[0]
    out[right] = fit.y[-1] + value[right] - fit.x[-1]
    return out


def m1_crossfit(a, b, row_folds: Sequence[int]) -> dict[str, object]:
    """Frozen M1-v1, independently fit query-by-query using A/B only."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    folds = np.asarray(row_folds)
    if av.ndim == 1:
        av, bv = av[:, None], bv[:, None]
    if av.shape != bv.shape or av.ndim != 2 or folds.shape != (av.shape[0],):
        raise ValueError("A/B must share (row, query), with one fold per row")
    if set(np.unique(folds).tolist()) != {0, 1}:
        raise ValueError("row_folds must contain both 0 and 1")
    out = np.empty_like(av)
    recipes: list[dict] = []
    for q in range(av.shape[1]):
        q_recipe = {"query": q, "folds": []}
        for held in (0, 1):
            train, test = folds != held, folds == held
            ab = pava_fit(av[train, q], bv[train, q])
            ba = pava_fit(bv[train, q], av[train, q])
            out[test, q] = 0.5 * (
                pava_predict(ab, av[test, q]) + pava_predict(ba, bv[test, q])
            )
            q_recipe["folds"].append({
                "held": held,
                "a_to_b": {"x": ab.x.tolist(), "y": ab.y.tolist(),
                           "weight": ab.weight.tolist()},
                "b_to_a": {"x": ba.x.tolist(), "y": ba.y.tolist(),
                           "weight": ba.weight.tolist()},
            })
        recipes.append(q_recipe)
    return {"prediction": out.squeeze() if np.asarray(a).ndim == 1 else out,
            "recipes": recipes}


def independent_risk(candidate, baseline, target, *, input_scale=None,
                     scale_power: int = 1) -> dict[str, np.ndarray]:
    """Input-only-normalisable M1/M2 risk and direction differences."""
    f = np.asarray(candidate, dtype=np.float64)
    g = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(target, dtype=np.float64)
    if f.ndim == 1:
        f, g, c = f[:, None], g[:, None], c[:, None]
    if f.shape != g.shape or f.shape != c.shape or f.ndim != 2:
        raise ValueError("candidate/baseline/target must share (row, query)")
    delta_mse = np.mean((f-c)**2 - (g-c)**2, axis=0)
    nf = np.linalg.norm(f, axis=0)
    ng = np.linalg.norm(g, axis=0)
    ok = (nf > 0) & (ng > 0)
    delta_j = np.full(f.shape[1], np.nan)
    delta_j[ok] = np.sum((f[:, ok]/nf[ok] - g[:, ok]/ng[ok]) * c[:, ok], axis=0)
    scale = (np.ones(f.shape[1], dtype=np.float64) if input_scale is None
             else np.asarray(input_scale, dtype=np.float64).reshape(-1))
    if scale.shape != (f.shape[1],) or scale_power not in (1, 2):
        raise ValueError("input scale must have one value per query; power must be 1 or 2")
    scale_degenerate = ~np.isfinite(scale) | (scale <= 0)
    mse_denom = scale**(2*scale_power)
    direction_denom = np.sqrt(f.shape[0])*scale**scale_power
    norm_mse = np.divide(delta_mse, mse_denom, out=np.full_like(delta_mse, np.nan),
                         where=~scale_degenerate)
    norm_j = np.divide(delta_j, direction_denom, out=np.full_like(delta_j, np.nan),
                       where=(~scale_degenerate) & np.isfinite(delta_j))
    return {"delta_mse": delta_mse, "delta_J": delta_j,
            "normalized_delta_mse": norm_mse, "normalized_delta_J": norm_j,
            "input_scale": scale, "direction_degenerate": ~ok,
            "scale_degenerate": scale_degenerate}


def p2_candidates(z, row_folds: Sequence[int]) -> dict[str, object]:
    """Build the frozen two-block M0/M1/M2 values and C/D validations."""
    value = _crossed(z)
    if value.shape[0] < 4 or value.shape[1] < 4:
        raise ValueError("P2 candidates require a 4x4 crossed design")
    a, b, c, d = value[0, 0], value[1, 1], value[2, 2], value[3, 3]
    g = value[:2, :2].mean(axis=(0, 1))
    m1 = m1_crossfit(a, b, row_folds)["prediction"]
    u2 = 0.5 * (value[0, 0] * value[1, 1] + value[0, 1] * value[1, 0])
    g2 = g * g
    w = c * d
    input_scale = np.sqrt(np.mean((a*a+b*b)/2, axis=0))
    diag = .5*(a+b)
    m0c = independent_risk(g, diag, c, input_scale=input_scale)
    m0d = independent_risk(g, diag, d, input_scale=input_scale)
    rc = independent_risk(m1, g, c, input_scale=input_scale)
    rd = independent_risk(m1, g, d, input_scale=input_scale)
    ravg = {name: .5*(rc[name]+rd[name]) for name in (
        "delta_mse", "delta_J", "normalized_delta_mse", "normalized_delta_J")}
    ravg.update(input_scale=input_scale,
                direction_degenerate=rc["direction_degenerate"] | rd["direction_degenerate"],
                scale_degenerate=rc["scale_degenerate"] | rd["scale_degenerate"])
    return {
        "A": a, "AB_diagonal_mean": 0.5*(a+b), "G_2x2": g,
        "M1": m1, "U_2": u2, "G_2x2_squared": g2, "W": w,
        "M0_vs_C": m0c, "M0_vs_D": m0d,
        "M0_vs_average": {
            **{name: .5*(m0c[name]+m0d[name]) for name in (
                "delta_mse", "delta_J", "normalized_delta_mse", "normalized_delta_J")},
            "input_scale": input_scale,
            "direction_degenerate": (m0c["direction_degenerate"] |
                                     m0d["direction_degenerate"]),
            "scale_degenerate": m0c["scale_degenerate"] | m0d["scale_degenerate"],
        },
        "M1_vs_C": rc, "M1_vs_D": rd, "M1_vs_average": ravg,
        "M2_vs_W": independent_risk(u2, g2, w, input_scale=input_scale,
                                     scale_power=2),
        "U_cross": u_cross(value),
    }
