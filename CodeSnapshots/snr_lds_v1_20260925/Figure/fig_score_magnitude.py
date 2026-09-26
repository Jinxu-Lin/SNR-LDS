"""§3.1 figure + table: how unevenly the attribution magnitude is spread over training rows.

Object: the filed CIFAR-2 score matrices τ of the four same-source constructions
(FMAS, EK-FAC IF, D-TRAK, DAS linear term), shape (N=5000 training rows, Q=100 queries)
per track and model seed, i.e. the exact scores of §3.1 table 1.

Per-query normalization, and why it is required before pooling
  LDS only sees each query column up to a positive factor (Spearman of summed scores),
  and the columns differ in scale from query to query. Flattening the raw (N, Q)
  matrix would pool values measured in different units: a mixture of scales has heavy
  tails even when every single column is Gaussian, so the pooled shape would partly be
  an artifact. Each column is therefore divided by its RMS, r_q = sqrt(mean_i τ_qi²).
  This keeps the sign and the zero, fixes Σ_i (τ_qi / r_q)² = N, and makes the four
  methods (whose raw scales differ by orders of magnitude) share one x-axis.
  The table reports the cross-query spread of r_q (p90/p10) and the excess kurtosis
  with and without this step, as the evidence.

Density (figure)
  All N·Q normalized values of one (method, track, seed) are pooled (500,000 values).
  Gaussian KDE via linear binning on a fine symmetric grid and convolution with the
  kernel; one bandwidth for all curves (robust Silverman, median over cells). The line
  is the mean of the three seeds' densities, which equals the density of the pooled
  1.5 M values because the groups are equal-sized; the band is the seed min–max.
  The y-axis is logarithmic, otherwise only the spike at 0 is visible; the dashed curve
  is N(0, 1), the Gaussian with the same RMS. Values beyond the plotted range still
  count in the normalization; their fraction is written to the JSON.

Proportions (table), threshold-free
  Per query, rows sorted by |τ|: share of Σ|τ| and of Στ² held by the top 1 / 5 / 20 %
  and by the bottom 50 % of rows; averaged over the 100 queries, then mean±std over the
  three seeds. The Σ|τ| → Στ² change of the same row set is what squaring does to the
  relative weight. Plus the fraction of rows below 0.1 RMS and above 3 RMS.

Why not "sort each column, average across queries, plot the (5000,) vector": that
profile is a quantile average, not the distribution of any actual set of coordinates,
and it hides the per-query variation the band shows.

Run: PYTHONPATH=Codes/src python Codes/Figure/fig_score_magnitude.py
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis
from _common import (CORE, FIG_LABELS, METHOD_COLORS, SEEDS, TRACKS, load_scores, md_table, mpl, save_fig,
                     write_json, write_table)

DS = "cifar2_5k"
TOP = (0.01, 0.05, 0.20)
BOTTOM = 0.50
SMALL, LARGE = 0.1, 3.0


def normalize(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt((a ** 2).mean(axis=0, keepdims=True))
    return a / r, r.ravel()


def shares(a: np.ndarray) -> dict:
    """Mean over queries of the Σ|τ| and Στ² shares of the top/bottom |τ|-ranked rows."""
    N = a.shape[0]
    mag = -np.sort(-np.abs(a), axis=0)                         # descending per column
    out = {}
    for power, key in ((1, "abs"), (2, "sq")):
        w = mag ** power
        tot = w.sum(axis=0)
        cum = np.cumsum(w, axis=0)
        for f in TOP:
            k = int(round(f * N))
            out[f"top{round(f * 100)}_{key}"] = float(np.mean(cum[k - 1] / tot))
        kb = int(round(BOTTOM * N))
        out[f"bottom{round(BOTTOM * 100)}_{key}"] = float(np.mean((tot - cum[N - kb - 1]) / tot))
    return out


def robust_bw(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return 0.9 * min(x.std(), (q75 - q25) / 1.34) * len(x) ** (-0.2)


def binned_density(x: np.ndarray, lim: float, bw: float, nbins: int = 8001):
    edges = np.linspace(-lim, lim, nbins + 1)
    width = edges[1] - edges[0]
    counts, _ = np.histogram(x, bins=edges)
    dens = gaussian_filter1d(counts.astype(np.float64), sigma=bw / width, mode="constant") / (len(x) * width)
    return 0.5 * (edges[:-1] + edges[1:]), dens, float(np.mean(np.abs(x) >= lim))


def main():
    cells = {}
    for m, _ in CORE:
        for t in TRACKS:
            for s in SEEDS:
                a = load_scores(m, DS, t, s)
                x, r = normalize(a)
                raw_pooled = (a / np.sqrt((a ** 2).mean())).ravel()          # one global scale: the mixture
                cells[(m, t, s)] = dict(
                    x=x.ravel(), shares=shares(a),
                    frac_small=float(np.mean(np.abs(x) < SMALL)), frac_large=float(np.mean(np.abs(x) > LARGE)),
                    kurt_norm=float(kurtosis(x.ravel(), fisher=True)), kurt_raw=float(kurtosis(raw_pooled, fisher=True)),
                    rms_p90_p10=float(np.percentile(r, 90) / np.percentile(r, 10)),
                    frac_pos=float(np.mean(a > 0)))

    # ---- shared plotting range and bandwidth ----
    allx = np.concatenate([c["x"] for c in cells.values()])
    lim = float(np.ceil(np.percentile(np.abs(allx), 99.9)))
    bw = float(np.median([robust_bw(c["x"]) for c in cells.values()]))
    del allx

    dens = {}
    for (m, t, s), c in cells.items():
        grid, d, outside = binned_density(c["x"], lim, bw)
        dens[(m, t, s)] = d
        c["outside_plot_range"] = outside

    # ---- tables ----
    def ms(vals, pct=True):
        v = np.asarray(vals, dtype=float) * (100 if pct else 1)
        return f"{v.mean():.1f}±{v.std(ddof=1):.1f}%" if pct else f"{v.mean():.2f}±{v.std(ddof=1):.2f}"

    keys = [f"top{round(f * 100)}" for f in TOP] + [f"bottom{round(BOTTOM * 100)}"]
    names = {f"top{round(f * 100)}": f"头部 {round(f * 100)}%" for f in TOP} | {"bottom50": "尾部 50%"}
    rows = []
    for m, label in CORE:
        for t in TRACKS:
            cs = [cells[(m, t, s)] for s in SEEDS]
            rows.append([label, t] + [ms([c["shares"][f"{k}_abs"] for c in cs]) for k in keys]
                        + [ms([c["shares"][f"{k}_sq"] for c in cs]) for k in keys])
    hdr = ["方法", "轨"] + [f"{names[k]} 占 Σ|τ|" for k in keys] + [f"{names[k]} 占 Στ²" for k in keys]
    write_table("tab_score_magnitude_shares", md_table(hdr, rows, ["---", "---"] + ["---:"] * 8),
                "§3.1 幅值集中度：逐查询按 |τ| 排序后各排名段占总幅值与平方能量的比例（CIFAR-2，查询平均，三种子 mean±std）")
    rows = []
    for m, label in CORE:
        for t in TRACKS:
            cs = [cells[(m, t, s)] for s in SEEDS]
            rows.append([label, t, ms([c["frac_small"] for c in cs]), ms([c["frac_large"] for c in cs]),
                         ms([c["frac_pos"] for c in cs]), ms([c["kurt_norm"] for c in cs], pct=False),
                         ms([c["kurt_raw"] for c in cs], pct=False), ms([c["rms_p90_p10"] for c in cs], pct=False)])
    write_table("tab_score_magnitude_shape",
                md_table(["方法", "轨", f"|τ|<{SMALL} RMS 的行", f"|τ|>{LARGE:g} RMS 的行", "正分数行",
                          "超额峰度（逐查询 RMS 归一）", "超额峰度（不归一直接混合）", "逐查询 RMS 的 p90/p10"],
                         rows, ["---", "---"] + ["---:"] * 6),
                "§3.1 分布形状与归一化依据（CIFAR-2，三种子 mean±std）")

    write_json("fig_score_magnitude", {
        "plot_range_rms": lim, "bandwidth_rms": bw,
        "cells": {f"{m}|{t}|{s}": {k: v for k, v in c.items() if k != "x"} for (m, t, s), c in cells.items()}})

    # ---- figure ----
    plt = mpl()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    ref = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    for ax, t in zip(axes, TRACKS):
        for m, _ in CORE:
            D = np.stack([dens[(m, t, s)] for s in SEEDS])
            keep = D.mean(axis=0) > 1e-4          # hide the sparse far tail (few points per kernel width)
            ax.fill_between(grid[keep], D.min(axis=0)[keep], D.max(axis=0)[keep], color=METHOD_COLORS[m], alpha=0.15, lw=0)
            ax.plot(grid[keep], D.mean(axis=0)[keep], color=METHOD_COLORS[m], lw=1.5)
        ax.plot(grid, ref, color="#8b939a", ls="--", lw=1.1)
        ax.axvline(0, color="#c9cfd4", lw=0.8, ls=":")
        ax.set(yscale="log", xlim=(-lim, lim), ylim=(1e-4, None), xlabel="Attribution score / per-query RMS",
               title=f"CIFAR-2, {t} queries")
        ax.grid(color="#e9ecee", lw=0.6)
    axes[0].set_ylabel("Density (log scale)")
    handles = [Line2D([], [], color=METHOD_COLORS[m], lw=1.8, label=FIG_LABELS[m]) for m, _ in CORE]
    handles.append(Line2D([], [], color="#8b939a", ls="--", lw=1.1, label="N(0, 1), same RMS"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, "fig-score-magnitude")
    print(f"plot range ±{lim:g} RMS, bandwidth {bw:.4f} RMS")


if __name__ == "__main__":
    main()
