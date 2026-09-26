"""Figure 1 (§3.1, gen), Figure C.1 (val) and Table C.4: terminal power response.

For a fixed score matrix τ (no λ/ϱ re-selection), evaluate LDS of the folded
family |τ|^p and the signed family sign(τ)|τ|^p on p ∈ {0, 0.2, …, 4}
(plus the legacy anchors 6 and 8 for the C.1 sparse grid). Per query the
column is first divided by max|τ| — a positive per-query constant, which
leaves every Spearman rank unchanged and only avoids float overflow at high p.
Peak position: grid argmax, refined by a parabola through the three grid
points around it (descriptive only).

CIFAR-2: 13 score constructions × 3 seeds × 2 tracks; CIFAR-10: the 4 core
constructions × 3 seeds × 2 tracks (three-chain-average GT).
"""
import numpy as np
from _common import (CORE, FIG_LABELS, SEEDS, TRACKS, Cell, fmt, has_scores, md_table, mpl, save_fig,
                     write_json, write_table)

P_GRID = [round(x, 1) for x in np.arange(0.0, 4.001, 0.2)]
P_LEGACY = [6.0, 8.0]
EXTRA = (("ekfac_mean", "EK-FAC 均值读出", "读出轴"), ("ekfac_msl2", "EK-FAC 平方型读出", "读出轴"),
         ("trak_T100", "TRAK", "基线"), ("gas_T100", "GAS", "基线"), ("tracincp_T100", "TracInCP", "基线"),
         ("relative_if_T100", "Relative IF", "基线"), ("renorm_if_T100", "Renorm. IF", "基线"),
         ("grad_dot_T100", "Gradient dot", "基线"), ("journey_trak_T100", "Journey-TRAK", "基线"))
ALL = [(m, l, "正文") for m, l in CORE] + list(EXTRA)
FAMILY_COLORS = {"fold": ["#245c8b", "#4e88b6", "#94bad6"], "signed": ["#985120", "#c77b3d", "#e3b387"]}


def refine_argmax(ps, ys):
    i = int(np.argmax(ys))
    if 0 < i < len(ps) - 1:
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        den = y0 - 2 * y1 + y2
        d = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
        return float(ps[i]), float(ps[i] + d * (ps[i] - ps[i - 1]))
    return float(ps[i]), float(ps[i])


def sweep(c: Cell, family: str, grid) -> list[float]:
    a = c.a / np.maximum(np.abs(c.a).max(axis=0, keepdims=True), 1e-300)
    mag, sg = np.abs(a), np.sign(a)
    return [c.lds(mag ** p if family == "fold" else sg * mag ** p) for p in grid]


def run_dataset(ds, methods, seeds):
    curves = {}
    for m, label, group in methods:
        for track in TRACKS:
            for seed in seeds:
                if not has_scores(m, ds, track, seed):
                    continue
                c = Cell(m, ds, track, seed)
                for fam in ("fold", "signed"):
                    ys = sweep(c, fam, P_GRID + P_LEGACY)
                    g, r = refine_argmax(P_GRID, ys[: len(P_GRID)])
                    curves[(m, track, seed, fam)] = dict(lds=ys, argmax_grid=g, argmax_refined=r,
                                                         label=label, group=group)
    return curves


def power_figure(curves, track, name):
    plt = mpl()
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(13.4, 6.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.44)
    left = outer[0].subgridspec(2, 2, wspace=0.24, hspace=0.38)
    for idx, (m, label) in enumerate(CORE):
        ax = fig.add_subplot(left[idx // 2, idx % 2])
        for fam, marker in (("fold", "o"), ("signed", "^")):
            for j, seed in enumerate(SEEDS):
                ys = np.asarray(curves[(m, track, seed, fam)]["lds"])[: len(P_GRID)]
                ax.plot(P_GRID, ys, color=FAMILY_COLORS[fam][j], lw=1.35, ls="-" if fam == "fold" else "--")
                pk = int(np.argmax(ys))
                ax.plot(P_GRID[pk], ys[pk], marker, ms=3.2, color=FAMILY_COLORS[fam][j], mfc="white")
        ax.axvline(2, color="#b7bec4", lw=0.8, ls=":")
        ax.set(title=FIG_LABELS[m], xlabel="Exponent p", ylabel="Full LDS", xlim=(0, 4))
        ax.grid(color="#e9ecee", lw=0.6)
    ax = fig.add_subplot(outer[1])
    rows = [(m, FIG_LABELS[m]) for m, _, _ in ALL if (m, track, SEEDS[0], "fold") in curves]
    for row, (m, label) in enumerate(rows):
        y = len(rows) - 1 - row
        ax.axhline(y, color="#f0f2f4", lw=7, zorder=0)
        for fam, marker, shift in (("fold", "o", -0.16), ("signed", "^", 0.16)):
            for j, seed in enumerate(SEEDS):
                if (m, track, seed, fam) not in curves:
                    continue
                ax.plot(curves[(m, track, seed, fam)]["argmax_refined"], y + shift + (j - 1) * 0.055, marker,
                        ms=4.4, color=FAMILY_COLORS[fam][j], mfc=FAMILY_COLORS[fam][j] if fam == "fold" else "white")
    ax.set(yticks=range(len(rows)), yticklabels=[l for _, l in rows][::-1], xlabel="Refined peak exponent p*",
           xlim=(-0.1, 3.6), title="Peak shifts across score constructions")
    ax.axvline(2, color="#b7bec4", lw=0.8, ls=":")
    ax.grid(axis="x", color="#e9ecee", lw=0.6)
    handles = [Line2D([], [], color=FAMILY_COLORS["fold"][0], marker="o", ms=4, label="Folded |τ|^p"),
               Line2D([], [], color=FAMILY_COLORS["signed"][0], marker="^", mfc="white", ls="--", ms=4,
                      label="Signed sign(τ)|τ|^p")]
    handles += [Line2D([], [], color=f"#{x:02x}{x:02x}{x:02x}", lw=2, label=f"Seed {s}")
                for s, x in zip(SEEDS, [65, 125, 185])]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.025))
    fig.suptitle(f"Terminal power response and peak location | CIFAR-2, {track}", y=1.01)
    fig.subplots_adjust(bottom=0.10)
    save_fig(fig, name)


def argmax_table(curves, methods, seeds, name, title):
    rows = []
    for m, label, group in methods:
        if (m, "gen", seeds[0], "fold") not in curves:
            continue
        cells = [label, group]
        for track in TRACKS:
            for fam in ("fold", "signed"):
                ks = [(m, track, s, fam) for s in seeds if (m, track, s, fam) in curves]
                if not ks:
                    cells += ["—", "—"]
                    continue
                cells.append("/".join(f"{curves[k]['argmax_grid']:g}" for k in ks))
                r = [curves[k]["argmax_refined"] for k in ks]
                cells.append(f"{min(r):.2f}–{max(r):.2f}")
        rows.append(cells)
    hdr = ["方法", "组", "gen 折叠 argmax（逐种子）", "gen 折叠 p* 范围", "gen 保号 argmax", "gen 保号 p* 范围",
           "val 折叠 argmax", "val 折叠 p* 范围", "val 保号 argmax", "val 保号 p* 范围"]
    write_table(name, md_table(hdr, rows), title)


def sparse_grid_table(curves, name):
    """C.1 sparse high-power grid for FMAS: p ∈ {0.5,1,1.5,2,3,4,6,8}, gen."""
    grid = P_GRID + P_LEGACY
    rows = []
    for p in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        # 0.5 and 1.5 are not on the 0.2 grid: evaluate them directly
        cells = [f"{p:g}"]
        for fam in ("fold", "signed"):
            vals = []
            for seed in SEEDS:
                key = ("fmas_raw", "gen", seed, fam)
                if p in grid:
                    vals.append(curves[key]["lds"][grid.index(p)])
                else:
                    vals.append(sweep(Cell("fmas_raw", "cifar2_5k", "gen", seed), fam, [p])[0])
            cells += [fmt(float(np.mean(vals))), "/".join(fmt(v) for v in vals)]
        rows.append(cells)
    write_table(name, md_table(["p", "折叠 mean", "折叠逐种子（42/123/456）", "保号 mean", "保号逐种子"], rows),
                "附录 C.1 FMAS 高幂稀疏网格（CIFAR-2 gen，三种子）")


def main():
    c2 = run_dataset("cifar2_5k", ALL, SEEDS)
    write_json("fig1_power_c2", {"|".join(map(str, k)): v for k, v in c2.items()} | {"p_grid": P_GRID + P_LEGACY})
    power_figure(c2, "gen", "fig1-power-diagnosis")
    power_figure(c2, "val", "figC1-power-diagnosis-val")
    argmax_table(c2, ALL, SEEDS, "tab_C4_argmax_cifar2", "附录 C.4 CIFAR-2 峰位（网格 argmax 逐种子；细化范围）")
    sparse_grid_table(c2, "tab_C1_power_sparse")
    c10 = run_dataset("cifar10_v2", [(m, l, "正文") for m, l in CORE], SEEDS)
    write_json("fig1_power_c10", {"|".join(map(str, k)): v for k, v in c10.items()} | {"p_grid": P_GRID + P_LEGACY})
    argmax_table(c10, [(m, l, "正文") for m, l in CORE], SEEDS, "tab_C4_argmax_cifar10",
                 "附录 C.4 CIFAR-10 峰位（三链平均真值；逐种子 42/123/456）")


if __name__ == "__main__":
    main()
