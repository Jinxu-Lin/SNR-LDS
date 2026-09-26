"""Figure 2 (§4.2) + the κ table it is drawn from: head-protocol LDS curves.

For each core score construction, track and seed, keep the top-κ rows per
query (κ ∈ KAPPA_GRID, from 5 % to 100 %), by support direction (``pos``,
largest τ — the main protocol) and by magnitude (``abs``, largest |τ| — the
diagnostic contrast), sum the ORIGINAL values of the kept rows, and evaluate
LDS. κ = 100 % is the ordinary full LDS in both modes.

The table and the figure are written from the same array, so they cannot
disagree. R16 repeatability is reproduced separately by balds-repeat.
"""
import numpy as np
from _common import (CORE, FIG_LABELS, KAPPA_GRID, SEEDS, TRACKS, Cell, METHOD_COLORS, fmt_ms, head_mask,
                     keep_only, md_table, mpl, save_fig,
                     write_json, write_table)

DS = "cifar2_5k"
MODES = ("pos", "abs")


def main():
    curves = {}                                   # (method, track, mode) -> (3, len(KAPPA)) per seed
    for method, _ in CORE:
        for track in TRACKS:
            per_seed = {mode: [] for mode in MODES}
            for seed in SEEDS:
                c = Cell(method, DS, track, seed)
                for mode in MODES:
                    per_seed[mode].append([c.lds(keep_only(c.a, head_mask(c.a, c.k(kp), mode)))
                                           for kp in KAPPA_GRID])
            for mode in MODES:
                curves[(method, track, mode)] = np.asarray(per_seed[mode])
    # ---- tables ----
    hdr = ["分数来源", "轨"] + [f"{k}%" for k in KAPPA_GRID]
    for mode, title in (("pos", "支持方向头部（主协议 LDS@κ）"), ("abs", "幅值头部（诊断对照 LDS_abs@κ）")):
        rows = [[label, track] + [fmt_ms(curves[(m, track, mode)][:, i]) for i in range(len(KAPPA_GRID))]
                for m, label in CORE for track in TRACKS]
        write_table(f"tab_head_kappa_{mode}", md_table(hdr, rows, ["---", "---"] + ["---:"] * len(KAPPA_GRID)),
                    f"{title}，原值求和，三种子 mean±std，CIFAR-2")
    write_json("fig2_head_profile", {f"{m}|{t}|{mode}": curves[(m, t, mode)] for (m, t, mode) in curves}
               | {"kappa_grid": KAPPA_GRID})

    # ---- figure ----
    plt = mpl()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    x = np.asarray(KAPPA_GRID)
    for ax, track in zip(axes[:2], TRACKS):
        for m, label in CORE:
            for mode, style in (("pos", "-"), ("abs", "--")):
                ax.plot(x, curves[(m, track, mode)].mean(axis=0), style, color=METHOD_COLORS[m],
                        lw=1.5, marker="o", ms=2.7)
        ax.set(xlabel="Retained training rows per query (%)", ylabel="LDS",
               title=f"Original values, {track}", xticks=x)
        ax.axvline(5, color="#bec6cd", ls=":", lw=0.8)
        ax.grid(color="#e9ecee", lw=0.6)
    handles = [Line2D([], [], color=METHOD_COLORS[m], label=FIG_LABELS[m], lw=1.7) for m, _ in CORE]
    handles += [Line2D([], [], color="#5d656d", lw=1.5, label="Support direction (solid)"),
                Line2D([], [], color="#5d656d", ls="--", lw=1.5, label="Absolute value (dashed)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.43, -0.095))
    fig.tight_layout(w_pad=2)
    save_fig(fig, "fig2-head-profile")


if __name__ == "__main__":
    main()
