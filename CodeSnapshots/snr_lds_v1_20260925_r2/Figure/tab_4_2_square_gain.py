"""§4.2 table: does the square gain depend on the tail?

Full-matrix square gain  = LDS(τ²) − LDS(τ).
Head-5 % square gain     = LDS(τ² on the |τ|-head, 0 elsewhere)
                         − LDS(τ  on the same |τ|-head, 0 elsewhere).
The head is fixed by |τ| BEFORE squaring, so squaring cannot change the
membership; residual = head gain / full gain.
"""
import numpy as np
from _common import CORE, KAPPA_MAIN, SEEDS, TRACKS, Cell, fmt, head_mask, keep_only, md_table, write_json, write_table

DS = "cifar2_5k"


def main():
    rows, res = [], {}
    for m, label in CORE:
        g = {}
        for t in TRACKS:
            full, head = [], []
            for s in SEEDS:
                c = Cell(m, DS, t, s)
                hm = head_mask(c.a, c.k(KAPPA_MAIN), "abs")
                full.append(c.lds(c.a ** 2) - c.lds())
                head.append(c.lds(keep_only(c.a, hm, c.a ** 2)) - c.lds(keep_only(c.a, hm)))
            g[t] = dict(full=float(np.mean(full)), head=float(np.mean(head)),
                        residual_pct=100 * float(np.mean(head)) / float(np.mean(full)))
        res[m] = g
        rows.append([label, fmt(g["gen"]["full"], signed=True), fmt(g["gen"]["head"], signed=True),
                     fmt(g["val"]["full"], signed=True), fmt(g["val"]["head"], signed=True),
                     f"{g['gen']['residual_pct']:.1f}% / {g['val']['residual_pct']:.1f}%"])
    hdr = ["分数来源", "gen 全量平方增益", "gen 头部 5% 增益", "val 全量平方增益", "val 头部 5% 增益", "残余 gen / val"]
    write_table("tab_4_2_square_gain", md_table(hdr, rows, ["---"] + ["---:"] * 5),
                "§4.2 平方增益是否依赖尾部（固定 |τ| 头部 5%；三种子均值；CIFAR-2）")
    write_json("tab_4_2_square_gain", res)


if __name__ == "__main__":
    main()
