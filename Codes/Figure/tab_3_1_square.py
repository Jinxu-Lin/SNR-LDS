"""Final appendix Table 5: linear term vs terminal square, four constructions, two tracks.

Same score matrix; ``终端平方`` = τ² (folded square), ``保号平方`` = sign(τ)τ²
reported alongside for reference. No λ/ϱ re-selection (the ME-2 rule of the
CIFAR-2 battery). mean±std over the three model seeds.
"""
import numpy as np
from _common import CORE, SEEDS, TRACKS, Cell, fmt_ms, md_table, write_json, write_table

DS = "cifar2_5k"


def main():
    res, rows = {}, []
    for m, label in CORE:
        vals = {t: {"identity": [], "fold": [], "signed": []} for t in TRACKS}
        for t in TRACKS:
            for s in SEEDS:
                c = Cell(m, DS, t, s)
                vals[t]["identity"].append(c.lds())
                vals[t]["fold"].append(c.lds(c.a ** 2))
                vals[t]["signed"].append(c.lds(np.sign(c.a) * c.a ** 2))
        res[m] = vals
        rows.append([label, fmt_ms(vals["gen"]["identity"]), fmt_ms(vals["val"]["identity"]),
                     fmt_ms(vals["gen"]["fold"]), fmt_ms(vals["val"]["fold"]),
                     fmt_ms(vals["gen"]["signed"]), fmt_ms(vals["val"]["signed"])])
    hdr = ["Method", "一次项 gen", "一次项 val", "终端平方 gen", "终端平方 val", "保号平方 gen", "保号平方 val"]
    write_table("tab_3_1_square", md_table(hdr, rows, ["---"] + ["---:"] * 6),
                "附录 Table 5：同一分数的一次项与终端平方（CIFAR-2，三种子 mean±std，不重选 λ/ϱ）")
    # relative gain of the folded square over the linear term
    gains = [[label, f"{100 * (np.mean(res[m]['gen']['fold']) / np.mean(res[m]['gen']['identity']) - 1):+.1f}%",
              f"{100 * (np.mean(res[m]['val']['fold']) / np.mean(res[m]['val']['identity']) - 1):+.1f}%"]
             for m, label in CORE]
    write_table("tab_3_1_square_gain_rel", md_table(["Method", "gen 相对增益", "val 相对增益"], gains),
                "终端平方相对一次项的相对增益（三种子均值之比）")
    write_json("tab_3_1_square", res)


if __name__ == "__main__":
    main()
