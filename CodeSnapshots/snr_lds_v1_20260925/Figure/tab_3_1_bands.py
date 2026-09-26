"""M1 magnitude-band LDS. Complete R16 is in balds-repeat, not this script."""
import itertools
import numpy as np
from _common import (BANDS, CORE, SEEDS, TRACKS, Cell, band_mask, fmt, fmt_ms, head_mask, keep_only,
                     md_table, write_json, write_table)

DS = "cifar2_5k"
HEADS = (5, 10, 20)
SHOULDER = ((5, 50), (50, 100))


def exp1():
    rows, res = [], {}
    for m, label in CORE:
        cell = {t: {} for t in TRACKS}
        for t in TRACKS:
            cs = [Cell(m, DS, t, s) for s in SEEDS]
            for kp in HEADS:
                cell[t][f"head{kp}"] = [c.lds(keep_only(c.a, head_mask(c.a, c.k(kp), "abs"))) for c in cs]
            for lo, hi in SHOULDER:
                cell[t][f"band{lo}_{hi}"] = [c.lds(keep_only(c.a, band_mask(c.a, lo, hi))) for c in cs]
            cell[t]["full"] = [c.lds() for c in cs]
        res[m] = cell
    keys = [(f"head{k}", f"幅值头部 {k}%") for k in HEADS] + [("band5_50", "仅排名 5%–50%"),
                                                              ("band50_100", "仅排名 50%–100%"), ("full", "全量")]
    for m, label in CORE:
        rows = [[name, fmt_ms(res[m]["gen"][k]), fmt_ms(res[m]["val"][k])] for k, name in keys]
        write_table(f"tab_3_1_exp1_{m}", md_table(["求和范围", "gen", "val"], rows, ["---", "---:", "---:"]),
                    f"§3.1 实验一：{label}，按 |τ| 排名带保留原值（CIFAR-2，三种子 mean±std）")
    write_json("tab_3_1_exp1", res)






def main():
    exp1()


if __name__ == "__main__":
    main()
