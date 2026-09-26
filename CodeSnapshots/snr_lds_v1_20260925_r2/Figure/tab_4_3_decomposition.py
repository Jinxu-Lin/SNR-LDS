"""§4.3 table (FMAS, 7 rows) and appendix C.6 (all 10 predictors, 4 constructions, gen/val).

Predictors at k = 5 % of N per query (H_a = |τ|-head, H_+ = support head, T = complement):
 1  |τ|-head, original values                       τ on H_a, 0 on T_a
 2  same head, absolute values                      |τ| on H_a
 3  support head, original values                   τ on H_+, 0 on T_+
 4  full square                                      τ² everywhere
 5  |τ|-head square + constant tail                  τ² on H_a, c_{2,q}=mean_{T_a} τ² on T_a
 6  |τ|-head square only                             τ² on H_a
 7  full positive-part square                        [τ]_+² everywhere
 8  support head [τ]_+² + constant tail              [τ]_+² on H_+, mean_{T_+}[τ]_+² on T_+
 9  support head [τ]_+² only
10  support head original values + constant tail     τ on H_+, c_{1,q}=mean_{T_+}[τ]_+ on T_+
Ratios: R_sq = (5−6)/(4−6), R_pos = (8−9)/(7−9), per method/track on seed means.
"""
import numpy as np
from _common import CORE, KAPPA_MAIN, SEEDS, TRACKS, Cell, const_tail, fmt_ms, head_mask, keep_only, md_table, write_json, write_table

DS = "cifar2_5k"
NAMES = {1: "幅值头部原值", 2: "同一幅值头部取绝对值", 3: "支持方向头部原值", 4: "全量平方",
         5: "幅值头部平方＋常数尾", 6: "仅幅值头部平方", 7: "全量正分支平方",
         8: "支持头部正分支平方＋常数尾", 9: "仅支持头部正分支平方", 10: "支持头部原值＋正部均值常数尾"}
ORDER_43 = (1, 2, 3, 4, 6, 5, 10)          # the §4.3 seven-row order


def predictors(a, k):
    hA, hP = head_mask(a, k, "abs"), head_mask(a, k, "pos")
    sq, ps = a ** 2, np.clip(a, 0, None) ** 2
    return {1: keep_only(a, hA), 2: keep_only(a, hA, np.abs(a)), 3: keep_only(a, hP), 4: sq,
            5: const_tail(a, hA, sq, lambda x: x ** 2), 6: keep_only(a, hA, sq), 7: ps,
            8: const_tail(a, hP, ps, lambda x: np.clip(x, 0, None) ** 2), 9: keep_only(a, hP, ps),
            10: const_tail(a, hP, a, lambda x: np.clip(x, 0, None))}


def main():
    res = {}
    for m, label in CORE:
        for t in TRACKS:
            per = {i: [] for i in NAMES}
            for s in SEEDS:
                c = Cell(m, DS, t, s)
                P = predictors(c.a, c.k(KAPPA_MAIN))
                for i in NAMES:
                    per[i].append(c.lds(P[i]))
            res[(m, t)] = per
    # §4.3 FMAS table
    rows = [[NAMES[i], fmt_ms(res[("fmas_raw", "gen")][i]), fmt_ms(res[("fmas_raw", "val")][i])] for i in ORDER_43]
    write_table("tab_4_3_fmas", md_table(["FMAS 预测器", "gen LDS", "val LDS"], rows, ["---", "---:", "---:"]),
                "§4.3 FMAS 七种预测器（k=5%，三种子 mean±std；CIFAR-2）")
    # C.6 gen / val
    for t in TRACKS:
        rows = [[f"{i}. {NAMES[i]}"] + [fmt_ms(res[(m, t)][i]) for m, _ in CORE] for i in NAMES]
        write_table(f"tab_C6_{t}", md_table(["预测器"] + [l for _, l in CORE], rows, ["---"] + ["---:"] * 4),
                    f"附录 C.6 十种预测器，{t} 轨（k=250，三种子 mean±std）")
    # ratios and pairwise facts on seed means
    lines = []
    for m, label in CORE:
        for t in TRACKS:
            mu = {i: float(np.mean(res[(m, t)][i])) for i in NAMES}
            rsq = (mu[5] - mu[6]) / (mu[4] - mu[6]) if abs(mu[4] - mu[6]) >= 0.005 else float("nan")
            rpos = (mu[8] - mu[9]) / (mu[7] - mu[9]) if abs(mu[7] - mu[9]) >= 0.005 else float("nan")
            lines.append([label, t, f"{rsq:.3f}", f"{rpos:.3f}", f"{mu[3] - mu[1]:+.4f}", f"{mu[2] - mu[1]:+.4f}",
                          f"{mu[10] - mu[4]:+.4f}", "是" if mu[10] > mu[4] else "否"])
    write_table("tab_C6_ratios", md_table(["方法", "轨", "R_sq=(5−6)/(4−6)", "R_pos=(8−9)/(7−9)", "行3−行1",
                                           "行2−行1", "行10−行4", "行10>行4"], lines),
                "附录 C.6 比值与配对差（三种子均值；|分母|<0.005 记 nan）")
    write_json("tab_4_3_decomposition", {f"{m}|{t}": v for (m, t), v in res.items()})


if __name__ == "__main__":
    main()
