"""Appendix C.1: transform battery on fixed scores (FMAS; EK-FAC IF for the listed rows).

Every transform acts per element (or per query column where stated) on the
FILED score matrix; nothing is re-selected. Rank = per-query rank of τ
(ascending); constant = τ ≡ 1; magnitude × random sign uses one RNG stream
per model seed; centred-then-squared = (τ − mean_q τ)²; squared-then-centred
= τ² − mean_q τ². Three-seed mean±std.
"""
import numpy as np
from scipy.stats import rankdata
from _common import SEEDS, TRACKS, Cell, fmt_ms, md_table, write_json, write_table

DS = "cifar2_5k"


def transforms(seed):
    rng = np.random.default_rng(20260901 + seed)
    return (("恒等", lambda a: a), ("绝对值", np.abs), ("平方", lambda a: a ** 2),
            ("保号平方", lambda a: np.sign(a) * a ** 2), ("三次方", lambda a: a ** 3), ("四次方", lambda a: a ** 4),
            ("秩", lambda a: rankdata(a, axis=0).astype(np.float64)), ("仅符号", np.sign),
            ("仅负部", lambda a: np.minimum(a, 0.0)), ("常数", lambda a: np.ones_like(a)),
            ("幅值×随机符号", lambda a: np.abs(a) * rng.choice([-1.0, 1.0], size=a.shape)),
            ("先去均值再平方 (τ−τ̄)²", lambda a: (a - a.mean(axis=0, keepdims=True)) ** 2),
            ("平方后去均值 τ²−mean(τ²)", lambda a: a ** 2 - (a ** 2).mean(axis=0, keepdims=True)))


EKFAC_ROWS = {"恒等", "平方", "保号平方", "三次方"}


def main():
    res, names = {}, [n for n, _ in transforms(42)]
    for m in ("fmas_raw", "ekfac_if"):
        for t in TRACKS:
            vals = {n: [] for n in names}
            for s in SEEDS:
                c = Cell(m, DS, t, s)
                for name, fn in transforms(s):
                    vals[name].append(c.lds(fn(c.a)))
            res[(m, t)] = vals
    rows = []
    for name in names:
        row = [name, fmt_ms(res[("fmas_raw", "gen")][name]), fmt_ms(res[("fmas_raw", "val")][name])]
        row += [fmt_ms(res[("ekfac_if", t)][name]) if name in EKFAC_ROWS else "—" for t in TRACKS]
        rows.append(row)
    write_table("tab_C1_battery", md_table(["变换", "FMAS gen", "FMAS val", "EK-FAC IF 对照 gen", "EK-FAC IF 对照 val"], rows),
                "附录 C.1 固定分数的变换电池（CIFAR-2，三种子 mean±std）")
    write_json("tab_C1_battery", {f"{m}|{t}": v for (m, t), v in res.items()})


if __name__ == "__main__":
    main()
