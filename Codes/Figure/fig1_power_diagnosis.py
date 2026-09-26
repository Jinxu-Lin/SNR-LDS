"""Compute final Figures 2(b)/6: four fixed C2 methods, signed/folded p=0:.2:4."""
import numpy as np
from _common import CORE, SEEDS, TRACKS, Cell, write_json
P_GRID = [round(float(x), 1) for x in np.arange(0, 4.001, .2)]

def main():
    curves = {"p_grid": P_GRID}
    for method, label in CORE:
        for track in TRACKS:
            for seed in SEEDS:
                cell = Cell(method, "cifar2_5k", track, seed)
                a = cell.a / np.maximum(np.abs(cell.a).max(axis=0, keepdims=True), 1e-300)
                for family in ("signed", "fold"):
                    values = [cell.lds(np.abs(a)**p * (np.sign(a) if family == "signed" else 1)) for p in P_GRID]
                    curves[f"{method}|{track}|{seed}|{family}"] = {"lds": values, "label": label}
    print(write_json("fig1_power_c2", curves))

if __name__ == "__main__":
    main()
