"""Compute final Figure 5: fixed absolute heads with signed linear readout."""
from _common import CORE, SEEDS, TRACKS, KAPPA_GRID, Cell, head_mask, keep_only, write_json

def main():
    curves = {"kappa_grid": KAPPA_GRID, "selection": "ceil(kappa*N), stable ascending-index ties"}
    for method, _ in CORE:
        for track in TRACKS:
            values = []
            for seed in SEEDS:
                cell = Cell(method, "cifar2_5k", track, seed)
                values.append([cell.lds(keep_only(cell.a, head_mask(cell.a, cell.k(k), "abs"))) for k in KAPPA_GRID])
            curves[f"{method}|{track}|abs"] = values
    print(write_json("fig2_head_profile", curves))

if __name__ == "__main__":
    main()
