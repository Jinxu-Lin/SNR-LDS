"""Compute the accepted common RMS-density bandwidth for final Figures 1(a)/4.

This pooled illustrative density is distinct from the per-query SNR background fit.
The published calculation uses population SD inside the pooled Silverman rule.
"""
import numpy as np
from _common import CORE, SEEDS, TRACKS, load_scores, write_json

def robust_bw(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return 0.9 * min(x.std(), (q75 - q25) / 1.34) * len(x) ** (-0.2)



def main():
    cells = {}
    for method, _ in CORE:
        for track in TRACKS:
            for seed in SEEDS:
                raw = load_scores(method, "cifar2_5k", track, seed)
                cells[(method, track, seed)] = (raw / np.sqrt(np.mean(raw**2, axis=0, keepdims=True))).ravel()
    pooled = np.concatenate(list(cells.values()))
    settings = {"plot_range_rms": float(np.ceil(np.percentile(np.abs(pooled), 99.9))),
                "bandwidth_rms": float(np.median([robust_bw(x) for x in cells.values()])),
                "normalization": "per-query RMS", "seeds": SEEDS, "tracks": TRACKS}
    print(write_json("fig_score_magnitude", settings))

if __name__ == "__main__":
    main()
