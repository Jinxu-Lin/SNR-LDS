# Figure 3(a) SNR fixed-selection controls

Date: 2026-09-25. Status: accepted and incorporated.

This CPU-only analysis updates Figure 3(a) from the historical density-based
selection to the accepted zero-mean Gaussian SNR rule at $\zeta=3$. It reuses
the accepted A3 selection masks, filed CIFAR-2 Flow Matching scores, 64 deletion
subsets, and measured responses for generation queries over seeds 42, 123, and
456. No model training, attribution scoring, background fitting, or threshold
selection was performed.

For every method and query, the accepted SNR mask is held fixed. Absolute-score
heads are defined from the pre-square scores and intersected with this mask.
Figure 3(a) reports linear and ordinary-squared readouts for all four methods
on identical selected samples. Valid queries are averaged within a seed,
followed by an equal average of the three seed means.

The recomputed full-head native values agree with every valid per-query SNR-LDS
value in the accepted A3 files to absolute tolerance $10^{-12}$.

| Method | Linear 5% | Linear 100% | Squared 5% | Squared 100% | Squared minus linear at 100% |
|---|---:|---:|---:|---:|---:|
| FMAS | 44.93 | 44.54 | 44.83 | 44.85 | +0.31 |
| EK-FAC IF | 43.46 | 43.31 | 42.57 | 42.81 | -0.50 |
| D-TRAK | 44.50 | 44.50 | 44.23 | 44.40 | -0.09 |
| DAS | 43.92 | 43.72 | 43.59 | 43.73 | +0.01 |

Outputs:

- `Paper/Figures/Fig3/snr-readouts.tex`
- `Reports/snr_lds_2026-09-25/fig3_snr_controls.json`
- `Reports/snr_lds_2026-09-25/render_fig3_snr_controls.py`

The independent-repeat SNR calibration remains pending and is outside this
accepted fixed-selection control.
