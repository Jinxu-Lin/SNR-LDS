# Figure 1(b): relative sampling variation

Figure 1(b) now reports the bin-level ratio

`sum_i sample_variance_i / sum_i repeat_mean_i^2`

for ten equal-sized FMAS score-magnitude bins and 16 generation queries. Samples
are ranked within each query by the absolute mean over the 16 repeated score
estimates. The plotted bar is the mean query-level ratio. No smoothing or
reordering is applied.

The values from low to high score magnitude are 19.7016, 2.9420, 1.0356,
0.5176, 0.3163, 0.2080, 0.1456, 0.1042, 0.0863, and 0.0452. The log vertical
axis displays the 436-fold range without clipping the smaller bins.

`fmas_varratio_deciles.csv` records the plotted statistics. The appendix
pilot-band ratios in `pilot_band_varratio.csv` were recomputed from the accepted
coordinate means, sample standard deviations, and independent-pilot band IDs.
The accepted band-only LDS values were not changed.

Reproduce with:

```bash
/path/to/miniconda3/envs/da/bin/python \
  /path/to/BA-LDS/Reports/figure_split_2026-09-25/render_relative_variation.py
```
