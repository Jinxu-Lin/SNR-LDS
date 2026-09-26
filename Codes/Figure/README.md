# Final manuscript figures

These CPU tools read filed scores, masks and responses. They never train models or write `Paper/`. Install from the repository root with `python -m pip install -e './Codes[train,figures]'` (Table10 uses Torch and the dataset loader; density/head/power analysis uses NumPy/SciPy).

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
export BALDS_FIGURE_ROOT="$BALDS_DATA_ROOT/results/paper/figures"
```

`_paths.py` resolves those two roots; regenerated numeric metadata goes to `$BALDS_FIGURE_ROOT/metadata`. Renderers prefer regenerated metadata; if absent, they explicitly read the accepted JSON snapshots in `$BALDS_DATA_ROOT/results/paper_figures/legacy`. Historical absolute references containing `_Data` are rebased only to the chosen data root. There is no fallback to a source checkout.

## Active producers

| Final assets | Numeric calculation | Rendering |
|---|---|---|
| Fig1(a), Fig4 | `fig_score_magnitude.py` | `render_fig1_density.py`, `render_appendix_density.py` |
| Fig1(b), Table4 band statistics | R16 score index or `balds-repeat summarize` | `render_relative_variation.py`; metadata CSVs |
| Fig2(a,c), Fig5 | `fig2_head_profile.py`, `compute_abs_heads.py` | `render_fig2.py`, `render_fig5.py` |
| Fig2(b), Fig6 | `fig1_power_diagnosis.py` | `render_fig2.py`, `render_core_power.py` |
| Table5 | `tab_3_1_square.py` | Numeric JSON and Markdown table |
| Table10 | `tab_C3_ddpm.py` | Numeric per-query JSON and Markdown table |

Run the calculation and rendering stages in order:

```bash
python Codes/Figure/fig_score_magnitude.py
python Codes/Figure/render_fig1_density.py
python Codes/Figure/render_appendix_density.py
python Codes/Figure/render_relative_variation.py
python Codes/Figure/fig2_head_profile.py
python Codes/Figure/compute_abs_heads.py
python Codes/Figure/fig1_power_diagnosis.py
python Codes/Figure/render_fig2.py
python Codes/Figure/render_fig5.py
python Codes/Figure/render_core_power.py
python Codes/Figure/tab_3_1_square.py
HF_DATASETS_OFFLINE=1 python Codes/Figure/tab_C3_ddpm.py
```

For newly generated R16 analysis, set `BALDS_R16_ROOT="$BALDS_DATA_ROOT/results/repeatability_r16/analysis"` before its renderer; unset it to use the archived accepted index. The new root must contain all four method/track `coordinates.npz` files. The historic path uses all16 FMAS generation score matrices and all four `coordinate_stats.pt` files. Variance is sample variance, and VarRatio is the sum of variances divided by the sum of squared repeat means; the main10 equal bins ascend in absolute repeat mean.

Main head calculations use `ceil(kappa*N)` and stable ascending training-row index ties. Both readouts share the same absolute pre-square head; DAS is squared once only for its native readout. Fig5 is linear for all methods. Powers are restricted to the four core methods, signed/folded families, and p=0,.2,...,4.

Table10 retains fixed projected lambda10, IF1e-12, FMAS gen1e-7/val1e-8. The imported1000-query feature matrix is an internal reconstruction dependency; only matched100 queries and first64 subsets are reported. No damping search or retired1000-query table is executed.

## Other active assets

The experiment-level tools own SNR and intervention outputs:

```bash
python Codes/tools/aggregation_controls.py transforms --data-root "$BALDS_DATA_ROOT" --output "$BALDS_DATA_ROOT/results/paper/matched_das"
python Codes/tools/render_matched_das.py --input "$BALDS_DATA_ROOT/results/paper/matched_das" --output "$BALDS_FIGURE_ROOT"
python Codes/tools/render_snr_tables.py --output "$BALDS_FIGURE_ROOT"
python Codes/tools/render_deletion_tables.py --output "$BALDS_FIGURE_ROOT"
python Codes/tools/render_fig3_snr_controls.py --data-root "$BALDS_DATA_ROOT" --output "$BALDS_FIGURE_ROOT"
python Codes/tools/render_deletion_visual.py --data-root "$BALDS_DATA_ROOT" --output "$BALDS_FIGURE_ROOT"
python Codes/tools/plot_snr_benchmarks.py --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels" --manifest "$BALDS_DATA_ROOT/results/paper/snr/inputs/benchmark_manifest.json" --data-root "$BALDS_DATA_ROOT" --output "$BALDS_FIGURE_ROOT"
python Codes/tools/replay_retrieval.py --data-root "$BALDS_DATA_ROOT" --output "$BALDS_DATA_ROOT/results/paper/retrieval" --paper Paper
```

These cover Tables8/9, Tables1/14/15/16, Tables12/13, Fig3, Figs9/10 and numerical replay of Tables2/18/19. The benchmark input manifest and global deletion selection are produced by `reproduce_benchmarks.py`; see `Reports/experiments/E_BENCH.md` before running the dependent commands. Table17 is the fixed injection mapping documented in `E_SOURCE.md`.

## Reproduction differences recorded on September26

The final text explicitly specifies stable head ties, but the filed Fig2/5 means used NumPy `argpartition`. Replaying that old rule on the same arrays exactly reproduces the historical means. The public calculators follow the final written rule; fresh Fig2/5 points therefore change slightly (maximum across the88 head/readout cells0.07717 LDS×100; FMAS maximum0.04372). The strict fresh head-calculator comparison remains enabled. Do not mix regenerated matched-head values with the old head curve. Unchanged manuscript PDF/JSON snapshots preserve the originally submitted points.

Table10 curvature rows reproduce all displayed values and intervals. CPU reconstruction of its float32 projected features changes some printed cells by about0.01–0.02 LDS×100; the new per-query JSON is retained rather than replaced by archived means. Full receipts and exact head membership counts are in `Reports/final_alignment_2026-09-26/FIGURE_REPRODUCTION.md` and `figure_validation.json`.

Retired scripts and the previous README are archived under `Reports/final_alignment_2026-09-26/before/Codes/Figure`. There are no current producers for Table3, Tables6/7/11, Figs7/8,13-method/high-power extensions or old soft-BA diagnostics.
