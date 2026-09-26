# Appendix retirement and mechanism reorganization — 2026-09-26

The finalized main text was not edited. Appendix changes follow the current
measurement-noise → subset aggregation → truncation/power weighting → squaring
and method-comparison argument.

## Retired from the compiled manuscript

- Table6 and Fig7: CIFAR-2 signed-support head controls.
- Table7 and Fig8: cross-dataset native signed-support range diagnostics.
- Table11: DDPM 1,000-query extension, including dedicated setup and results.
- Old Fig6 PDFs: 13-construction power sweeps and extra EK-FAC readouts.
- High-power p=6,8 discussion and DAS 8-of-12 win counting.

Retired figure/table directories are under `retired/Figures/`. Complete prior
mechanism text, theory introduction, and notation are under `before/Sections/`;
the previous Fig6 wrapper and plots are under `before/Fig6/`. Files were archived,
not destroyed. Original experimental data were not changed.

## Retained evidence and distinctions

- Score densities, R16 VarRatio diagnostics, pilot-defined bands, band-only LDS.
- Absolute-score truncation and matched linear/squared main-figure controls.
- Four core methods' signed and folded powers, p=0..4, both query tracks.
- Tables8–10 cross-setting squaring evidence, including results limiting the claim.
  Cross-dataset signed-support controls are explicitly distinguished from the
  main absolute-score heads; these are not the retired range sweeps.
- FMAS R16 fluctuations are conditional on its fitted curvature.

`render_core_power.py` regenerates Fig6 from the existing JSON results without
recomputing experiments. Values are equal averages over seeds 42, 123, and 456.
The stale power-equation reference in the notation index now points to Section 4.2.
