# Figure migration into BA-LDS/Paper — 2026-09-25

Migrated the accepted figure layout from `/path/to/CFA/Paper/ICLR/Figures/section31/` into this manuscript, preserving its current SNR-LDS revision.

- Figure 1, page 2: half-width Introduction figure (density and R=16 mean-ranked repeatability deciles).
- Figure 2, page 6: full-width row (signed powers, native heads, matched-head squaring, magnitude heads).
- Added wrapfig; updated figure references in Introduction, Method, and the relevant appendices.
- Updated the main-plot binning description; independent-pilot band-only LDS results remain explicitly distinguished in the appendix.
- Existing manuscript changes and SNR-LDS scientific content were preserved.

Four figure assets: `motivation.pdf`, `motivation.tex`, `aggregation-row.pdf`, `overview.tex`, all under `Paper/Figures/section31/`.

The full target manuscript compiled successfully with `latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=/tmp/ba-lds-figure-split-build iclr2027_conference.tex`. No undefined references, compilation errors, or overfull boxes. Figure pages visually inspected. The compiled PDF is updated at `Paper/iclr2027_conference.pdf`; build intermediates remain outside Paper.

Before-images of changed target files are in `before/`; the text changes are recorded in `changes.patch`. Source-data verification and plotting script: `/path/to/CFA/_Data/reports/author_fig1_split_2026-09-25/`.

## Axis update requested by author

Figure 2 panels b/c/d now use y limits 30–50 with ticks 30, 40, 50; panel a remains 8–55. Some b/c values exceed 50 and are outside the plotting window, as explicitly noted in the caption. Source values are unchanged. Re-rendered using `render_aggregation.py`, visually checked, and recompiled successfully; the Paper PDF is updated.

## Natural-range adjustment (supersedes the 30–50 window)

At the author's request, b/c/d now share 35–55 with 5-point ticks. Their actual ranges are b: 38.47–50.60, c: 38.41–53.96, d: 38.40–47.89. Every curve is visible with margin. Removed the obsolete clipping note from the caption. Figure visually inspected, manuscript recompiled, and Paper PDF updated.

## Separate compact Figure 2 panels

Figure 2 is now assembled from four separately saved PDFs: `aggregation-reweighting.pdf`, `aggregation-native.pdf`, `aggregation-matched-square.pdf`, and `aggregation-magnitude.pdf`. All four use one common tight page size (116.314 × 102.96 pt), so equal-width inclusion preserves aligned axes and equal panel heights. The surrounding subplot gaps and unused lower canvas from `aggregation-row.pdf` are excluded. `overview.tex` follows the adjacent-image composition used by `motivation.tex`. The combined manuscript rendering was visually inspected and compiled without undefined references, errors, or overfull boxes.

The first tight crop placed panel titles too close to the PDF boundary. The final crop adds 5.04 pt above each title, 0.72 pt on each side, and only 0.36 pt below the tight content. All panels retain an identical 117.754 × 108.36 pt page size. The titles are fully visible while the lower and inter-panel whitespace remains compact.

## Figure 1(b) direction and rank labeling

Figure 1(b) now displays magnitude bins from low influence on the left to high influence on the right, matching the direction of Figure 1(a). Its title is `FMAS repeatability ↑`, and the x-axis reads `Magnitude rank (low → high)`. The caption states that higher repeatability is better. The values are the same verified R=16 decile statistics; only their display order changed. The Method and appendix descriptions were updated to match the displayed direction. The manuscript compiled and the Figure 1 page was visually inspected. The current manuscript has pre-existing unresolved references to `eq:loo-target`, `eq:lds-prediction`, and `eq:lds-response`; this figure change does not introduce them.
