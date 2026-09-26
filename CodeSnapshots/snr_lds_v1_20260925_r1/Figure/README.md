# Paper figures and diagnostic tables

Run from the checkout after installing `Codes/` with the `figures` extra and
preparing the selected artifacts. `BALDS_DATA_ROOT` chooses the data root;
otherwise scripts use the checkout's `_Data/`. The numerical LDS functions are
shared with `balds.evaluation.lds`.

The retained scripts cover the paper's magnitude, power, head, square, band,
cross-platform and DDPM diagnostics. See the experiment reports in
`Reports/experiments/` for the required order and exact settings. There is no
all-experiments launcher: retired experiments are excluded.

Most scripts write tables, figures and numerical JSON below
`_Data/results/paper_figures/{tables,figures,legacy}/`. The relative link
`out/json` preserves access to migrated historical JSON for the unchanged
paper scripts. It is intentionally unresolved until those artifacts arrive.

`compute_square_heads.py` and `render_figure1.py` are portable counterparts of
the preserved scripts in `Paper/`. They write the requested figure/table assets
to `Paper/ICLR/Figures/section31/`. They are not run during repository migration.
`render_figure1.py` also reads the accepted R16 analysis; set `BALDS_R16_ROOT`
to a freshly reproduced analysis directory to render those results.

Head ties use the original NumPy `argpartition` convention. The scripts retain
the paper's scoring/query/subset conventions; they do not select new parameters
for each transformed score. A successful renderer does not certify a pending
scientific result or resolve the historical DAS V3 protocol mismatch.
