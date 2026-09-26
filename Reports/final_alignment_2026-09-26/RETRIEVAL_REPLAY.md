# Retrieval replay and paper consistency — 2026-09-26

`Codes/tools/replay_retrieval.py` recomputed every held-out query metric from the migrated matrices for CFM and DDPM, with13 methods on each platform, 400 test queries each, 50,000 candidate training images, validation-only regularization selection and equal concept weights. No inference or feature computation was run. Outputs: `_Data/results/paper/retrieval/`; numeric comparison: `retrieval-verification.json`.

**All 572 numeric cells in Paper Tables2/18/19 agree at their printed precision.** The paper was read only.

A remaining inconsistency is in `Paper/Sections/XY05_EvaluationExperiments.tex` lines195–200, whose CFM fine/coarse prose values do not match the current Table18 or the accepted matrices:

| Method/group | Reproduced global AP | Reproduced Recall@200 |
|---|---:|---:|
| FMAS fine | 0.0494322394 | 0.089900 |
| FMAS coarse | 0.4567854335 | 0.463575 |
| EK-FAC IF fine | 0.0460366060 | 0.088250 |
| EK-FAC IF coarse | 0.4239293689 | 0.433200 |
| D-TRAK fine | 0.0617142571 | 0.096300 |

The prose currently states FMAS AP0.056/0.486 and recall0.100/0.490; IF AP0.050/0.444 and recall0.097/0.453; D-TRAK fine AP0.071 and recall0.106. These are recorded as a paper consistency issue, not used to alter the code or accepted results. The subsequent DDPM fine/coarse AP prose agrees with the replay.
