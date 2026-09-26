# Coder receipt — CODE_SNR_LDS_20260925 r1

Date: 2026-09-25 AEST  
Status: produced and code-verified; formal experiment not launched.

## Source identity

`/path/to/BA-LDS` had no Git metadata at assignment time, as the task book states. Therefore this receipt does not claim a commit. The delivered immutable identity is:

- read-only tree: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2`
- archive: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2.tar.gz`
- archive SHA-256: `2034f7e3c35bbb08eedd0682a75471cb0d53096ef2a8480aa92d56138376fcfb`
- sorted per-file digest aggregation: `8e09f31ef405e164dc3848943926d21012493fc15008b65a5014e6857353b203`

Earlier local snapshots without suffix `_r2` predate the final non-overwriting B analysis output and are superseded; they are not execution inputs.

## Delivered behavior

- Exact float64 `snr_zero_mean_gaussian_v1`: stable RMS, linear quartiles, zero-centered robust grid, weighted rank-2 `[1,x²]` fit, strict two-sided `abs(x)/sigma_x > zeta`, ζ=1/2/3/4 from one fit, and no N-point KDE.
- Full/SNR paired evaluation and distinct all-zero, valid-empty, constant, fit-failed, missing-input and not-applicable states.
- Signed DAS `t` fit with `t²` aggregation exactly once; ordinary methods retain signed native values.
- Explicit non-contiguous query-ID handling and completed-manifest overlays; AB2 signed lambda=1 recovery only into caller-owned current inputs.
- Single-file atomic query checkpoints, same-identity resume, method completion records, status and source-algebra verification.
- Equal-seed/query-bootstrap summaries and separate `snr_lds` outputs; archived `ba_lds` remains archived and is never renamed.
- Three-seed benchmark-level four-candidate E-DEL selection, exact-tie utility averaging, complete 50-query k300/k1000 evaluation, and unavailable output on an empty common set.
- Fixed q0/q1 plots plus sigma/coverage distributions.
- Appendix fixed raw-SNR mask controls and independent-pilot R16 diagnostics, including complete subset-prediction variance and fixed two-way repeat0–7/repeat8–15 evaluation.
- CPU-only A/B launch, status, resume and verify wrappers. Neither wrapper was launched by Coder.

## Modified or added code

- `src/balds/evaluation/background.py`
- `src/balds/workflows/{benchmark,controls,repeatability}.py`
- `src/balds/cli/ba.py`
- `src/balds/report/{__init__,tables}.py`
- `tools/{prepare_benchmarks,summarize_benchmarks,aggregation_controls,plot_snr_benchmarks,verify_snr_appendix}.py`
- `tools/run_snr_lds_{a,b}.sh`
- `configs/snr_lds_j3.json`
- `tests/test_snr_background.py`
- `README.md`, `ARCHITECTURE.md`

## Verification

Environment: `/path/to/CFA-envs/e3c-torch260/bin/python`, `PYTHONPATH=src`.

```text
targeted affected tests: 26 passed in 1.20s
full tests: 326 passed, 1 skipped, 2 warnings in 68.60s
layer check: passed (92 modules)
launcher shell syntax: passed
```

The skip is an existing optional-environment condition. The warnings are existing PyTorch scheduler deprecation notices. No model execution, scoring, formal smoke or formal SNR-LDS evaluation was run.
