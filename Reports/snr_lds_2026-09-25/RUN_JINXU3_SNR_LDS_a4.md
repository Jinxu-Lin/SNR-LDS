# jinxu3 Executor receipt — SNR-LDS CPU A V4 closeout

## Identity and reuse

- `task_id`: `SNR_LDS_CPU_JINXU3_20260925`
- taskbook revision: `4`
- attempt: `j3-snr-a4`
- host: `anonymous-lab3` (`jinxu3`)
- fixed snapshot: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4`
- Python: `/path/to/CFA-envs/e3c-torch260/bin/python`
- read-only source: `/path/to/CFA/_Data/results/snr_lds_20260925/a3/`
- closeout output: `/path/to/CFA/_Data/results/snr_lds_20260925/a4/`
- B line: **not started**, as required.

V4 reused the already completed a3 manifest and 243 panels (24,300 query
records).  It did not run prepare, batch scoring, fitting, feature extraction,
model work, or GPU work.  The five a3 missing-input cells were preserved as
missing; no values were fabricated.

## Command and process evidence

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh launch
```

The r4 launcher uses `nohup setsid` internally as required by the taskbook.

- launcher PID: `3302052` (detached, parent PID 1)
- observed source-verification Python PID: `3302110`
- launcher interval: approximately 04:26:13–04:28:04 AEST
- final `a4/launcher_a/exit_code`: `0`
- `CUDA_VISIBLE_DEVICES` was empty; no GPU process was started.

## Delivered artifacts

The closeout produced all required outputs:

- `tables/summary.json` and `tables/summary.csv`
- `edel/edel_selection.json`
- `figures/fixed_c2_s42_gen_q0_q1.png`
- `figures/sigma_coverage_distributions.png`
- `verify_a.json`
- `launcher_a/{attempt_id,pid,run.log,exit_code}`

The summary uses the repaired SNR schema: all-failed methods remain `NA`
and are not populated from `ba_lds`.  It reports 119 aggregate records and
476 zeta-sensitivity rows; the existing fit-failure/empty counts and the five
missing score inputs remain explicitly recorded.  E-DEL includes the filed
`k=300` and `k=1000` comparisons.

## Final verification

The required `SNR_STAGE=closeout ./tools/run_snr_lds_a.sh verify` returned
`0`.  `verify_a.json` reports:

```json
{
  "rule": "snr_zero_mean_gaussian_v1",
  "verdict": "pass",
  "verified_queries": 24300,
  "missing": 5,
  "errors": []
}
```

The five missing entries are exactly C2 DAS seed42 gen
`tracincp_T100`, `gas_T100`, `journey_trak_T100` and val
`tracincp_T100`, `gas_T100`.  This is a successful V4 closeout of the
available a3 data, with those documented source gaps; it is not a claim that
248/248 source cells exist.  a3, a1, and all prior recovery evidence remain
unchanged.
