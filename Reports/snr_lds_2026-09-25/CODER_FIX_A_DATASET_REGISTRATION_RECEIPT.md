# Coder receipt — A dataset registration fix

Date: 2026-09-25 AEST  
Task: `CODER_FIX_A_DATASET_REGISTRATION_V1`  
Status: produced and verified; formal A/B experiment not launched.

## Minimal change

`tools/prepare_benchmarks.py::prepare` now conditionally imports the existing
`balds.workflows.das_import` module before resolving `cifar2_das`. This executes
the existing `DATASETS.add(DATASET, _load_cifar2_das)` registration. No loader,
split logic, query rule, score method, SNR fit, threshold or scientific
parameter was copied or changed, and the full application container was not
imported.

The regression test starts a separate Python process, proves `cifar2_das` is
initially absent, patches only the archive/pixel IO boundary, invokes the real
`prepare_benchmarks.py::main`, confirms the registered function is the existing
`_load_cifar2_das`, constructs all 16 panels, and preserves DDPM val IDs
`0..94,96..100` with a 1000-column response identity.

## Fixed delivery

- read-only snapshot: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3`
- archive: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3.tar.gz`
- archive SHA-256: `06aea6241d18ffa74c723b89f7be064f972f7c4cc893c1b24606a9753f10553b`
- per-file digest aggregation: `ffa52021965cdc691e82f40b6f5d37213d4e604fa18400067ce8659bbeba512e`
- attempt: `j3-snr-a3`
- output root: `/path/to/CFA/_Data/results/snr_lds_20260925/a3/`

The r2 snapshot and `/a1/` logs/recovery evidence were not modified. The r3
launcher, its `derived-input-root`, README examples and
`configs/snr_lds_j3.json` all point to `/a3/`; the launcher atomically records
`launcher_a/attempt_id` as `j3-snr-a3`.

## Ordinary verification

```text
fresh-process targeted test:
  2 passed in 4.20s

full test suite:
  327 passed, 1 skipped, 2 warnings in 71.54s

layer check:
  Dependency layers passed (92 modules).

launcher syntax:
  bash -n tools/run_snr_lds_a.sh — passed
```

Read-only real-source prepare was also run with both Hugging Face offline flags,
writing only this temporary manifest:

`/tmp/snr-a3-prepare-preflight.N6FnjC/benchmark_manifest.json`

Receipt:

```json
{
  "panels": 16,
  "ddpm_val_ids": "0..94,96..100",
  "ddpm_response": "results/gt_matrix_ddpm_cifar2_das_val_seed_42.npy",
  "restored_arrays": []
}
```

No AB2 restoration, SNR batch, fitting, scoring, model loading, download or
formal output was performed. `/a3/` did not exist at handoff time.

## A3 command package

These are the complete corrected entry points for Experimenter review and later
Executor authorization:

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3

# First a3 launch only after Experimenter publishes the recovery basis.
./tools/run_snr_lds_a.sh launch

./tools/run_snr_lds_a.sh status

# Resume only after confirming the recorded a3 shell and child are no longer alive.
./tools/run_snr_lds_a.sh resume

./tools/run_snr_lds_a.sh verify
```

The Coder did not execute any of these four formal commands. The unexplained
empty-log r2 launch remains unexplained; this receipt attributes only the later
complete traceback to the confirmed missing registration import.
