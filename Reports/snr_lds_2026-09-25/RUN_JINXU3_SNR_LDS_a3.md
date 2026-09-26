# jinxu3 Executor receipt — SNR-LDS CPU A (V3, partial/blocked)

## Identity

- `task_id`: `SNR_LDS_CPU_JINXU3_20260925`
- taskbook revision: `3`
- attempt: `j3-snr-a3`
- host: `anonymous-lab3` (`jinxu3`)
- fixed snapshot: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3`
- Python: `/path/to/CFA-envs/e3c-torch260/bin/python`
- output root: `/path/to/CFA/_Data/results/snr_lds_20260925/a3/`
- B line: **not started**, as required by V3.

The snapshot, manifest inputs, scientific rule, query IDs, and parameters were
not edited.  The previous V2/a1 receipt and evidence remain untouched.

## Launch and process evidence

The authorized launch command was executed from the fixed snapshot:

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3
./tools/run_snr_lds_a.sh launch
```

That first detached shell exited with an empty log before producing an exit
code.  Its `pid`, `attempt_id`, and empty `run.log` were preserved in
`/path/to/CFA/_Data/results/snr_lds_20260925/a3/recovery.6LJK2e/`.
Following the single V3 bounded-recovery allowance, I resumed the same a3
identity with `setsid ./tools/run_snr_lds_a.sh resume` so the host shell would
not reap the detached process.  This was an operational process-group
isolation only; no code, input, parameter, or scientific rule changed.

- resumed launcher PID: `3272829` (parented to PID 1)
- batch worker PID: `3272914`
- launcher evidence: `a3/launcher_a/{attempt_id,pid,run.log,exit_code}`
- worker duration: approximately 03:19:11–03:38:36 AEST
- `CUDA_VISIBLE_DEVICES` was empty; `nvidia-smi` showed no compute process.
- final launcher `exit_code`: `1`

## Batch result before the stop

The prepare and CPU SNR batch completed its available cells:

- manifest: 16 panels, `inputs/benchmark_manifest.json` (100,076,468 bytes)
- planned cells: 248; complete: 243; missing input: 5; pending: 0
- missing input cells: C2 DAS seed42 gen `tracincp_T100`, `gas_T100`,
  `journey_trak_T100`; C2 DAS seed42 val `tracincp_T100`, `gas_T100`
- completed summary rows: 243 (`24,300` query evaluations)
- completed-row totals: `n_valid=16,946`, `n_failed=n_fit_failed=7,354`,
  `n_empty=3,876`, `n_all_zero=0`, `n_missing_query=0`
- manifest N/A: C2 DAS seed42 val `journey_trak_T100` (outside the filed
  validation definition)
- DDPM/C2 DAS validation IDs are exactly `0..94,96..100`.

The panel/query evidence is retained under `a3/panels/` (49,589 files,
approximately 608 MB).  The summarizer wrote a partial `tables/summary.json`
and a two-row `tables/summary.csv`; E-DEL and figures were not reached.

## Blocking evidence

The immutable snapshot's summarizer then failed deterministically while
writing the mixed-input CSV:

```text
ValueError: dict contains fields not in fieldnames:
'snr_lds_std', 'snr_lds', 'snr_lds_query_ci95'
```

This is a run-code/schema defect, so I did not patch the snapshot, generate a
replacement table, or spend the one permitted recovery a second time.

The read-only V3 verification command was also run after the worker stopped.
It failed its strict numerical anchor before the artifact-existence checks:

```text
ValueError: SNR verification failed:
cifar2_5k_s42_gen/das_native_sq/q0
Max absolute difference: 4.6104906e-10 (atol=1e-12)
```

No `edel/edel_selection.json`, `figures/`, or final `verify_a.json` was
claimed or manufactured.  The run is therefore **blocked/partial**, not an
accepted A delivery.  Experimenter/Coder action is required for the
summarizer schema and verification tolerance/anchor decision; a new authorized
attempt or explicit recovery instruction must preserve this a3 tree.
