# jinxu3 Executor receipt — SNR-LDS CPU A (blocked)

## Identity

- `task_id`: `SNR_LDS_CPU_JINXU3_20260925`
- taskbook revision: `2`
- attempt: `j3-snr-a1`
- host: `anonymous-lab3` / jinxu3
- fixed snapshot: `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2`
- snapshot archive SHA-256: `2034f7e3c35bbb08eedd0682a75471cb0d53096ef2a8480aa92d56138376fcfb`
- Python: `/path/to/CFA-envs/e3c-torch260/bin/python`
- output root: `/path/to/CFA/_Data/results/snr_lds_20260925/a1/`

V2 explicitly opened the A launch gate.  The immutable snapshot and its
scientific parameters were not edited.

## Commands and observed process

The authorized command was run from the fixed snapshot:

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2
./tools/run_snr_lds_a.sh launch
```

The first detached launch wrote PID `3174581`, but exited before writing a
log or exit code.  Following the V2 recovery procedure, the empty `pid` and
`run.log` were preserved at:

```
/path/to/CFA/_Data/results/snr_lds_20260925/a1/recovery.NhYHFr/
```

One authorized `resume` was then run.  Its launcher PID was `3175129`; the
worker exited during prepare with launcher `exit_code=1`.  Log:

```
/path/to/CFA/_Data/results/snr_lds_20260925/a1/launcher_a/run.log
```

No SNR worker remains active, no GPU process was started, and no B launcher
was started.

## Blocking failure

The formal A chain fails deterministically in `prepare_benchmarks.py` before
writing `inputs/benchmark_manifest.json` or entering the BA batch:

```text
KeyError: "unknown dataset 'cifar2_das'; registered:
['artbench2_256', 'cifar10_inj4', 'cifar10_v2', 'cifar2_5k']"
```

The fixed configuration requests `cifar2_das`, and V2 requires the corrected
DDPM val source.  Substituting `cifar2_5k`, changing the dataset registry,
editing the snapshot, or skipping the DDPM panel would change the authorized
scope, so no such workaround was attempted.  No panels, tables, E-DEL,
figures, or scientific SNR-LDS values were produced.

## Status and next action

`blocked` at A prepare due to a deterministic snapshot/input registry defect.
Coder/Experimenter must provide a corrected immutable snapshot or an explicit
scope-changing decision.  After that, start a new attempt while preserving
this failed attempt and its logs; do not rerun this identity in place.

