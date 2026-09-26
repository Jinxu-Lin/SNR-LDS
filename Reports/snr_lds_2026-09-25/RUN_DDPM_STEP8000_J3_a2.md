# DDPM step8000 reassigned from XC0 to J3 — a2

Authority: researcher explicitly requested moving the unfinished step8000 back
to idle jinxu3. This supersedes XC0 step8000 and the J3 incoming-wait join.

## Stopped ownership

At 2026-09-25 23:39:32–33 AEST, checked exact command, process start token and
absence of children, then SIGTERM:

- J3 incoming-wait PID3522779, token302348223;
- XC0 GPU worker PID3540777, token757618384.

Both supervisors persisted exit_code=-15. Both target PIDs disappeared; XC0
GPU1 no longer has a compute process. The unrelated XC0 GPU3 process614074 was
not signalled. Logs and existing on-disk artifacts were retained, no deletion.
Last XC0 progress was1550/5000 at25386.63s. No complete train tensor or per-row
checkpoint was saved, so that incomplete calculation cannot be resumed.

## Unchanged science / reused artifacts

- Core features and scorer: clean detached19c31140c300d344f827f15e04d5ff460ad9f6ac
  at `/tmp/cfa-ddpm-tracin-gas-j3-snr-v1`.
- SNR evaluator: accepted `snr_lds_v1_20260925_r4`; no adoption of the later
  DAS-native-square diagnostic, no threshold change.
- Existing nine verified matrices for step2000/4000/6000 remain untouched.
- Compute step8000 train5000 / gen100 / val100 only, T100/grid, MSE,
  normalization, cuda_jl4096, projection seed0, feature seed42, batch1 unchanged.
- Python `/path/to/CFA-envs/e3c-torch260/bin/python`, torch2.6.0+cu124;
  cuda0, CPU2 threads; existing inputs/checkpoint already local. Disk153GiB free.

## Command and observed launch

```bash
/path/to/CFA-envs/e3c-torch260/bin/python /path/to/BA-LDS/Reports/snr_lds_2026-09-25/run_ddpm_step8000_j3_a2.py launch
```

Adapter is only a caller of existing `features((8000,))`, followed by existing
`run_ddpm_tracin_snr.py run_local` with CUDA hidden. Mocked routing check passed
for exact step8000 scope and CPU-hidden downstream execution. No smoke run.

Started2026-09-25 23:39:46 AEST. Supervisor3679665, actual GPU worker3679683.
Job:
`/path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/recovery_j3_a2/supervised/launcher/repeat_0_2e299916e151.json`
and adjacent `.log`. At23:40, formal START_FEATURE step8000/train is present,
actual worker is allocated1248MiB on cuda0. First50-row formal timing pending.

On successful extraction, automatically run CPU four TracInCP/GAS gen/val
score matrices -> 400 SNR query records -> summary -> final verification.
Final CPU root remains `results/ddpm_tracin_snr_20260925/a1/`, which had no
output when this takeover started. Do not launch old join, XC0 or CPU V1 in
parallel. ETA from same-host step4000 is about6h for features, then CPU closeout.

Current status: running, not completed or accepted for paper adoption.

At23:43 AEST, first formal50 training samples completed in208.30s, or4.166s/row.
This reproduces the prior J3 throughput and is4.64x faster than XC0's first50
(965.92s), approximately3.8x faster than XC0's later stable throughput. Projected
5200-row extraction time is6.02h total, approximately2026-09-26 05:40 AEST,
then CPU score/SNR closeout. This timing is from the production task, not a probe.
