# CFM test400 four-card delivery — 2026-09-23

- Base d3e902436d0e7f4086767680e2dc9c926fe852ad; accepted execution ebbc59f5719af6d612e12576adcc4f674dedbc62.
- Experimenter explicitly covered the bounded wrapper implementation; no scientific GPU jobs launched or Executor contacted.
- Reuse: original EkfacScoreUseCase._stream, original query IDs/RNG, IsolatedScores/atomic blocks,
  inject_subset.report, e3c_launch supervisor; no core numeric change.
- Delta: split new reviewed test400 into four ordered 100-query sets; method sequence FMAS then IF;
  load val100 selection; per-lane block verification and master CPU merge with per-query/concept/fine-coarse outputs.
- CPU validation: targeted 9 tests passed (2.32s); all 437 passed, 1 skipped, 24 pre-existing CPU warnings (99.21s).
  Layer check, bash -n and git diff --check passed. Tests cover partitioning, no test selection,
  original IDs, reuse on restart, FMAS/IF ordering, all physical GPU mappings and CPU merge/group weighting.
- XC3 read-only checks: four 4090s idle, disk115GiB, memory available241GiB; no earlier val worker.
  Existing val verifier independently rerun for both methods: 50 finite blocks each, current v2 source;
  FMAS rho1.0, IF lambda1e-9. Both launcher exit_code0; wall5.0329213/5.6565709hours.
  These val artifacts still require master transfer, which is included in finish; not claimed already on J3.
- Compute budget: four lanes × (5.0329 + 5.6566) ≈42.76 4090h; ideal wall10.69h,
  plan11–12h with collection/merge. Prior13.4h included validation workload and is not test-only cost.
- No retired-v1 result reused; validation roots read only. DDPM, current11-method outputs and live jobs unaffected.
- Authority: researcher forwards TASK_XUCHANG3_TEST400_4GPU_V1.md. Ready, not running.
