# Ownership update — DDPM TracInCP/GAS recovery

**Current update23:39 AEST:** researcher moved unfinished step8000 back to J3
cuda0. Actual worker3679683/supervisor3679665, attemptj3-recovery-a2. XC0
worker3540777 and J3 waitingjoin3522779 were stopped and confirmed exited.
Steps2000/4000/6000 are complete (nine matrices). Only step8000 is recomputed,
then the existing CPU score/SNR chain runs automatically. Do not relaunch XC0,
dual_join, old XC3 lanes or standalone CPU V1. Current receipt:
`RUN_DDPM_STEP8000_J3_a2.md`. The following ownership entries are historical.

Effective 2026-09-25 16:18 AEST under the researcher's explicit direct-execution instruction.

Update16:35 AEST: researcher requested two-host execution. J3 retains step4000 (existingworker3516458), XC0 cuda1 now runs step8000 (worker3540777). J3 dual_join supervisor3522777 owns boundary stop, incoming8000 registration and finalCPU chain. Old XC3 a2 and standaloneCPU launch remain withdrawn. See `RUN_DDPM_DUAL_J3_XC0_20260925.md`; the original all-on-J3 assignment below is superseded for step8000.

- Experimenter now also acts as the Executor on jinxu3. The full recovery chain is running there on physical cuda0.
- Both original xuchang3 a1 workers were confirmed gone. Six completed feature matrices from step2000 and step6000 have been copied and registered on jinxu3. Original files remain intact on xuchang3.
- Remaining step4000/8000 features, four CPU score matrices, and current SNR evaluation will execute serially on jinxu3.
- **Do not launch the old xuchang3 cuda2/cuda3 a2 commands, even if those GPUs become free. Do not separately launch the old jinxu3 CPU V1 command.** This supersedes those recovery/launch permissions for this task while the new owner is active.
- Live supervisor/worker/feature PID: 3516072 / 3516074 / 3516458 on anonymous-lab3.
- Receipt: `/path/to/BA-LDS/Reports/snr_lds_2026-09-25/RUN_DIRECT_DDPM_RECOVERY_J3_a1.md`.
- No change to checkpoint recipe, feature MC budget, projection, query IDs, scoring method, or SNR rule. No other GPU task is affected.
