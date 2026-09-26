# DDPM recovery reassigned to J3 step4000 + XC0 step8000

**Superseded at 2026-09-25 23:39 AEST:** researcher reassigned unfinished
step8000 to jinxu3. XC0 worker and J3 incoming-wait controller were explicitly
terminated; do not relaunch either. Current execution is documented in
`RUN_DDPM_STEP8000_J3_a2.md`, worker3679683 on J3 cuda0. Earlier evidence below
is retained as history.

Authority: researcher directly requested SSH execution on xuchang0 physical cuda1, with one checkpoint on each machine. This supersedes the earlier J3 two-checkpoint serial recovery assignment; original xuchang3 a2 remains withdrawn.

## Ownership and unchanged configuration

- jinxu3 cuda0 keeps its live step4000 computation, PID3516458. It is not restarted or patched in memory.
- xuchang0 cuda1 receives step8000 only; step2000/6000 remain reused on the J3 master store.
- Scientific code: `19c31140c300d344f827f15e04d5ff460ad9f6ac`; supervisor only: `f686bd856ae9e9a99178ba7f34414941aa49db3b`; SNR: accepted r4 snapshot. T100/grid, MSE, normalize=True, p4096, projection seed0, feature seed42, batch1 unchanged.
- New `run_ddpm_dual_finish.py` is a task adapter only. Ordinary mocked routing/return tests confirm remote features((8000,)) only, GPU1/CPU-hidden environments, and receipt transfer after feature rsync.

## Resource and environment preparation

XC0 cuda1 observed idle1MiB/0%, 5.3TiB disk free. Python3.12, torch2.6.0+cu124, CUDA12.4 compiler, driver570.190.
Transported only inputs.pt, step_8000.pt, source bench_manifest.json, task dependencies, and the task script. No retraining or global environment change.
XC3's existing fast_jl binary requires GLIBC_2.34 unavailable on XC0, so rebuilt identical fast-jl0.1.3 in the task-specific python_deps using XC0 CUDA12.4; no projection-backend substitution.

```bash
ssh xuchang0 'git -C /path/to/CFA fetch origin codex/ddpm-tracin-gas-j3-20260924; git -C /path/to/CFA worktree add --detach /tmp/cfa-ddpm-step8000-xc0 19c31140c300d344f827f15e04d5ff460ad9f6ac'
ssh xuchang0 'git -C /path/to/CFA worktree add --detach /path/to/CFA-worktrees/ba-native-gpu-wave-20260921 f686bd856ae9e9a99178ba7f34414941aa49db3b'
ssh xuchang0 'CUDA_VISIBLE_DEVICES="" CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 /path/to/miniconda3/envs/da/bin/python -m pip install --target /path/to/CFA/_Data/results/ddpm_tracin_gas_20260924/python_deps --upgrade --no-deps --no-build-isolation --no-binary=fast-jl --no-cache-dir fast-jl==0.1.3'
```

## Execution and checkpoint-boundary handoff

```bash
ssh xuchang0 '/path/to/miniconda3/envs/da/bin/python /path/to/CFA/_Data/reports/ddpm_tracin_gas_2026-09-24/run_ddpm_dual_finish.py launch-xc0'
/path/to/CFA-envs/e3c-torch260/bin/python /path/to/BA-LDS/Reports/snr_lds_2026-09-25/run_ddpm_dual_finish.py launch-join
```

J3 join started at16:33 AEST, supervisor3522777/worker3522779, job `results/ddpm_tracin_snr_20260925/dual_join_a1/supervised/launcher/repeat_-1_eb8c4c0c9b52.json`.
It waits for the existing log's exact `DONE_FEATURE step=4000 split=val`, then SIGTERMs only original feature PID3516458 after checking command and process start token302259470. This intentionally ends the old serial chain before it can compute a complete step8000. Its nonzero exit is a planned handoff, not a new experimental failure; completed step4000 tensors remain.
The original controller must exit before new CPU work starts. No changes to the running worktree/script or already-loaded method are made.

XC0 computes three step8000 matrices and meta, rsyncs to J3 `results/ddpm_tracin_gas_20260924/incoming/xc0_step8000/`, then returns `RETURNED.json`. J3 independently reads shape/finite/query IDs, registers master data, and runs existing score→SNR summary→verify with CUDA hidden. The final output remains `results/ddpm_tracin_snr_20260925/a1/`.

XC0 launch confirmed at2026-09-25 16:35:10 AEST: supervisor3540767, worker3540777, physicalcuda1, torch2.6.0+cu124; log reports START_FEATURE step=8000 split=train and actual GPU allocation1246MiB. Job `results/ddpm_tracin_snr_20260925/xc0_step8000_a1/supervised/launcher/repeat_1_d42539d7aaad.json`. Formal first50-row timing is pending, not a smoke run. J3 step4000 reached200/5000 in831.85s, approximately4.16s/row, consistent with roughly6h for one complete checkpoint at that speed.

The task-local fast-jl0.1.3 build and CPU import check succeeded; no global environment was modified. Both workers are active; no score/SNR completion is claimed yet. J3 original serial controller may later report nonzero exit after the intentional boundary handoff; check the new dual_join job for the actual task outcome.

## 2026-09-25 23:16 AEST observed update / partial acceptance

Researcher requested completion verification and supplementation of the paper.
Live evidence does not establish full completion:

- J3 step4000 train/gen/val completed at approximately 22:04/22:11/22:18.
  Its original feature child has exited at the intended boundary; the SIGTERM
  traceback is the documented handoff, not loss of step4000.
- Independent CPU readback of all nine tensors for steps2000/4000/6000 passed:
  train `(5000,4096)`, gen/val `(100,4096)`, float32 and finite. Each metadata
  record names the correct step and the same 100 unique query IDs per track.
  Validation IDs remain `0..94,96..100`, not a positional `0..99` substitute.
- XC0 PID3540777 remains active, step8000 train progress1450/5000 at elapsed
  23834.09s. Its launcher has exit_code=null, not success. GPU1 has this worker
  as its only compute process at observation; low instantaneous GPU utilization
  alone does not indicate completion. No cause for its slower throughput is
  asserted by this check.
- XC0 train throughput is about16s/row versus J3 about4.16s/row. At unchanged
  speed, remaining train3550 plus gen/val200 rows imply roughly17h to finish
  features, excluding transfer and CPU closeout. This is an estimate, not a
  new resource authorization or worker change.
- J3 join supervisor3522777/worker3522779 is alive and correctly waiting for
  XC0 RETURNED.json, which is absent. Master step8000 contains metadata only;
  no complete step8000 feature tensor or final score/SNR output exists.

Accepted scope is therefore 9/12 complete feature tensors, not four new score
matrices or 400 SNR records. Table1 DDPM TracInCP/GAS gaps remain unfilled;
no paper numbers were changed. No live job was interrupted or reallocated.

## 2026-09-25 23:27–23:31 AEST throughput diagnosis (read-only)

Researcher asked why feature extraction is slow. No worker or configuration was
changed, and no additional GPU computation was launched.

Confirmed computational scope: existing GradFeaturizer, T100 fixed grid,
batch_size=1, MSE gradient, per-timestep normalization, average then cuda_jl
projection. One checkpoint covers 5000 train + 100 gen + 100 val samples,
therefore 520,000 sample/timestep gradient evaluations. Four checkpoints cover
2,080,000. This is not an ordinary image embedding forward pass.

Reference: original same-recipe DDPM feature log
`results/ddpm_noncore_20260923/shared_gpu/supervised/launcher/repeat_0_89a2d88cdc54.log`
records train24194.53s + gen519.79s + val518.63s = 7.009h per checkpoint while
sharing GPU with Journey. Current J3 step4000 took approximately6h. XC0 latest
train1500/5000 at24627.73s, latest50 rows793.64s (~15.87s/row), nearly4x J3.

Live XC0 observations:

- GPU1 only contains the authorized PID3540777. Five consecutive dmon samples
  show SM9–10%, power77–137W, temperature41–42C, clocks2520–2715MHz; memory
  roughly1.25GiB. No evidence of GPU capacity saturation or thermal throttling.
- Worker lifetime CPU99.7%, main/autograd threads roughly41%/58%; Xeon Gold6133
  @2.50GHz, versus J3 i9-13900. Whole host CPU94–95% idle during vmstat samples,
  RAM351GiB available, worker VmSwap=0, current swap-in/out zero. Full system
  swap is historical occupancy, not evidence this worker is swapping.
- CPU cgroup quota=-1 and nr_throttled=0; affinity0–79. No observed quota or
  CPU-core restriction. GPU1 PCIe Gen3 x16, not an observed x1 link fault.
- Both environments report torch2.6.0+cu124 and diffusers0.37.0. Both logs emit
  the same efficient-attention-backward vmap batching fallback warning. It is
  a shared inefficiency, not by itself an explanation for the host difference.
- Code executes 100 Python/functional-autograd iterations per sample, plus
  per-parameter gradient copying and a per-sample synchronization. Together
  with low GPU activity and CPU usage, this points to host-side dispatch /
  autograd overhead, amplified on XC0's older CPU. Exact time attribution to
  individual functions is not established without profiling; no assertion
  that a specific operator or fast_jl rebuild alone caused the slowdown.

The first XC0 formal50 rows already took965.92s (~19.32s/row). That deviation
should have triggered earlier scheduling reassessment; the earlier J3 speed
estimate was not transferable to XC0. No mid-matrix row checkpoint exists,
so migration would recompute the unfinished step8000 train matrix.

## Remaining scope

This completes the four pending TracInCP/GAS method-track inputs, not the whole paper's remaining plan. Journey gen still has new-query features without matching GT. SNR B1/B2 (fixed-selection head/square controls and existing R16 calibration) remain unexecuted and need the already-filed runtime fixes. The current plan also separately lists new EK-FAC IF/DAS repeats and MC-budget intervention as pending GPU work, outside this dispatch. Retired weighted D-TRAK/AbU+/NDA and Journey val are not reopened.
