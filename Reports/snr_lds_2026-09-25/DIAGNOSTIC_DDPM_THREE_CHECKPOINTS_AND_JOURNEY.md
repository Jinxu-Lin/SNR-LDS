# Three-checkpoint diagnostic and Journey next steps — 2026-09-25

Researcher requested Journey next steps and numerical results without step8000.
This report does not change the active four-checkpoint run or paper values.

## TracInCP / GAS: step2000,4000,6000 only

Used verified tensors from the retrained trajectory, same original gen/val
queries and accepted response/mask ID mapping. Existing TracInMethod at19c3114
computes uniform checkpoint mean; GAS normalizes both vectors within each
checkpoint before averaging. Accepted SNR R4 amplitude threshold3; no fitting
or threshold changes, no DAS diagnostic adoption.

Executed CPU-only, two threads:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /path/to/CFA-envs/e3c-torch260/bin/python /path/to/BA-LDS/Reports/snr_lds_2026-09-25/diagnose_ddpm_three_checkpoints.py
```

Exit0. Four finite `(5000,100)` score matrices plus400 per-query records are
stored separately at
`/path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/diagnostics/three_checkpoints/`.
Checkpoint metadata and exact query IDs checked; validation keeps0..94,96..100.

All LDS values below are multiplied by100:

| Method | Track | Full(all100) | Full(SNR-valid set) | SNR | Valid / total | Empty among valid |
|---|---|---:|---:|---:|---:|---:|
| TracInCP | val | 5.645513 | 10.003374 | 5.569593 | 19/100 | 12 |
| TracInCP | gen | 6.642491 | 8.621625 | 4.951587 | 27/100 | 17 |
| GAS | val | 5.551465 | 9.030409 | 1.726664 | 29/100 | 28 |
| GAS | gen | 5.662729 | 7.779304 | 0.779570 | 31/100 | 29 |

Fit failures are NA, not zeros; valid empty predictions follow the accepted
zero-LDS rule. Low coverage limits interpretation. Full(all100) and SNR have
different populations; the paired Full column is supplied explicitly.
These values describe a three-checkpoint variant, not the pending four-checkpoint
baseline. No paper entry, production score, or active step8000 worker changed.

## Journey: available artifacts and actual dependency

Existing source root:
`/path/to/CFA/_Data/results/ddpm_gaps_20260923/journey_j3/`.
Recipe/result indicate original archive seed42 model, fresh100 generation
queries, DDIM50 eta0, noise seed20260923, ten captured trajectory positions,
T100 training features, cuda_jl4096. Existing train `(5000,4096)` and query
`(100,4096)` tensors were accepted as features on09-24. Result explicitly says
scores=false, gt=false. New query indices0..99 are not identities of the old
generated images with the same numeric indices.

The remaining score solve / LDS / SNR are CPU work. New-query GT requires GPU
forward losses on matching subset models, not another attribution-feature run.

Read-only inventory of the main local archive at
`_Data/raw/das_archive/CIFAR2/saved/5000-0.5/lds-val` found192 UNet weight files:
subsets64..127, replicas0,1,2. No subset0..63 UNet weight files were found in
that archive directory. The accepted DDPM panel requests the first64 masks,
and import code preserves subset-index order0..127. Historical loss files do
not replace model weights for evaluating new queries. Other hosts/archives
have not been exhaustively searched in this turn.

Two distinct routes, neither newly launched:

1. Keep current main-table subset protocol: recover subset0..63 model weights
   from other archives/hosts, then compute losses for the new100 queries using
   the matched existing GT evaluation recipe. Query identity still differs
   from the original main table and must be disclosed; a strictly common-query
   comparison also requires other methods' scores on the new queries (or exact
   recovery of the old generation trajectories).
2. Reuse available64..127 models for an explicitly separate Journey panel:
   compute new-query GT, then CPU score and SNR. No new training is needed if
   the recorded model/mask provenance checks out. Do not silently substitute
   these subsets/queries into the existing main table. A paired multi-method
   comparison would have to use the same new queries and subset bank.

The immediate low-cost step is locate/recover missing weights before deciding
whether to change evaluation scope; do not automatically schedule subset
retraining. Journey validation remains n/a, not a missing-result task.
