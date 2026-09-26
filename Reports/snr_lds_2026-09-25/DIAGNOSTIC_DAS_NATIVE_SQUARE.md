# DAS native-square fitting diagnostic — 2026-09-25

Status: completed diagnostic, not a replacement of accepted A4 / paper results.

## Scope and execution

Researcher asked what happens when DAS uses its native squared attribution
scores for both fitting and aggregation. Only the fitting input changes from
signed `t` to native `t**2`; aggregation is `t**2` in both arms. The accepted
R4 zero-mean Gaussian amplitude-SNR kernel and thresholds 1/2/3/4 are unchanged.
This is NOT the newly discussed signal/noise density-ratio rule.

Reuses A3 benchmark manifest, exact query/train IDs, masks, responses and scores.
No training, feature extraction, GPU work, production-code changes or overwrite
of accepted results. Current long-running GPU feature jobs are untouched.

Executed from `/path/to/CFA`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /path/to/CFA-envs/e3c-torch260/bin/python /path/to/BA-LDS/Reports/snr_lds_2026-09-25/diagnose_das_native_square.py --panels all --output /path/to/CFA/_Data/results/snr_lds_20260925/diagnostics/das_native_square_all
```

Exit 0. Summed panel runtime 66.01 s, two CPU threads. Initial C2 seed42-only
diagnostic is separately retained under `diagnostics/das_native_square_s42`.

## Results at amplitude threshold 3

All 16 panels / 1,600 queries valid in both arms. Every Full LDS value was
checked against its accepted per-query record (absolute difference < 1e-10).
No fit-failure-based change to the comparison population. C2/C10 FM average
seeds 42/123/456 equally; each seed has 100 queries per track. DDPM and AB2 use
seed42, 100 queries per track. LDS below is percent; retention is the pooled
per-query median, also percent. Original signed fit / new native square fit:

| Platform | Track | Old LDS | New LDS | Old retention | New retention |
|---|---|---:|---:|---:|---:|
| C2 FM | gen | 43.7332 | 46.4359 | 4.670 | 25.900 |
| C2 FM | val | 50.4497 | 52.5262 | 4.710 | 25.860 |
| C10 FM | gen | 40.5059 | 39.9845 | 5.157 | 29.348 |
| C10 FM | val | 44.0747 | 43.5411 | 5.290 | 29.343 |
| C2 DDPM | gen | 31.0248 | 31.4509 | 3.000 | 25.170 |
| C2 DDPM | val | 41.1879 | 41.7061 | 3.640 | 25.560 |
| AB2 | gen | 30.9150 | 30.4716 | 6.630 | 27.040 |
| AB2 | val | 38.4828 | 37.9751 | 6.310 | 26.920 |

## Interpretation and boundaries

- Gaussianity of native attribution error is a different assumption from
  Gaussianity before DAS's terminal square. The latter cannot be silently
  imposed to reject the researcher's native-score design.
- Direct native-score fitting admits substantially more samples at numerical
  threshold 3. Different fit spaces do not have matched threshold severity.
- Numeric fitting success does not establish Gaussian measurement errors.
  Fitting the cross-sectional score concentration remains a modeling proxy.
- These are descriptive LDS changes, not paired significance claims, and not
  evidence of greater attribution accuracy.
- This does not yet evaluate the proposed density-ratio SNR formulation.

Machine-readable summary and complete paired per-query records, including all
four thresholds, are in the output directory above. No paper adoption or
scientific protocol change has been made.
