# Final implementation review — 2026-09-26

Scope: final flat `Paper/`, SNR background evaluation, benchmark preparation/run/resume/summary, global deletion choice, matched DAS controls, final SNR/deletion/Figure 3 renderers, and the public Codes README. Historical CFA inputs and the migrated data were read-only during this review.

## Confirmed scientific behavior

- The final SNR implementation fits the prescribed zero-mean quadratic to log KDE density on 101 central grid points. It uses sample SD, linear quartiles, MAD about the median, RMS normalization, density weights, and strict `abs(x)/sigma_x > zeta` selection. Failed fits, valid empty selections, and all-zero inputs remain distinct.
- DAS uses signed pre-square `t` for fitting/selection and `t²` for native aggregation exactly once. Full/SNR method means use identical valid queries. Global deletion choice uses all four candidates' common valid queries within each of seeds 42/123/456, then equal seed weights; it never selects from observed utilities.
- Retained Tables 8/9 use the native DAS configuration (including ArtBench lambda 1) with matching linear, square, signed-square readouts and paired query intervals. Figure 3 holds accepted SNR masks fixed across readouts/heads and verifies its recomputed native LDS against accepted per-query rows.
- Missing DDPM input cells remain missing in computation and coverage. Four author-supplied TracInCP/GAS table means are isolated in `Codes/configs/paper_aggregate_supplement.json`; the renderer does not fabricate query records or sensitivity results.
- Reviewed README stage names and renderer arguments agree with the current command entry points. Normal outputs remain beneath the configured data/figure roots rather than the manuscript snapshot.

## Corrected resume and verification defect

Before correction, a completed method identity did not include masks/response input paths, input modification stamps, or all ordered axes. A two-query CPU fixture proved that changing only the response path to negated responses silently reused Full LDS `+1.0`, although the correct value was `-1.0`; verification still reported PASS.

`Codes/src/balds/workflows/benchmark.py` now writes `snr_method_v2` identities containing data-relative score/mask/response addresses, byte sizes, nanosecond modification times, and ordered training/query axis identities. Arithmetic progressions are represented as start/stop/step, avoiding repeated 50,000-element metadata arrays. Changed contracts fail with an instruction to choose a fresh output root; archived results are not overwritten. Input symlinks resolving outside the data root are rejected.

Saved-result verification still reads archived v1 results. For both versions it recomputes Full and every SNR-threshold LDS using the current aligned responses, checks both checkpoint metadata and published per-query statistics, and retains absolute numerical tolerance `1e-12` without refitting KDE. New v2 results additionally enforce the input stamps/axis contract. These stamps are a lightweight change detector, not a content hash: edits deliberately preserving both size and mtime are outside that contract.

Validation:

```sh
cd Codes
python -m pytest -q tests/test_benchmark_resume_identity.py tests/test_snr_background.py
```

Result: **26 passed**. Cases cover unchanged resume, changed response path/content, masks, scores, training order, query order, archived-v1 readback, primary/secondary-threshold correlation tampering, and symlink escape. A separate read-only check of the actual accepted CIFAR-2 seed42 generation archive passed **300 queries** across FMAS, DAS, and pixel-dot, including failed-fit records. No fit, score, model, or source-data regeneration was performed.

## Additional observations sent to the parent reviewer

At the time of review, partial score-query coverage was added only to the in-memory panel summary after the method summary file had been saved; downstream summary reads could lose missing-query counts. Also, generic cross-seed query CIs used the intersection of valid IDs and common draws for generation as well as validation, unlike the generation bootstrap protocol already correctly implemented for Tables 8/9. Neither changes accepted final table means: selected available benchmark cells each have 100 queries, and main multi-seed tables report seed SD. These observations were handed to the parent reviewer for separate corrections and final combined testing.

## Parent integration follow-up

Both handoff observations were corrected in the final summarizer: panel-level requested/missing coverage overrides evaluated-column counts; generation bootstraps use independent within-model draws while validation shares query IDs and keeps each seed's valid set. Regression tests cover disjoint valid query IDs, anticorrelated paired validation draws and partial coverage. Final integrated suite:326passed. Filed paper means, seed SD, coverage, threshold means and global selection match A4; only optional query CIs change. See ../summary_readback.json and ../FINAL_VALIDATION.json.
