# Core alignment — 2026-09-26

This entry records changes after the September 22 release and preserves the later user edits. The complete pre-alignment code is saved in `before/Codes/`; historical `Reports/CODE_CORE.md` evidence is not rewritten. `/path/to/BA-LDS/Paper` was authoritative and remained untouched. CFA and the accepted worktrees were read only.

## Final scope and implementation

- Removed the retired parameter-weighted D-TRAK, AbU+, NDA implementations, grouped-gradient producer, orchestration, CLI commands, artifact kinds/codecs and exclusive tests. Kept ordinary D-TRAK, `l1norm_T100`, the matched-readout numerical transforms and feature kernels required by the final mechanism tables.
- R16 now reports `sum(sample variance) / sum(repeat mean squared)`, with no centering or averaging of coordinate-wise ratios. Figure 1 uses ten equal bins in ascending absolute repeat mean for each query. Appendix five-band analysis retains the independent pilot and band-only LDS. The independent repeated-SNR calibration/cross-reference branch is retired; `coordinates.npz` and `statistics.json` remain the public summary products.
- Replaced the CFM common200 retrieval producer with explicit CFM/DDPM reviewed-v2 support: 100 validation and all 400 test queries, original query positions preserved as MC keys and block identity. A stale v1 bundle or 200-query subset is rejected. Both process and query-source identity participate in validation. Existing streamed kernels, MC250/250, gradient chunk125, query chunk100 and row chunk1000 are retained.
- Added portable DDPM reference and four-checkpoint adapters. Reference embeddings and shared projected features preserve the accepted source recipes. The retrained family retains steps2000/4000/6000/8000 and a separate dataset identity. TracInCP/GAS use the uniform mean of four matched checkpoint pairs. Actual selected query labels survive GT slicing and output manifests. Neither aggregate results nor a final-only checkpoint can replace missing scientific inputs.
- Added archived retrieval replay, allowing the accepted separate validation/test artifacts to be evaluated without pretending they follow the new full500 block layout.

## Selected sources

| Source | Adapted scope |
|---|---|
| `3756ec8c224b9a318b5ce384e0f5ee6f935b3e83`, `run_noncore.py` | DDPM pixel/CLIP, shared TRAK/gradient/relative-IF/renormalized-IF recipes and response-column view |
| `19c31140c300d344f827f15e04d5ff460ad9f6ac`, `run_j3.py` | Separate retrained checkpoint trajectory; four-step feature and score recipe |
| `ca43565544504c207cdc64fbefd0d502fe2fcfe3`, `cfm_val100.py` | Reviewed-v2 validation identity and native damping selection |
| `ebbc59f5719af6d612e12576adcc4f674dedbc62`, `cfm_test400.py` | Original test IDs, fixed validation choice, full400 results |
| `4993acd4f1fcdc4f42d180b9fadce00e23d8a0de` | DDPM original-query-ID curvature protocol |
| Accepted `Reports/figure_split_2026-09-25/render_relative_variation.py` formula | Ratio-of-sums VarRatio and ascending absolute-mean deciles |

No whole worktree was copied. Source machine launchers, SSH/rsync and retired BA orchestration were excluded. Current module records are in `Codes/configs/core_provenance.json`; entries for deleted destinations were removed after the complete previous catalog was preserved in the snapshot.

## Verified behavior

| Check | Result |
|---|---|
| Owned full CPU suite: retained core plus source-scoring/assembly and DDPM adapters | **283 passed**, 57.97s; `core-tests.txt` |
| Final focused source, DDPM, retrieval-replay and R16 suite | **19 passed**, 1.04s; `core-focused-final.txt` |
| Layer boundary checker at handoff | **90 modules passed** |
| Actual archived retrieval replay | **26 methods × 400 test queries = 10,400 query metrics**; all 572 printed numeric cells of Tables2/18/19 agree at printed precision |

The retrieval replay also recomputed all 30 curvature validation matrices, recovering CFM IF1e-9, DDPM IF1e-12 and FMAS rho1 on both platforms. For the 22 native projected/embedding method cells it checked the complete archived validation curve and first-maximum order, then recomputed the selected matrix's validation pool AP. Full per-damping projected score banks were not required or claimed to have been replayed.

The first full test attempt exposed two stale tests: retired E4 addresses and an old `cifar2` registry key that only became live when the newly migrated raw cache was present. The obsolete E4 test was removed with its implementation. The retained5k-loader test now verifies deterministic balanced subsampling and original row order with an in-memory source, so test execution no longer mutates an existing Hugging Face cache.

The first paper comparator used Python formatted-string equality, which treated three binary half-rounding ties as mismatches. It was corrected to compare the displayed decimal interval (`absolute error <= half of the last printed unit + 1e-12`), without changing scores or computed metrics. `retrieval-verification.json` explicitly records all boundary cells and reports zero actual mismatches.

CUDA was disabled and the existing environment was not modified. No training, model inference, real gradient extraction, curvature fitting, external download or source-file hashing was performed. GPU execution and missing DDPM checkpoint/Journey results remain outside this validation.

## Public commands

Run from the release root with the installed package and `BALDS_DATA_ROOT` set.

```bash
python Codes/tools/replay_retrieval.py \
  --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/paper/retrieval" --paper Paper
```

Outputs are `{cfm,ddpm}/{method}.json` and `summary.json`. `--manifest` supplies alternative relative archive addresses; `--train-labels` can supply a50000-row NumPy label array, otherwise cached CIFAR labels are read. The command uses saved native retrieval scores directly; it does not apply the LDS-specific DAS readout transformation to those archived rankings.

```bash
python Codes/tools/score_retrieval.py prepare --process cfm \
  --input-data-root "$BALDS_DATA_ROOT" \
  --output-root "$BALDS_DATA_ROOT/results/paper/retrieval_blocks/cfm" --method fmas_raw
# Change prepare to run (with --device/worker flags), then status.
python Codes/tools/assemble_retrieval.py --process cfm \
  --data-root "$BALDS_DATA_ROOT" \
  --sources "$BALDS_DATA_ROOT/results/paper/retrieval_blocks/cfm" \
  --output "$BALDS_DATA_ROOT/results/paper/retrieval_fresh/cfm" --method fmas_raw
```

The same commands accept `--process ddpm` and `--method ekfac_if`. `--selection` is optional and, if supplied, must name all400 original test positions. `--comparison-manifest` is optional for the new block assembler. Use separate output roots for different processes. Accepted historic val/test lanes are replayed with `replay_retrieval.py`, not silently overlaid into the new full500 producer identity.

```bash
python Codes/tools/ddpm_baselines.py features --family checkpoints \
  --data-root "$BALDS_DATA_ROOT" --manifest "$DDPM_PANEL_MANIFEST" \
  --output "$BALDS_DATA_ROOT/results/paper/ddpm_checkpoints" \
  --checkpoint-root results/ddpm_gaps_20260923/retrain_j2 --device cuda:0
python Codes/tools/ddpm_baselines.py score --family checkpoints \
  --data-root "$BALDS_DATA_ROOT" --manifest "$DDPM_PANEL_MANIFEST" \
  --output "$BALDS_DATA_ROOT/results/paper/ddpm_checkpoints"
```

`--family reference --line embedding` produces the four image baselines; `--family reference --line gradient` produces the five shared projected baselines. `--steps` can assign a subset of the four checkpoint feature jobs; scoring always requires all four. The emitted `benchmark_manifest.json` is the input to `balds batch`. Reference source embeddings reuse the original CIFAR10 rows through the DAS train-index mapping. Foreign/incompatible feature or score identities are rejected, so fresh reproduction requires a clean compatible artifact family rather than overwriting an unrelated historical run.
