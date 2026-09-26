# Architecture

The public package is `balds`, installed from the `src/` layout. Importing its CPU BA evaluator does not import PyTorch. Optional training and model dependencies are loaded only by their workflows or numerical kernels.

| Package | Responsibility |
|---|---|
| `schema` | Run identity, artifact kinds, protocol interfaces, registries and stable RNG/hash primitives |
| `data` | Active CIFAR/ArtBench datasets, masks and deterministic CIFAR contamination |
| `models` | CFM/DDPM training and sampling, imported DAS DDPM adapter, SD3.5 LoRA |
| `attribution` | Projected gradients, embeddings, kernels, curvature, direct baselines and reusable transforms |
| `evaluation` | Pure LDS/BA statistics, retrieval/deletion metrics and repeatability estimators |
| `artifacts` | Relative addresses, codecs, local atomic files, manifests and block assembly inputs |
| `workflows` | Connect data, model, attribution and evaluation stages through explicit artifact identities |
| `cli` | Argument parsing for `balds`, `balds-run` and `balds-repeat` |
| `report` | Text tables for workflow results |

`tools/check_layers.py` checks the permitted dependency directions, including relative imports. Workflow-specific experiment files are handled explicitly by workflow adapters; the BA numerical evaluator performs no filesystem IO. `Figure/` contains standalone paper figure/table entry points.

## Stage composition

The former monolithic pipeline is split into `workflows/training.py`, `queries.py`, `features.py` and `subsets.py`. Shared dataset/configuration recipes live in `common.py`; `container.py` composes their local dependencies. Scientific stage bodies were retained while imports and package names changed.

Curvature fitting/scoring stays in `curvature.py` and `curvature_joint.py` because its streamed row/query blocks, deterministic RNG and partial-file assembly are tightly coupled. It is not duplicated into a second scoring implementation. Direct baselines have their own `baselines.py` orchestration. The R16 adapter reuses the original `e3c.py` numerical workflow and supplies portable configuration and a separate summary adapter in `repeatability.py`.

The CPU evaluation path is `cli/ba.py` → `workflows/benchmark.py` → `evaluation/background.py`. Fitting/selection values and aggregation values are explicit separate arrays. The current SNR rule fits a zero-mean `A+C*x²` background on a 101-point grid once per query and derives all requested zeta masks from that fit. Query checkpoints contain the rule/source/query identity, compact fit state, selections and predictions and are published by one atomic rename. For DAS the fit uses signed pre-square `t` and aggregation uses native `t²` exactly once. The current FMAS alias maps to `fmas_raw`. The deletion workflow keeps the archived per-query selector separate from the new three-seed benchmark-level selector.

## Artifact compatibility

Existing relative scientific paths are retained, including checkpoints, query generations, projected features, curvature packages, per-lambda scores, masks, subset losses and responses. Renaming the Python package does not require renaming those artifacts. The data root is supplied through `--data-root` or `BALDS_DATA_ROOT`; no server location or remote backend is embedded in configuration.

The SQLite manifest records provenance, but a manifest row never proves that bytes still exist. Writes retain the original atomic publication/validation behavior. Imported arrays and block assemblies retain their recorded shapes, ordering and explicit identities. Historical protocol strings in RNG/hash primitives intentionally remain unchanged: changing their spelling would change seeds or hashes even when the numerical implementation is identical.

The projection backend is part of feature identity. CUDA-JL and the pure PyTorch fallback use different random bases. Their outputs cannot be combined merely because tensor shapes match. Cross-platform numerical equality is not promised for model training, random projections, mixed precision or linear algebra libraries.

## Scope reduction

Retained datasets are `cifar2_5k`, `cifar10_v2`, `artbench2_256`, imported `cifar2_das` and contamination `cifar10_inj4`. Both generated and validation query tracks remain supported. Pending paper baselines remain available with explicit scientific configuration tags.

Removed orchestration includes predecessor prediction/sign/positive-square campaigns, regional experiments, private remote launch/sync/prune operations, SD1.5, old low-resolution ArtBench and latent foreign-style injection. Shared canonical identity and MC RNG functions were extracted from retired contracts rather than importing an obsolete experiment package.

Reusable transforms and legacy numerical helpers remain where retained workflows or paper figure sweeps depend on them. Historical denoised FMAS method aliases are not registered as the current method; no published default substitutes a shrinkage score for the paper's raw bilinear FMAS.

`configs/core_provenance.json` records the module-level source selection. Integration was selective: baseline fixes, streamed curvature/R16 changes and retrieval replacement support were adapted without copying entire divergent worktrees. The validation report lists the limits of that integration.
