# SNR-LDS

Code for the final paper's **SNR-LDS**, attribution benchmarks and source-retrieval experiments. The GitHub repository is `SNR-LDS`; the package and command prefix remain `balds`. Version 0.2.0 follows the manuscript finalized on 2026-09-26; the retired positive-only BA evaluator and per-query method selector are no longer public evaluation modes.

## Install

The distribution name is `snr-lds`. The `balds` import package, CLI commands,
and `BALDS_*` environment variables are retained for script compatibility;
they are not the current project name. FMAS names an attribution method, not
the repository. Historical protocol IDs and provenance paths remain unchanged.

From `Codes/`, in a Python 3.10+ virtual environment:

```bash
python -m pip install .
balds --help
```

The base package uses NumPy, SciPy and PyYAML. Evaluating NPY score/mask/response arrays does not require PyTorch. Install the components needed by your experiment:

```bash
# Pixel CFM/DDPM training, features, curvature and CLIP
python -m pip install '.[train]'
# ArtBench SD3.5 with LoRA
python -m pip install '.[train,latent]'
# CUDA-JL projection, figure generation and CPU tests
python -m pip install '.[train,projection,figures,dev]'
```

Choose matching PyTorch/torchvision builds for your accelerator. The tested Python 3.12 CPU environment is recorded in `configs/constraints-cpu-tested.txt`. The retained Hugging Face dataset format uses `datasets>=4.7,<5`.

The paper's `cuda_jl` projector also needs the separately built `fast_jl` extension. For a fresh run, `--set featurize.projection=torch_chunked` uses pure PyTorch. These projectors produce different bases: use the same backend, dimension and seed for training and query features; do not mix regenerated features with archived CUDA-JL features.

## Data locations and migrated artifacts

```bash
export BALDS_DATA_ROOT=/path/to/SNR-LDS/_Data
```

In a checkout this variable is optional: the default is the adjacent `_Data/`. Outside a checkout the installed package uses `_Data/` under the working directory. `balds-run --data-root PATH` overrides the environment variable. Outputs and input manifests use data-relative addresses. Historical reports retain their original provenance; current commands do not need the original server or checkout.

The finalized artifact inventory and actual copy receipt are in `../Reports/final_alignment_2026-09-26/migration/`. The local `_Data/` now includes the selected paper inputs, accepted results, main checkpoints, features and factors. See `../_Data/README.md` for included and optional families. The large subset/deletion checkpoint banks and external pretrained weight caches are optional; tables can be recomputed from the copied measured responses without retraining those models. Experiment artifacts are not Python package contents or Git source files.

For reproduction from raw inputs, acquire CIFAR through Hugging Face datasets; supply ArtBench images under `_Data/raw/artbench-10-imagefolder-split` and configure access to the upstream SD3.5 model using `artbench.base_model`. The imported DAS DDPM platform needs the ordered archive split metadata or its original release archive for `import-das`. CLIP/model downloads respect normal Hugging Face cache and offline settings. The migrated standard DDPM checkpoint embeds its model configuration and weights.

## Final SNR evaluator

Inputs are aligned scores `(N,Q)`, retained-training masks `(M,N)` and measured responses `(M,Q)`:

```bash
balds evaluate --scores scores.npy --masks masks.npy --responses responses.npy \
  --zetas 1,2,3,4 --primary-zeta 3 --output evaluation --save-fits

# DAS: fit signed pre-square t, aggregate native t² exactly once.
balds evaluate --scores das_presquare.npy --score-space das-presquare \
  --masks masks.npy --responses responses.npy --output das_evaluation
```

The sole rule is `snr_zero_mean_gaussian_v1`. Each query is RMS-normalized. Silverman's bandwidth and a robust central width define an exact float64 KDE on 101 zero-centered grid points. Density-weighted least squares fits only `A + C*x²`; there is no fitted mean, mixture proportion or extra linear term. A valid negative curvature gives the noise scale. Strict two-sided `abs(score)/sigma > zeta` selects entries without changing their native readout. The fit is reused for thresholds 1/2/3/4; the primary threshold is 3.

Fit failures are NA. Valid empty selections, all-zero scores, constant predictions and constant responses retain their separately counted statuses. DAS fitting never uses its already-squared native score. `--fit-scores` is available for explicitly separate fitting and aggregation arrays; do not combine it with `--score-space`.

## Paper benchmark entry point

Run from `Codes/`. The configuration is `configs/paper.json`. These stages are explicit; preparing or postprocessing never launches training or refits a KDE.

```bash
# Build a portable manifest, using the final accepted source overlay when present.
python tools/reproduce_benchmarks.py prepare

# Recompute all available SNR cells, saving fits and resumable query results.
python tools/reproduce_benchmarks.py evaluate --device cpu
python tools/reproduce_benchmarks.py status
python tools/reproduce_benchmarks.py verify
python tools/reproduce_benchmarks.py postprocess
```

A fresh run can prepare directly from newly produced canonical artifacts without the optional archived overlay; the preparer reports which overlays are available. Outputs default to `_Data/results/paper/snr/`. Use `--output` for another output root. Atomic query checkpoints resume only matching scientific identities.

To reuse the migrated results without fitting again:

```bash
python tools/reproduce_benchmarks.py prepare
python tools/reproduce_benchmarks.py postprocess \
  --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels"
python tools/reproduce_benchmarks.py verify \
  --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels"
python tools/render_snr_tables.py \
  --summary "$BALDS_DATA_ROOT/results/paper/snr/tables/summary.json"
python tools/render_deletion_tables.py
python tools/render_fig3_snr_controls.py
python tools/render_deletion_visual.py
```

When using the shell examples with `$BALDS_DATA_ROOT`, set it first (for the default checkout, `export BALDS_DATA_ROOT="$(cd ../_Data && pwd)"`). Figure/table outputs go under `_Data/results/paper/figures/`, or `BALDS_FIGURE_ROOT`; no renderer writes the finalized Paper. `Figure/README.md` lists the mechanism figures and matched-readout tables. Reports link each active paper figure/table to inputs and commands.

The accepted array-level archive has 243 cells and 24,300 query records. Five DDPM score inputs are absent: TracInCP/GAS generation and validation, plus Journey-TRAK generation. The manuscript supplies four rounded TracInCP/GAS aggregate means separately in `configs/paper_aggregate_supplement.json`; those four values have no query coverage and are not inserted into the per-query data. Journey-TRAK validation is outside the assessed definition. Pixel/CLIP NA entries are fit failures, not missing files.

Deletion selection is global: intersect valid queries across four methods within each of three C2 model seeds, compute each seed mean, then average seeds equally. Original LDS selects DAS and SNR-LDS selects FMAS. Their native deletion utilities use the same 50 seed42 generation queries at budgets 300/1000. The random visual comparator averages five models within each query before averaging queries; CLIP is cosine similarity and pixel L2 uses common uint8/255 images.

## Training and attribution

Global options such as `--data-root`, `--device` and `--set` precede the `balds-run` subcommand. Example stage order for unconditional CIFAR-2 CFM:

```bash
balds-run --device cuda:0 train --dataset cifar2_5k --seed 42 --uncond
balds-run --device cuda:0 generate --dataset cifar2_5k --seed 42 --uncond
balds-run --device cuda:0 featurize --dataset cifar2_5k --seed 42 --uncond \
  --feat dtrak_T100 --split both --query-type gen
balds-run --device cuda:0 ekfac fit --dataset cifar2_5k --seed 42 --uncond --method fmas
balds-run --device cuda:0 ekfac score --dataset cifar2_5k --seed 42 --uncond \
  --method fmas --query-type gen
balds-run subsets masks --dataset cifar2_5k --seed 42 --uncond
balds-run --device cuda:0 subsets train --dataset cifar2_5k --seed 42 --uncond
balds-run --device cuda:0 subsets losses --dataset cifar2_5k --seed 42 --uncond --query-type gen
balds-run subsets gt --dataset cifar2_5k --seed 42 --uncond --query-type gen
balds-run --device cuda:0 score --dataset cifar2_5k --seed 42 --uncond \
  --method dtrak_T100 --query-type gen
```

`fmas` resolves to the signed bilinear `fmas_raw`; SNR selection is a later evaluation stage, not a modification to FMAS. Retained baselines include EK-FAC IF, DAS/D-TRAK/TRAK, gradient similarity, TracInCP/GAS, Journey-TRAK, Relative/Renormalized IF and embeddings. PW-DTRAK, AbU+ and NDA are discussed only in related work and have been removed from the experimental registry and CLI. Other retained stages cover contamination (`inject`), removal/retraining (`counterfactual`), SD3.5 latents and DAS archive import.

`tools/ddpm_baselines.py` separates the imported reference-model family from the retrained four-checkpoint family. Use `--family reference|checkpoints`; features retain the filed T100, p4096 CUDA-JL identity. Checkpoint TracInCP/GAS average steps 2000/4000/6000/8000. Its emitted manifest carries actual query IDs and family identity. Newly produced scores do not certify the four author-supplied aggregate entries.

## Reviewed500 source retrieval

Both CFM and DDPM use 500 reviewed queries: 100 validation and 400 test, with original source positions preserved in MC random streams. Validation AP selects damping; test AP/Recall@200 use global training-set rankings and macro-averaging across ten concepts. CFM and DDPM have separate model and artifact identities.

To replay all 13 methods from the migrated score arrays and compare the final retrieval tables:

```bash
python tools/replay_retrieval.py --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/paper/retrieval" --paper ../Paper
```

`tools/score_retrieval.py` runs tiled FMAS/IF scoring with explicit `--process cfm|ddpm`; `tools/assemble_retrieval.py` handles complete coverage, validation selection and metrics. Use the exact pipeline and archived score replay commands in `../Reports/experiments/E_SOURCE.md`; partial common200 results are not the final test set.

## R16 variance analysis

The bundled repeat recipe fixes the original 16 generation and 16 validation query positions, 5,000 rows and 16 independent repeats. It retains FMAS rho 0.01, D-TRAK lambda 0.05 and the original RNG streams.

```bash
balds-repeat prepare --output results/repeatability/r16
balds-repeat run --output results/repeatability/r16 --repeat 0 --device cuda:0
balds-repeat status --output results/repeatability/r16
balds-repeat summarize --output results/repeatability/r16 --pilot-manifest pilots.json
```

Run repeat IDs 0 through 15 independently. Pilot-manifest keys are `fmas_raw`/`dtrak_T100`, each containing `gen`/`val` paths to `(5000,16)` arrays in recipe query order. Paths resolve relative to the manifest. Final VarRatio is `sum(sample variance) / sum(repeat mean²)` per region/query. The final panels use pilot bands and absolute-repeat-mean deciles; retired repeat-SNR calibration is not emitted. See `../Reports/experiments/M2A_REPEATABILITY.md` for archived R16 replay and full preparation.

## Validation and architecture

```bash
python tools/check_layers.py
python -m pytest tests -q
```

CPU tests cover SNR numerics, DAS readouts, global choice, RNG/identity, score kernels, damping, subset responses, storage and the retained workflow adapters. They do not rerun GPU training. `ARCHITECTURE.md` describes the package boundaries. Current copy, numerical readback and validation receipts live under `../Reports/final_alignment_2026-09-26/`; older reports describe their dated versions.
