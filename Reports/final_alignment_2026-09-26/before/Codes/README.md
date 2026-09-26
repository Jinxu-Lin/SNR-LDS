# BA-LDS

Paper code for background-adaptive Linear Datamodeling Score. The package separates CPU evaluation from model training and attribution. See `../Reports/` for the paper's experiment recipes and artifact migration plan.

## Install

From this directory, in a Python 3.10+ virtual environment:

```bash
python -m pip install .
balds --help
```

The base install uses NumPy, SciPy and PyYAML. It supports evaluating existing score matrices without PyTorch. Install only the additional components you need:

```bash
# Pixel CFM/DDPM training, gradient features, curvature and CLIP baselines
python -m pip install '.[train]'

# ArtBench SD3.5 + LoRA (includes the training dependencies)
python -m pip install '.[train,latent]'

# Optional TRAK projector, paper figures and development tests
python -m pip install '.[train,projection,figures,dev]'
```

Choose matching PyTorch/torchvision builds for your accelerator. `train` bounds Hugging Face datasets to versions below 5 because the retained CIFAR cache format and dataset identifiers were validated with 4.7.0. `configs/constraints-cpu-tested.txt` records the tested Python 3.12 environment; it is a reproducibility reference, not a guarantee that every GPU combination works.

The `cuda_jl` projector also requires the separately built `fast_jl` extension. The pure PyTorch `torch_chunked` projector has no extension dependency. Its basis differs from `cuda_jl`: train and query features must use the same projector, dimension and seed. Never mix newly generated fallback features with archived CUDA-JL features. Set `--set featurize.projection=torch_chunked` explicitly for a new run if needed.

## Data and model locations

```bash
export BALDS_DATA_ROOT=/path/to/your/BA-LDS/_Data
```

`balds-run --data-root PATH ...` overrides that environment variable. In a checkout the default is the top-level `_Data/`; an installed wheel outside a checkout uses `_Data/` below the working directory. Derived manifest, cache and raw-data paths move with the selected root. Storage is local only.

No experiment artifacts have been copied into this release yet. The migration plan in `../Reports/` identifies the required files and their future relative locations. A fresh reproduction needs the raw data and the model weights used by its platform:

- CIFAR datasets are acquired through Hugging Face datasets into `_Data/hf_cache`.
- ArtBench reads `_Data/raw/artbench-10-imagefolder-split`; its SD3.5 weights use the `artbench.base_model` setting. Supply access to the upstream model and image dataset before running the latent stages.
- The imported DAS track requires the original release archive under `_Data/raw/das_archive`, or an explicit `import-das --root` path. The archive includes inputs needed for its distinct split and DDPM implementation.
- CLIP and pretrained detector/model components use their normal upstream caches. Downloads follow the user's `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and Hugging Face cache settings; this package does not force global offline mode.

## Evaluate an existing cell with zero-mean SNR-LDS

Inputs are aligned matrices: scores `(N,Q)`, retained-training masks `(M,N)` and responses `(M,Q)`. NPY and the formats documented by the command are supported.

```bash
balds evaluate --rule snr_zero_mean_gaussian_v1 --zetas 1,2,3,4 \
  --primary-zeta 3 --scores scores.npy --masks masks.npy \
  --responses responses.npy --output evaluation --save-fits

# DAS input is signed pre-square t: fit t, aggregate native t².
balds evaluate --rule snr_zero_mean_gaussian_v1 --scores das_presquare.npy \
  --score-space das-presquare \
  --masks masks.npy --responses responses.npy --output das_evaluation

# Evaluate the portable paper panel described in a manifest.
python tools/prepare_benchmarks.py --data-root "$BALDS_DATA_ROOT" \
  --derived-input-root results/snr_lds_20260925/a3/inputs \
  --source-manifest "$BALDS_DATA_ROOT/results/c10_cpu_closeout_20260923/inputs/bench_manifest.json" \
  --source-manifest "$BALDS_DATA_ROOT/results/ddpm_noncore_val_repair_20260924/cpu/ba_manifest.json" \
  --source-manifest "$BALDS_DATA_ROOT/results/ddpm_noncore_val_repair_20260924/shared_gpu/ba_manifest.json" \
  --output "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/inputs/benchmark_manifest.json"
balds batch --rule snr_zero_mean_gaussian_v1 --zetas 1,2,3,4 --primary-zeta 3 \
  --manifest "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/inputs/benchmark_manifest.json" \
  --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels" --save-fits
balds status --manifest "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/inputs/benchmark_manifest.json" \
  --output "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels"
balds verify --manifest "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/inputs/benchmark_manifest.json" \
  --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels"
```

The SNR rule computes one exact float64 101-point KDE fit per query, then reuses
it for strict two-sided thresholds 1/2/3/4. The zero-mean fit has only `A` and
`C*x²`; it does not estimate a mean or mixture proportion and never evaluates
an N-point KDE at the training coordinates. Full LDS, SNR-LDS, fit failures,
valid empty sets and constants remain separate outputs. Query checkpoints are
published atomically and a repeated `batch` command resumes only checkpoints
whose full rule/source/query identity matches. Archived BA outputs retain their
old rule and `ba_lds` field; they are not relabelled as SNR-LDS.

The manifest preparer resolves actual artifact IDs and can overlay explicitly
named completed manifests. Use `--restore-ab2-das` only with a caller-owned
`--derived-input-root` to reconstruct ArtBench signed lambda=1 DAS inputs from
the existing features. It does not write into historical scientific inputs.
`--fit-scores` permits explicit fitting values when they differ from aggregation
values; it must not be used together with `--score-space`.

For appendix controls, `python tools/aggregation_controls.py snr-transforms ...`
fits the mask once on raw scores and freezes it across filed head/square
readouts. `balds-repeat summarize` now also writes `snr_statistics.json` and
`snr_coordinates.npz`: the independent pilot supplies the frozen zeta=3 mask,
while the fixed repeat 0–7/8–15 cross-reference uses sample SD (not SD/sqrt(8)).

## Training and attribution stages

`balds-run` exposes the retained experiment stages. Global options such as `--data-root`, `--device` and `--set` precede the subcommand. Run each stage's `--help` for all supported parameters. The following illustrates the stage order for one unconditional CIFAR-2 CFM identity; the reports specify the exact paper seeds, subset chains, damping grids, query tracks and baseline configurations.

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

`fmas` resolves to the current paper's signed bilinear score, `fmas_raw`. Background adaptation is applied by `balds evaluate`/`batch` after attribution; it is not a shrinkage layer baked into FMAS. The DAS signed pre-square values remain available for the required background fit.

The registry also retains EK-FAC IF, DAS/D-TRAK, gradient similarity, TracInCP/GAS, Journey-TRAK, and embeddings. Parameter-weighted D-TRAK, AbU+, and NDA remain available as historical implementations, but were retired from the paper's experimental baselines on 2026-09-25 and are discussed only as related work. They are excluded from paper benchmarks and missing-result tasks. Additional stage commands cover deterministic contamination retrieval (`inject`), top-k removal/retraining (`counterfactual`), SD3.5 latent preparation (`latents`) and DAS archive import (`import-das`).

## Independent MC repeatability

The bundled R16 configuration fixes the original 16 generated and 16 validation query positions, 5,000 training rows, 16 independent repeats, FMAS rho 0.01 and D-TRAK lambda 0.05. It preserves the recorded RNG streams. A prepared run reads existing main-checkpoint, curvature, generation, masks and response artifacts.

```bash
balds-repeat prepare --output results/repeatability/r16
balds-repeat run --output results/repeatability/r16 --repeat 0 --device cuda:0
balds-repeat status --output results/repeatability/r16
balds-repeat summarize --output results/repeatability/r16 \
  --pilot-manifest pilots.json
```

Run `--repeat 0` through `--repeat 15` independently. `--output` must be a subdirectory of the selected data root. `--config` accepts an explicit recipe instead of the bundled one. The independent pilot manifest has the following structure; paths resolve relative to the manifest, and each array is `(5000,16)` in the recipe's selected query order:

```json
{
  "fmas_raw": {"gen": "pilot_fmas_gen.npy", "val": "pilot_fmas_val.npy"},
  "dtrak_T100": {"gen": "pilot_dtrak_gen.npy", "val": "pilot_dtrak_val.npy"}
}
```

Missing repeats are reported explicitly. They are never replaced with reused main scores or fabricated results.

## Validation and code map

### Existing SNR A3 results: postprocessing repair

Use the delivered r4 snapshot for the 2026-09-25 repair:

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh launch
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh status
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh verify
```

Closeout reads the existing `results/snr_lds_20260925/a3/{inputs,panels}`
without modifying them and writes only `a4/{tables,edel,figures,verify_a.json,launcher_a}`.
It never calls prepare, batch, KDE fitting, training or scoring. All-failed SNR
groups keep the SNR schema and NA values; Full query intervals use the same
valid queries as SNR. Verification uses float64 before DAS squaring and the
producer's row-wise dot-product order, retaining `rtol=0, atol=1e-12`.
Five currently absent DDPM method/track score inputs remain explicit missing;
they are not fabricated or treated as failed fits. B is not included.

`resume` is available with the same `SNR_STAGE=closeout` environment after the
previous worker and children have exited; preserve its launcher log first.
The launcher uses `nohup setsid` for session isolation. Do not invoke the default
full mode for this closeout, and do not overwrite the older read-only snapshots.

```bash
python -m pytest tests -q
python tools/check_layers.py
```

The retained CPU tests exercise score kernels, RNG identity, damping, subset ground truth, artifact addressing, atomic writes, row/query sharding, CLI dispatch and small workflow stubs. They do not reproduce GPU training or paper measurements. Read `ARCHITECTURE.md` and `../Reports/CODE_CORE.md` for the retained module boundaries, source provenance and actual validation results.
