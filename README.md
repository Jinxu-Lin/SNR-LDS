# SNR-LDS

**Noise-aware evaluation of training data attribution in image generation**

[Anonymous code](https://anonymous.4open.science/r/SNR-LDS-65B2/) · [Data & artifacts](https://drive.google.com/drive/folders/1n0ifYq0msF78UByf8Xs5OLgHN_BpxnRi?usp=sharing) · [Experiment guide](Experiments/README.md) · [Technical documentation](Codes/README.md)

## Overview

How reliably can we compare training data attribution methods when their scores are noisy?
Low-influence samples often have scores near zero, where sampling noise can obscure the attribution signal.
Aggregating these scores can make the Linear Datamodeling Score (LDS) sensitive to score reweighting,
even when the underlying influence estimates do not improve.

This project studies that measurement problem and provides:

- **SNR-LDS:** a common signal-to-noise selection rule for subset aggregation. Noise scales are estimated separately for each method and query; scores above a shared empirical SNR threshold are retained.
- **Controlled-source retrieval:** a complementary benchmark measuring how well attribution methods recover injected training sources of generated concepts.
- **FMAS:** an influence-function-based attribution method instantiated for Flow Matching and evaluated alongside existing methods.

Experiments cover CIFAR Flow Matching and diffusion models, and Stable Diffusion 3.5 Medium fine-tuned on ArtBench-2.
The repository includes implementations, configurations, reproduction protocols, and selected numerical verification records.

## Quick start

### 1. Obtain the anonymous code

Access the source through the [anonymous repository](https://anonymous.4open.science/r/SNR-LDS-65B2/).
Download and extract the source archive, then open a terminal in the directory containing `Codes/` and `Experiments/`.
All commands below run from that **repository root**. The evaluation commands do not require a Git checkout.

### 2. Install

Use Python **3.10 or newer**, preferably in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./Codes
balds --help
```

The base installation evaluates saved score arrays using NumPy, SciPy, and PyYAML.
Install additional components as needed:

```bash
# Pixel-space CFM/DDPM, attribution features, and plots
python -m pip install -e './Codes[train,projection,figures]'

# SD3.5 / ArtBench workflows
python -m pip install -e './Codes[train,latent,projection,figures]'

# Automated tests
python -m pip install -e './Codes[dev]'
```

Choose PyTorch/torchvision builds appropriate for your accelerator.
The original CUDA-JL projection backend requires an additional extension;
see the [installation notes](Codes/README.md) before regenerating gradient features.

### 3. Download the data

**[Experimental data and artifacts — Google Drive](https://drive.google.com/drive/folders/1n0ifYq0msF78UByf8Xs5OLgHN_BpxnRi?usp=sharing)**

Large artifacts are distributed separately from the code. Download the files required by your experiment
and preserve their relative directory structure. The experiment guides specify the required model weights,
generated queries, gradient features, curvature factors, score arrays, subset masks, and measured responses.

Place the downloaded `_Data` directory alongside `Codes/`:

```text
SNR-LDS/
├── README.md
├── Codes/                 # Implementation, configurations, tests, and tools
├── Experiments/           # Experiment protocols and small evidence records
└── _Data/                 # Downloaded artifacts; excluded from Git
```

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
```

Alternatively, set `BALDS_DATA_ROOT` to an absolute path on another disk.
It must point to the data root itself, not a parent containing an extra nested `_Data/` directory.
Downloading the code alone does not download the data.

## Reproduction workflows

### Evaluate your own attribution scores

Provide aligned score arrays `(N, Q)`, retained-training masks `(M, N)`, and measured query-loss responses `(M, Q)`:

```bash
balds evaluate \
  --scores scores.npy \
  --masks masks.npy \
  --responses responses.npy \
  --zetas 1,2,3,4 \
  --primary-zeta 3 \
  --save-fits \
  --output evaluation
```

`N` is the number of training samples, `Q` the number of queries, and `M` the number of retrained subsets.
For DAS pre-square inputs, add `--score-space das-presquare`: noise estimation uses the signed scores,
while aggregation applies the native square exactly once.

### Recompute the paper's SNR-LDS benchmark

After downloading the required scores, masks, and responses:

```bash
python Codes/tools/reproduce_benchmarks.py prepare
python Codes/tools/reproduce_benchmarks.py evaluate --device cpu
python Codes/tools/reproduce_benchmarks.py postprocess
python Codes/tools/reproduce_benchmarks.py verify
```

These stages evaluate saved artifacts; they do not train models. Outputs default to `_Data/results/paper/snr/`.
The `postprocess` stage also reads the archived deletion-utility table for the method-selection comparison.
To reuse accepted evaluation panels without fitting again, follow the [SNR-LDS protocol](Experiments/05_snr_lds/README.md).

### Replay controlled-source retrieval

With the accepted score arrays and query metadata in place:

```bash
python Codes/tools/replay_retrieval.py \
  --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/paper/retrieval"
```

This recomputes retrieval metrics without retraining or generating new images.
The [retrieval protocol](Experiments/07_source_retrieval/README.md) also documents injection,
query review, parameter selection, and score computation.

### Train models and regenerate inputs

Start with the [training index](Experiments/01_training/README.md) and [shared production pipeline](Experiments/00_pipeline/README.md).
The [CIFAR-2 CFM dossier](Experiments/01_training/cifar2_cfm/README.md) includes the three-seed recipe,
recorded training losses, and checkpoint checksums.
From-scratch reproduction requires the corresponding data, compute resources, and any upstream model access.

## Experiments at a glance

| Experiment | What it examines | Guide |
|---|---|---|
| Score distributions | Concentration near zero and large attribution signals | [Magnitude analysis](Experiments/02_magnitude/README.md) |
| Repeated estimation | Sampling variation across 16 repeated measurements | [Sampling noise](Experiments/03_sampling_noise/README.md) |
| Aggregation controls | Head selection, power transformations, and score squaring | [Reweighting](Experiments/04_reweighting/README.md) |
| SNR-LDS benchmarks | Method comparisons, threshold sensitivity, and query coverage | [SNR-LDS evaluation](Experiments/05_snr_lds/README.md) |
| Deletion and retraining | Actual effects of removing identified influencers | [Deletion experiments](Experiments/06_deletion/README.md) |
| Controlled-source retrieval | Recovery of injected sources across semantic similarity groups | [Source retrieval](Experiments/07_source_retrieval/README.md) |

The [experiment index](Experiments/README.md) maps these workflows to the current paper's figures and tables.
Detailed dossiers currently include Chinese-language protocol notes, with executable commands and configurations alongside them.

## Code navigation

| Location | Contents |
|---|---|
| [Codes/src/balds/](Codes/src/balds/) | Attribution, evaluation, models, and workflow implementations |
| [Codes/configs/](Codes/configs/) | Benchmark and repeated-measurement configurations |
| [Codes/tools/](Codes/tools/) | Benchmark reproduction, retrieval replay, and table rendering |
| [Codes/Figure/](Codes/Figure/README.md) | Mechanism figure calculations and rendering |
| [Codes/tests/](Codes/tests/) | Automated tests |
| [Experiments/](Experiments/README.md) | Protocols, training records, and numerical evidence |

The Python distribution is **`snr-lds`**. The import package `balds`, commands
`balds`, `balds-run`, `balds-repeat`, and `BALDS_*` environment variables retain their names for compatibility.
**FMAS is an attribution method within SNR-LDS, not a separate project name.**

## Reproducibility notes

- **Artifact replay and new training are different workflows.** Reconstructed training commands are distinguished from original run evidence.
- **Coverage is explicit.** Some DDPM baseline entries are author-provided aggregate values without per-query score artifacts; see the SNR-LDS guide.
- **Numerical differences are documented.** Stable head tie handling and float32 reconstruction can produce small differences from archived results; see the reweighting guide.
- **The manuscript is not bundled.** The evaluation commands above do not require `Paper/`. Optional comparisons against manuscript tables require its sources separately.

```bash
python -m pytest Codes/tests -q
```

The recorded validation passed **329 tests**; this validates code behavior, not a fresh execution of all GPU experiments.
See the [validation notes](Experiments/validation/README.md) and [technical documentation](Codes/README.md) for further details.
