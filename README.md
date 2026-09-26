# SNR-LDS

Code, configurations, tests, and experiment reports for SNR-LDS and training
data attribution in image generation models.

## Repository layout

- [Codes/](Codes/): current implementation, configurations, tests, and tools.
- [Experiments/](Experiments/README.md): experiment-oriented protocols, execution scripts, training evidence, and numerical validation.

## Installation

Use Python 3.10+ in a virtual environment:

```bash
cd Codes
python -m pip install .
balds --help
```

See [Codes/README.md](Codes/README.md) for optional dependencies, training,
evaluation, and reproduction commands, and [Experiments/README.md](Experiments/README.md)
for the paper's experiment map and the three-seed CIFAR-2 CFM training dossier.

## Data and manuscript

Datasets, model weights, gradient features, and other large data artifacts are
not included in Git. Set BALDS_DATA_ROOT to your external data directory, or
place artifacts under the ignored _Data/ directory. The Paper/ manuscript and
its archived source/rendered copies are also excluded. Historical reports may
reference these local files; those links require the separately available assets.

The project is **SNR-LDS** and its Python distribution is **snr-lds**. The
import package, commands (balds, balds-run, balds-repeat), and BALDS_* environment
variables retain their existing names for compatibility. FMAS is an attribution
method within this project, not an alternative project name. Old names in dated
snapshots, provenance records, and stable protocol identifiers describe historical
inputs rather than the current release.
