"""Composition root — the ONLY module that constructs concrete implementations.

Builds the content-addressed store from config, stamps the git ``code_version``,
fires the registry side-effects (importing the business packages), and hands the
CLI fully-wired use-cases. Nothing else in the codebase ``new``s a store/backend.
"""
from __future__ import annotations

import subprocess

# importing these registers processes, datasets, methods, transforms, selectors
import balds.data  # noqa: F401
import balds.evaluation  # noqa: F401
import balds.evaluation.lambda_select  # register selectors
import balds.models  # noqa: F401
import balds.attribution  # noqa: F401
from balds.schema.logging import configure_logging
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
from .config import load_config
from .counterfactual import CounterfactualUseCase
from .curvature import EkfacFitUseCase, EkfacRepeatsUseCase, EkfacScoreUseCase
from .das_import import DasImportUseCase   # also registers the cifar2_das dataset key
from .inject import InjectUseCase
from .latent import EncodeUseCase
from .features import FeaturizeUseCase
from .queries import GenerateUseCase
from .subsets import SubsetsUseCase
from .training import TrainUseCase
from .usecases import EvaluateUseCase, ScoreUseCase
from .baselines import AbuUseCase, NdaUseCase, ParameterWeightingUseCase


def _git_sha(*, short: bool = True) -> str:
    try:
        args = ["git", "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        return subprocess.check_output(
            args, text=True,
            stderr=subprocess.DEVNULL).strip() or "dev"
    except Exception:
        return "dev"


class Container:
    """Wires the store + use-cases from a loaded config."""

    def __init__(self, overrides: dict | None = None, *, device: str = "cuda:0") -> None:
        configure_logging()
        self.cfg = load_config(overrides)
        self.code_version = _git_sha(short=True)
        self.full_code_version = _git_sha(short=False)
        try:
            self.code_dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip())
        except Exception:
            self.code_dirty = True
        self.device = device
        self.store = self._build_store()

    def _build_store(self) -> ArtifactStore:
        s = self.cfg["storage"]
        backends = [LocalBackend(s["data_root"])]
        manifest = ManifestDB(s["manifest_db"])
        return ArtifactStore(manifest, backends, code_version=self.code_version)

    # --- use-case factories ---------------------------------------------
    def score_usecase(self) -> ScoreUseCase:
        return ScoreUseCase(self.store, self.cfg, device=self.device)

    def evaluate_usecase(self) -> EvaluateUseCase:
        return EvaluateUseCase(self.store, self.cfg)


    def train_usecase(self) -> TrainUseCase:
        return TrainUseCase(self.store, self.cfg, device=self.device)

    def generate_usecase(self) -> GenerateUseCase:
        return GenerateUseCase(self.store, self.cfg, device=self.device)

    def featurize_usecase(self) -> FeaturizeUseCase:
        return FeaturizeUseCase(self.store, self.cfg, device=self.device)

    def subsets_usecase(self) -> SubsetsUseCase:
        return SubsetsUseCase(self.store, self.cfg, device=self.device)

    def encode_usecase(self) -> EncodeUseCase:
        return EncodeUseCase(self.store, self.cfg, device=self.device)

    def ekfac_fit_usecase(self) -> EkfacFitUseCase:
        return EkfacFitUseCase(self.store, self.cfg, device=self.device)

    def ekfac_score_usecase(self) -> EkfacScoreUseCase:
        return EkfacScoreUseCase(self.store, self.cfg, device=self.device)

    def counterfactual_usecase(self) -> CounterfactualUseCase:
        return CounterfactualUseCase(self.store, self.cfg, device=self.device)

    def ekfac_repeats_usecase(self) -> EkfacRepeatsUseCase:
        return EkfacRepeatsUseCase(self.store, self.cfg, device=self.device)

    def das_import_usecase(self) -> DasImportUseCase:
        return DasImportUseCase(self.store, self.cfg, device=self.device)

    def inject_usecase(self) -> InjectUseCase:
        return InjectUseCase(self.store, self.cfg, device=self.device)




    def parameter_weighting_usecase(self) -> ParameterWeightingUseCase:
        return ParameterWeightingUseCase(self.store, self.cfg, device=self.device)

    def abu_usecase(self) -> AbuUseCase:
        return AbuUseCase(self.store, self.cfg, device=self.device)

    def nda_usecase(self) -> NdaUseCase:
        return NdaUseCase(self.store, self.cfg, device=self.device)


def build_container(overrides: dict | None = None, *, device: str = "cuda:0") -> Container:
    return Container(overrides, device=device)
