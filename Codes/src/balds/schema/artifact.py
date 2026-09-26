"""The Data-Access boundary: artifact kinds, manifest records, and the
``ArtifactStore`` abstraction the Business layer depends on (DIP).

Business code never constructs a path, opens a file, or talks to a backend. It
holds an :class:`ArtifactStore` (an abstraction defined here) and asks for
artifacts by ``(ArtifactKind, RunSpec)``. The concrete content-addressed store
lives in :mod:`balds.artifacts` and is wired in by the composition root.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .runspec import RunSpec


class ArtifactKind(str, Enum):
    """Every logically distinct artifact the pipeline reads or writes.

    The string value is *not* a path; :func:`balds.artifacts.addressing.relpath` maps
    ``(kind, RunSpec)`` to the human-mirror path. Kinds are finer-grained than
    top-level directories because, e.g., ``featurize/.../`` holds several files.
    """

    CHECKPOINT = "checkpoint"            # checkpoints/{ds}/{model}/seed_{n}/final.pt
    SUBSET_CHECKPOINT = "subset_ckpt"    # checkpoints/subsets/{fm|ddpm}/{ds}/seed_{n}/subset_{m}/final.pt
    LOSS_HISTORY = "loss_history"        # checkpoints/.../loss_history.json
    GENERATION = "generation"            # generations/{ds}/{model}/seed_{n}/samples.pt
    GEN_TRAJECTORY = "gen_trajectory"    # generations/{ds}/{model}/seed_{n}/trajectory.pt
    GEN_POOL = "gen_pool"                # generations/{ds}/{model}/seed_{n}/pool_{tag}.pt

    LATENTS = "latents"                  # latents/{ds}/{split}[_flip].pt (latent platform)
    PROMPT_EMBEDS = "prompt_embeds"      # latents/{ds}/prompt_embeds.pt

    INJECT_META = "inject_meta"          # subsets/{ds}_inject_meta.json
    INJECT_DETECTOR = "inject_detector"  # results/inject/{ds}/detector.pt
    INJECT_CANDIDATES = "inject_candidates"  # generations/.../pool_{tag}_candidates.json
    INJECT_REVIEW_SHEET = "inject_review_sheet"  # results/inject/.../review_{tag}/sheet_*.png
    INJECT_REVIEW_TABLE = "inject_review_table"  # results/inject/.../review_{tag}/review.csv
    INJECT_QUERIES = "inject_queries"    # generations/.../inject_queries.pt
    INJECT_MINED = "inject_mined"        # generations/{ds}/{model}/seed_{n}/mine_{tag}.pt
    INJECT_RESULT = "inject_result"      # scores/{method}/{ds}_inject/.../inject_result.json
    INJECT_CF_MASK = "inject_cf_mask"    # counterfactual/{ds}/inject_{tag}/seed_{n}/mask.npy
    INJECT_CF_META = "inject_cf_meta"    # counterfactual/{ds}/inject_{tag}/seed_{n}/meta.json
    INJECT_CF_CHECKPOINT = "inject_cf_ckpt"  # checkpoints/inject_cf/{fm|ddpm}/.../final.pt

    SUBSET_MASKS = "subset_masks"        # subsets/{ds}_masks.pkl
    GT_MATRIX = "gt_matrix"              # results/gt_matrix_fm_{ds}{_val}_seed_{n}.npy
    GT_LOSSES = "gt_losses"              # results/gt_losses_fm_{ds}{_val}_seed_{n}[_eseed_{e}].npy
    GT_LOSSROW = "gt_lossrow"            # results/gt_lossrows_fm_{ds}{_val}_seed_{n}_eseed_{e}/subset_{m}.npy
    REPEAT_FEATURES = "repeat_features"  # featurize/{feat}/{ds}[/p..]/seed_{n}/repeat_features.pt
    TRAIN_FEATURES = "train_features"    # featurize/{feat}/{ds}/seed_{n}/train_features.pt
    ERROR_WEIGHT = "error_weight"        # featurize/{feat}/{ds}/seed_{n}/error_train.npy
    QUERY_FEATURES = "query_features"    # featurize/{feat}/{ds}/seed_{n}/query_features_{gen|val}.pt
    FEATURE_META = "feature_meta"        # featurize/{feat}/{ds}/seed_{n}/meta.json
    EKFAC_FACTORS = "ekfac_factors"      # curvature/ekfac/{ds}[/ddpm]/seed_{n}/factors.pt
    SCORES = "scores"                    # scores/{method}/{ds}{_val}/seed_{n}/scores.npy
    SCORES_LAMBDA = "scores_lambda"      # scores/{method}/{ds}{_val}/seed_{n}/scores_lambda_{lam}.npy
    REPEAT_SCORES = "repeat_scores"      # scores/{method}/{ds}{_val}/seed_{n}/repeat_scores.npz (EK-FAC σ̂ path)
    # Block partials of the curvature streams (TASK EKFAC-SHARD, 2026-09-09):
    # one npz per [q0, q1) query block and, one level below (EKFAC-ROWSHARD),
    # optionally per [r0, r1) train-row range — so a 40 h `ekfac score` survives
    # an interruption and can be split across cards/machines on two axes.
    # Intermediates, not results — the whole-matrix SCORES/SCORES_LAMBDA
    # addresses stay whole-matrix.
    SCORES_BLOCK = "scores_block"        # scores/{method}/{ds}{_val}/seed_{n}/blocks/scores_q{q0}_{q1}[_r{r0}_{r1}].npz
    REPEAT_SCORES_BLOCK = "repeat_scores_block"  # .../blocks/repeat_scores_q{q0}_{q1}[_r{r0}_{r1}].npz
    LDS_RESULT = "lds_result"            # results/lds_results_{ds}{_val}.json
    SCORES_META = "scores_meta"          # scores/{method}/{ds}{_val}/seed_{n}/meta.json (λ/γ/selector provenance)

    # E4 direct-baseline precomputes. Final matrices remain SCORES/SCORES_META.

    # AN-2 counterfactual chain (2026-08-30): removal sets -> LoRA retrains ->
    # same-seed regeneration -> L2/CLIP analysis
    CF_MASK = "cf_mask"                  # counterfactual/{ds}/{arm}/seed_{n}/q{qi}.npy  (bool (N,), True = keep)
    CF_META = "cf_meta"                  # counterfactual/{ds}/{arm}/seed_{n}/meta.json
    CF_CHECKPOINT = "cf_ckpt"            # checkpoints/counterfactual/{fm|ddpm}/{ds}/{arm}/seed_{n}/q{qi}/final.pt
    CF_GENERATION = "cf_generation"      # counterfactual/{ds}/{arm}/seed_{n}/q{qi}/regen.pt
    CF_ANALYSIS = "cf_analysis"          # counterfactual/{ds}/analysis/seed_{n}/{name}.json
    CF_FIGURE = "cf_figure"              # counterfactual/{ds}/analysis/seed_{n}/{name}.png



@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row of the manifest: identity -> content + provenance.

    ``upstream`` records the content hashes of the inputs that produced this
    artifact (a ``scores`` blob points back to its exact feature + GT blobs),
    giving journal-grade lineage.
    """

    kind: str
    spec_digest: str
    blob_hash: str
    size: int
    codec: str
    code_version: str
    created_at: str
    upstream: tuple[str, ...] = field(default_factory=tuple)


class ArtifactStore(ABC):
    """Abstraction the Business layer depends on instead of any path/backend.

    Implementations are content-addressed and may transparently fetch from
    remote backends. ``**key`` carries kind-specific selectors (e.g. ``lam`` for
    :attr:`ArtifactKind.SCORES_LAMBDA`, ``step`` for intermediate checkpoints).
    """

    @abstractmethod
    def exists(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> bool:
        """Whether the artifact is materialized locally or on a remote tier."""

    @abstractmethod
    def has_local(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> bool:
        """Whether the artifact's bytes are on THIS machine (no remote lookup).

        Distinct from :meth:`exists` for callers that must not treat a remote
        copy as their own — notably ``balds sync push``, where "the master already
        has it" means *nothing left to do*, not *fetch it back and send it again*.
        """

    @abstractmethod
    def load(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> Any:
        """Materialize and decode the artifact, fetching from a remote tier if needed."""

    @abstractmethod
    def save(
        self,
        kind: ArtifactKind,
        spec: RunSpec,
        obj: Any,
        *,
        upstream: tuple[str, ...] = (),
        **key: Any,
    ) -> ManifestEntry:
        """Encode + store ``obj`` content-addressed; record a manifest entry."""

    @abstractmethod
    def save_atomic(
        self,
        kind: ArtifactKind,
        spec: RunSpec,
        obj: Any,
        *,
        upstream: tuple[str, ...] = (),
        validator: Callable[[Any], None] | None = None,
        replace: bool = False,
        **key: Any,
    ) -> ManifestEntry:
        """Validate a same-directory temporary before atomic publication.

        With ``replace=False`` (the pilot default), conflicting final bytes are
        never overwritten.  Byte-identical orphan bytes may be registered on a
        retry after an interruption between rename and manifest commit.
        """

    @abstractmethod
    def discard(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> bool:
        """Delete THIS machine's copy of an intermediate; return whether it was there.

        The only legitimate callers are artifacts
        whose content is fully captured by an already-saved downstream artifact
        (the per-query-block score partials, once the whole matrix is filed).
        Never call it on a primary artifact — there is nothing to fall back on.
        """

    @abstractmethod
    def local_blocks(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> list[dict]:
        """Store keys of every block partial of ``kind`` present on THIS machine.

        ``[{"q0", "q1"[, "r0", "r1"]}, ...]`` in a deterministic order. Business
        code needs it to discover which train-row ranges of a query block are on
        disk when the shards that produced them chose their own ranges — the one
        place a caller asks "what is there?" instead of "is THIS there?". The
        keys are opaque: nothing outside the store parses a path.
        """

    @abstractmethod
    def ensure_local(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> str:
        """Require local artifact bytes and return their filesystem path.
        """

    @abstractmethod
    def local_temporary_files(self, kind: ArtifactKind, spec: RunSpec,
                              **key: Any) -> list[str]:
        """Relative temporary files below an artifact's directory tree."""

    @abstractmethod
    def describe_external(self, relpath: str) -> dict:
        """Hash a read-only file below the configured data root."""

    @abstractmethod
    def load_external(self, relpath: str) -> Any:
        """Decode read-only JSON/NPZ evidence below the configured data root."""

    @abstractmethod
    def external_directory(self, rel_dir: str) -> str:
        """Create a directory below the local data root and return its absolute path."""

    @abstractmethod
    def list_external_tree_files(self, rel_dir: str, *, suffix: str = "") -> list[str]:
        """Return absolute file paths recursively below an external directory."""

    @abstractmethod
    def remove_external_scratch(self, rel_dir: str) -> bool:
        """Remove a task scratch directory below ``_scratch/`` if present."""
