"""Paper method registry: projected baselines and raw curvature scores.

FMAS resolves to fmas_raw (loss gradients, stratified-antithetic sampling,
layerwise damping). DAS stores its signed pre-square kernel as das_T100;
evaluators explicitly choose fitting and aggregation representations.
"""
from __future__ import annotations

import re

import numpy as np

from balds.schema.estimator import FeatureSet, ScoreMethod
from balds.schema.registry import METHODS, TRANSFORMS
from .kernel import grad_similarity, trak_kernel
from .shrinkage import ShrinkageLayer, sigma_from_repeats


class WeightedKernelMethod(ScoreMethod):
    """``transform(kernel)`` for one registered projected-Gram method name."""

    def __init__(self, name, feat_method, *, transform="identity",
                 normalize="mean_abs", row_normalize="none"):
        self.name = name
        self.feat_method = feat_method
        self.needs_error_weight = False
        self._transform = TRANSFORMS.get(transform)
        self._normalize = normalize
        self._row_normalize = row_normalize

    def score(self, feats: FeatureSet, query_grads, lam: float, *, device: str = "cuda") -> np.ndarray:
        raw = trak_kernel(feats.grads, query_grads, lam,
                          normalize=self._normalize,
                          row_normalize=self._row_normalize, device=device)
        return self._transform(raw, None, params={}).astype(np.float32)


class ShrinkageKernelMethod(ScoreMethod):
    """Projected-Gram kernel followed by the shrinkage layer ``f_c``.

    ``f_c(τ) = τ³/(τ² + γσ̂²)`` with σ̂ measured, not assumed: the R repeat-
    featurized copies carried on the :class:`FeatureSet` are scored through the
    *same* ``K⁻¹`` as the real scores, and their spread per query is the noise
    scale that sets the knee. Rebuilding the kernel per repeat would cost more
    than the scoring itself, so the repeats ride along in one kernel call.

    γ is NOT stored on the method: it is resolved per call by the app layer
    (``resolve_gamma``, per dataset AND per method) and passed in — a registered
    singleton must never carry a knee a later call could inherit. Without
    repeats the method refuses rather than silently falling back to a guessed
    σ̂: an σ̂ taken from the wrong source over-shrinks badly (0.62 → 0.28 in the
    pilot), and that failure is invisible in the output.
    """

    needs_repeats = True
    sigma_source = "repeat_features"     # σ̂ comes from `featurize --split repeat`

    def __init__(self, name, feat_method, *, normalize="mean_abs"):
        self.name = name
        self.feat_method = feat_method
        self.needs_error_weight = False
        self._normalize = normalize

    def raw_and_sigma(self, feats: FeatureSet, query_grads, lam: float, *,
                      device: str = "cuda"):
        """``(raw (N,Q) float32, sigma (1,Q))`` for one λ — the layer's inputs."""
        if feats.repeats is None or len(feats.repeats) < 2:
            raise ValueError(
                f"method '{self.name}' needs >=2 repeat featurizations to measure "
                f"sigma-hat; run `featurize --split repeat` for this identity "
                f"(and the same --proj-dim, since the noise scale depends on it)")
        raw, extra = trak_kernel(feats.grads, query_grads, lam,
                                 normalize=self._normalize, device=device,
                                 extra_features=list(feats.repeats))
        return raw, sigma_from_repeats(extra)                  # (1, Q)

    def score(self, feats: FeatureSet, query_grads, lam: float, *, device: str = "cuda",
              gamma: float | None = None) -> np.ndarray:
        if feats.repeats is None or len(feats.repeats) < 2:
            self.raw_and_sigma(feats, query_grads, lam, device=device)   # raises: no σ̂ source
        if gamma is None:
            raise ValueError(
                f"method '{self.name}' needs an explicit gamma per call: resolve it "
                f"with balds.workflows.config.resolve_gamma(cfg, dataset, '{self.name}') — "
                f"the knee is calibrated per dataset AND per method, never inherited")
        raw, sigma = self.raw_and_sigma(feats, query_grads, lam, device=device)
        return ShrinkageLayer(float(gamma)).apply(raw, sigma)


class CurvatureMethod(ScoreMethod):
    """Registry descriptor for a curvature-path method (scored by ``balds-run ekfac``).

    Carries only the declarative facts the app layer dispatches on: the raw
    method whose per-damping cache the scores come from, and which denoising
    layer (if any) sits on top — ``None`` (bare), ``"shrink"`` (``f_c``, needs
    σ̂ from ``balds-run ekfac repeats`` + a calibrated γ). ``balds-run score`` refuses.
    """

    needs_error_weight = False

    def __init__(self, name: str, raw_method: str, layer: str | None) -> None:
        self.name = name
        self.raw_method = raw_method
        self.layer = layer
        self.feat_method = None                   # no projected-feature recipe
        self.needs_repeats = layer == "shrink"
        self.sigma_source = "repeat_scores" if layer == "shrink" else None

    def score(self, feats, query_grads, lam, *, device="cuda", gamma=None):
        raise ValueError(
            f"'{self.name}' is scored by `balds-run ekfac score --method {self.name}` "
            f"(curvature path), not by `balds-run score`")




class GradSimilarityMethod(ScoreMethod):
    """Gradient (dot / cosine) baseline — raw similarity of projected loss
    gradients, no kernel inverse; ``lam`` is accepted and ignored (the λ-sweep
    then collapses to identical score matrices, which is harmless)."""

    def __init__(self, name, feat_method, *, cosine: bool):
        self.name = name
        self.feat_method = feat_method
        self.needs_error_weight = False
        self._cosine = cosine

    def score(self, feats: FeatureSet, query_grads, lam: float, *, device: str = "cuda") -> np.ndarray:
        return grad_similarity(feats.grads, query_grads, cosine=self._cosine, device=device)


#: Timestep counts every projected method is registered at, alongside the bare
#: T=10 name. Single source of truth: ``workflows.common`` imports this to build the
#: matching feature recipes, so a T can never exist as a feature but not as a
#: method. The paper's projected-family caliber is T=100 (adjudicated
#: 2026-08-16); the unsuffixed T=10 names remain for the legacy/HP-p artifacts.
FEAT_T_GRID = (100,)


class TracInMethod(ScoreMethod):
    """TracInCP / GAS — gradient similarity AVERAGED over training checkpoints.

    ``Σ_c ⟨g_i(θ_c), g_q(θ_c)⟩`` (TracInCP) or the same with both sides
    row-normalised (GAS). Matching D-TRAK's reference, checkpoints are the
    {25,50,75,100}% ones and the sum is a mean. Refuses to run without matched
    checkpoint pairs rather than silently degenerating to ``grad_dot``.
    """

    needs_checkpoints = True

    def __init__(self, name, feat_method, *, cosine: bool):
        self.name = name
        self.feat_method = feat_method
        self.needs_error_weight = False
        self._cosine = cosine

    def score(self, feats: FeatureSet, query_grads, lam: float, *, device: str = "cuda") -> np.ndarray:
        if not feats.ckpt_grads or not feats.ckpt_query:
            raise ValueError(
                f"{self.name} averages over training checkpoints and none were loaded. "
                f"Featurize the mid-training checkpoints first: "
                f"`balds-run featurize --feat {self.feat_method} --ckpt-step <N> --split both` "
                f"for each step in train.checkpoint_fracs.")
        if len(feats.ckpt_grads) != len(feats.ckpt_query):
            raise ValueError(f"{self.name}: {len(feats.ckpt_grads)} train checkpoints vs "
                             f"{len(feats.ckpt_query)} query checkpoints — pairs must match")
        total = None
        for g, q in zip(feats.ckpt_grads, feats.ckpt_query):
            s = grad_similarity(g, q, cosine=self._cosine, device=device)
            total = s if total is None else total + s
        return (total / len(feats.ckpt_grads)).astype(np.float32)


def _reg(name, feat, *, transform="identity", row_normalize="none"):
    """Register a projected method and one sibling per :data:`FEAT_T_GRID`."""
    METHODS.add(name, WeightedKernelMethod(name, feat, transform=transform,
                                           row_normalize=row_normalize))
    for _t in FEAT_T_GRID:
        METHODS.add(f"{name}_T{_t}", WeightedKernelMethod(
            f"{name}_T{_t}", f"{feat}_T{_t}", transform=transform,
            row_normalize=row_normalize))


# --- projected-Gram readout baselines (feat name = readout; see pipeline._READOUTS)
_reg("das", "das")          # mean(v) readout — the projected FMAS ancestor / DAS-style kernel
_reg("dtrak", "dtrak")      # mean(v²) readout = L_Square — the real D-TRAK
_reg("trak", "trak")        # L_Simple (mse) loss kernel — TRAK
_reg("l1norm", "l1norm")    # Σ|v|  output-function baseline (DAS's operator; DAS-archive import)
_reg("l2norm", "l2norm")    # sqrt(Σv²) output-function baseline

# --- E1 baselines on the L_Simple ("trak") features --------------------------
_reg("relative_if", "trak", row_normalize="h_inv_grad")   # ⟨g_q, K⁻¹g_i⟩ / ||K⁻¹g_i||
_reg("renorm_if", "trak", row_normalize="grad")           # ⟨g_q, K⁻¹g_i⟩ / ||g_i||
for _nm, _cos in (("grad_dot", False), ("grad_cos", True)):
    METHODS.add(_nm, GradSimilarityMethod(_nm, "trak", cosine=_cos))
    for _t in FEAT_T_GRID:
        METHODS.add(f"{_nm}_T{_t}", GradSimilarityMethod(
            f"{_nm}_T{_t}", f"trak_T{_t}", cosine=_cos))

# --- TracInCP / GAS ([f]): the same similarity averaged over checkpoints ------
for _nm, _cos in (("tracincp", False), ("gas", True)):
    METHODS.add(_nm, TracInMethod(_nm, "trak", cosine=_cos))
    for _t in FEAT_T_GRID:
        METHODS.add(f"{_nm}_T{_t}", TracInMethod(f"{_nm}_T{_t}", f"trak_T{_t}", cosine=_cos))


# --- Journey-TRAK ([e]) ------------------------------------------------------
# The bare kernel with the query side replaced by trajectory features. gen
# track only. NOTE (researcher call, 2026-08-16): the reference Journey-TRAK is
# stated over L_Simple, which a trajectory latent cannot supply; this is the
# target-free mean-readout analogue, and the table must say so.
class _JourneyKernelMethod(WeightedKernelMethod):
    query_feat_method = "journey"


for _t in FEAT_T_GRID:
    METHODS.add(f"journey_trak_T{_t}", _JourneyKernelMethod(f"journey_trak_T{_t}", f"das_T{_t}"))
METHODS.add("journey_trak", _JourneyKernelMethod("journey_trak", "das"))

# --- non-gradient baselines ([k]): raw pixels and CLIP embeddings -------------
for _nm in ("pixel", "clip"):
    METHODS.add(f"{_nm}_dot", GradSimilarityMethod(f"{_nm}_dot", _nm, cosine=False))
    METHODS.add(f"{_nm}_cos", GradSimilarityMethod(f"{_nm}_cos", _nm, cosine=True))

# --- curvature path (balds ekfac): the paper's method + the EK-FAC IF baseline ---
#: raw (bare-kernel) curvature methods -> their solver caliber. The name IS the
#: caliber: `balds-run ekfac score` binds ekfac.sampling / ekfac.damping_mode from
#: this table and refuses a conflicting --set (2026-09-03 audit §3.3).
CURVATURE_RAW: dict[str, dict] = {
    "fmas_raw": {"readout": "loss", "sampling": "stratified_antithetic", "damping_mode": "blockshrink"},
    "ekfac_if": {"readout": "loss", "sampling": "iid", "damping_mode": "global"},
}
CURVATURE_LAYERED: dict[str, tuple[str, str]] = {}
for _raw in CURVATURE_RAW:
    METHODS.add(_raw, CurvatureMethod(_raw, _raw, None))
VARIANT_TAGS: dict[str, dict] = {}
