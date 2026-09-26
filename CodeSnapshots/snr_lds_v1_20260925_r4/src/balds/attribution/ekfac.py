"""EK-FAC influence functions — the "K-FAC Influence" baseline ported to this
pipeline (Mlodozeniec et al., ICLR 2025, "Influence Functions for Scalable Data
Attribution in Diffusion Models"; reference code github.com/BrunoKM/diffusion-influence).

score(i, q) = ∇_θ m(x_q)ᵀ (F + λI)⁻¹ ∇_θ ℓ(x_i)

with m = the per-example diffusion/flow training loss on the query (the paper's
recommended "loss" measurement), ℓ = the training loss, and F the GGN^model
(MC-Fisher) of the training objective, approximated by eigenvalue-corrected
K-FAC-expand (EK-FAC) over the Conv2d/Linear layers.

Conventions ported from the reference stack (diffusion-influence + curvlinops),
so the damping value means the same thing as in the paper (default 1e-8, wide
plateau):

* Fisher MC draw: pseudo-loss ⟨v_θ(x_t, t), g_out⟩ with would-be output
  gradient g_out = η·√(2/D), η ~ N(0, I), D = per-datum output numel — the
  MSELoss(reduction="mean") convention of curvlinops, so A ⊗ B estimates the
  GGN of the **dataset-mean** loss (the 1/N is folded into the accumulators).
* K-FAC-expand: A = Σ aaᵀ / (n_draws·M), B = Σ bbᵀ / n_draws over the M
  weight-sharing positions (conv spatial patches; M=1 for plain Linear).
* separate weight/bias (curvlinops default): the weight block is A ⊗ B, the
  bias block is B alone; both get refit EK-FAC eigenvalues.
* EK-FAC eigenvalues: Λ_W = E_draws[(Q_Bᵀ G Q_A)²], Λ_b = E_draws[(Q_Bᵀ g_b)²]
  with G the per-datum parameter gradient of the same pseudo-loss — Λ therefore
  carries the only scale that matters (exact damping is (Λ + λ)⁻¹).
* Timestep sampling matches training: t ~ U(0, 1) through the process seam
  (cfm uses float t directly, ddpm rounds to its index grid — exactly what
  ``generative/train.py`` does), with random horizontal flip on the train side
  when the recipe uses it.

Everything here is pure compute (no file IO — store layer owns persistence).
All Monte-Carlo noise is drawn on the CPU from dedicated, explicitly seeded
``torch.Generator``s keyed by (sample, draw), so results do not depend on batch
size or execution order. The one uncontrolled source is dropout when computing
train-side gradients in train mode (reference parity), which uses the global
GPU RNG.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from balds.schema.generative import GenerativeProcess
from balds.schema.identity import MCReplicaKey

# Slice per-sample gradient tensors so no transient exceeds this many elements
# (biggest UNet layer is ~512x4608; a full batch of those at fp32 is GBs).
_EINSUM_BUDGET = 200_000_000


# --------------------------------------------------------------------------- #
# target modules
# --------------------------------------------------------------------------- #

def kfac_target_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Ordered ``name -> module`` of the layers EK-FAC covers.

    Conv2d and Linear only, matching the reference ("we compute (E)K-FAC for
    nn.Conv2d and nn.Linear modules, ignoring the parameters in the
    normalisation layers"). The influence inner product runs over exactly these
    parameters too — GroupNorm affine terms and any class embedding are outside
    the estimator, not merely outside the curvature.
    """
    out: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and any(
            p.requires_grad for p in mod.parameters(recurse=False)
        ):
            if isinstance(mod, nn.Conv2d) and mod.groups != 1:
                raise NotImplementedError(f"grouped conv not supported: {name}")
            out[name] = mod
    return out


def target_parameters(modules: dict[str, nn.Module]) -> list[nn.Parameter]:
    """Flat parameter list in (layer, weight-then-bias) order."""
    params: list[nn.Parameter] = []
    for mod in modules.values():
        params.append(mod.weight)
        if mod.bias is not None:
            params.append(mod.bias)
    return params


def _in_features(mod: nn.Module) -> int:
    if isinstance(mod, nn.Linear):
        return mod.in_features
    k = mod.kernel_size
    return mod.in_channels * k[0] * k[1]


def _out_features(mod: nn.Module) -> int:
    return mod.out_features if isinstance(mod, nn.Linear) else mod.out_channels


def _input_to_positions(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Layer input -> ``(B, M, d_in)`` weight-sharing (expand) format."""
    if isinstance(mod, nn.Conv2d):
        patches = F.unfold(x, mod.kernel_size, dilation=mod.dilation,
                           padding=mod.padding, stride=mod.stride)   # (B, d_in, M)
        return patches.transpose(1, 2).contiguous()
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x.reshape(x.shape[0], -1, x.shape[-1])


def _grad_to_positions(mod: nn.Module, g: torch.Tensor) -> torch.Tensor:
    """Grad w.r.t. layer output -> ``(B, M, d_out)`` expand format."""
    if isinstance(mod, nn.Conv2d):
        return g.flatten(2).transpose(1, 2).contiguous()             # (B, HW, C_out)
    if g.dim() == 2:
        return g.unsqueeze(1)
    return g.reshape(g.shape[0], -1, g.shape[-1])


# --------------------------------------------------------------------------- #
# factors container
# --------------------------------------------------------------------------- #

class EkfacFactors:
    """Per-layer EK-FAC eigenbases + corrected eigenvalues (+ fitting metadata).

    ``layers[name] = {"Q_A": (d_in, d_in), "Q_B": (d_out, d_out),
    "lam_w": (d_out, d_in), "lam_b": (d_out,) | None}`` — ``lam_*`` are the
    corrected eigenvalues of the mean-loss Fisher, so the damped inverse is
    ``(lam + damping)⁻¹`` applied in the eigenbasis (curvlinops "exact damping").
    """

    def __init__(self, layers: dict[str, dict], meta: dict) -> None:
        self.layers = layers
        self.meta = meta

    # -- (de)serialization through the store's pt codec ---------------------
    def state_dict(self) -> dict:
        return {"layers": self.layers, "meta": self.meta}

    @classmethod
    def from_state_dict(cls, state: dict) -> "EkfacFactors":
        return cls(state["layers"], state["meta"])

    # -- shape bookkeeping --------------------------------------------------
    def layer_order(self) -> list[str]:
        return self.meta["layer_order"]

    def num_params(self) -> int:
        total = 0
        for name in self.layer_order():
            ly = self.layers[name]
            total += ly["lam_w"].numel()
            if ly["lam_b"] is not None:
                total += ly["lam_b"].numel()
        return total

    def assert_matches(self, modules: dict[str, nn.Module]) -> None:
        if list(modules) != self.layer_order():
            raise ValueError("EK-FAC factors do not match this model's layer list — "
                             "they were fit for a different architecture/identity")
        for name, mod in modules.items():
            ly = self.layers[name]
            if ly["lam_w"].shape != (_out_features(mod), _in_features(mod)):
                raise ValueError(f"factor shape mismatch at {name}")
            if (mod.bias is not None) != (ly["lam_b"] is not None):
                raise ValueError(f"bias presence mismatch at {name}")

    def to(self, device: str) -> "EkfacFactors":
        self.layers = {
            n: {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ly.items()}
            for n, ly in self.layers.items()
        }
        return self

    def damped_inverse_flat(self, damping: float, device: str,
                            mode: str = "global") -> torch.Tensor:
        """``1 / (Λ + δ_l)`` flattened in the canonical (layer, W-then-b) order.

        ``mode="global"``: δ_l = ``damping`` for every layer (reference EK-FAC).
        ``mode="blockshrink"``: per-layer Ledoit–Wolf-style shrinkage toward a
        scaled identity, δ_l = ``damping`` · mean(Λ_l^W) — ``damping`` is then the
        dimensionless shrinkage intensity ϱ, and each block is damped on its own
        spectral scale (SOLVER_RESEARCH.md #1; arXiv 2410.22568 §K-FAC). The bias
        block of a layer uses its weight block's δ_l.
        """
        if mode not in DAMPING_MODES:
            raise ValueError(f"unknown damping mode {mode!r}; choose from {DAMPING_MODES}")
        parts = []
        for name in self.layer_order():
            ly = self.layers[name]
            lam_w = ly["lam_w"].to(device).flatten()
            delta = damping if mode == "global" else damping * float(lam_w.mean())
            if mode == "blockshrink" and delta == 0.0:
                # Dead block: Λ_l ≡ 0 means no draw ever produced a gradient here
                # (e.g. SD3's last joint block has a context q-projection that
                # nothing downstream reads, so its LoRA pair gets zero curvature).
                # ϱ·mean(Λ_l) = 0 would make 1/(0+0) = inf and inf·0 = NaN, which
                # then poisons every score through the sum over blocks
                # (ArtBench-2 gen track 2026-09-07: 500000/500000 NaN). The block's
                # gradients are identically zero, so any finite inverse gives a
                # zero contribution; use zeros. Blocks with δ_l > 0 are untouched.
                parts.append(torch.zeros_like(lam_w))
                if ly["lam_b"] is not None:
                    parts.append(torch.zeros_like(ly["lam_b"].to(device)))
                continue
            parts.append((lam_w + delta).reciprocal())
            if ly["lam_b"] is not None:
                parts.append((ly["lam_b"].to(device) + delta).reciprocal())
        return torch.cat(parts)


# --------------------------------------------------------------------------- #
# MC draws for the diffusion/flow objective
# --------------------------------------------------------------------------- #

def _draw_batch(images: torch.Tensor, process: GenerativeProcess, gen: torch.Generator,
                *, hflip: bool, device: str, with_target: bool,
                flipped: Optional[torch.Tensor] = None):
    """One (flip, t, ε[, target]) draw per element of ``images`` — CPU RNG only.

    Returns ``(x_t, t_vec, target | None)`` on ``device``. Uses the process seam
    for interpolation/target, so cfm and ddpm both follow their training-time
    conventions (t ~ U(0,1); ddpm rounds internally).

    ``flipped`` supplies the pre-encoded flipped orientation per element for the
    latent platform, where a spatial ``.flip(-1)`` of a latent is NOT the latent
    of the flipped image; ``None`` keeps the pixel-space flip (bit-identical to
    before the parameter existed — the flip Bernoulli draw order is unchanged).
    """
    B = images.shape[0]
    if hflip:
        flip = torch.rand(B, generator=gen) < 0.5
        alt = images.flip(-1) if flipped is None else flipped
        x1 = torch.where(flip.view(-1, 1, 1, 1), alt, images)
    else:
        x1 = images
    t = torch.rand(B, generator=gen)
    eps = torch.randn(images.shape, generator=gen)
    x1, t, eps = x1.to(device), t.to(device), eps.to(device)
    x_t = process.interpolate(x1, eps, t)
    target = process.target(x1, eps, t) if with_target else None
    return x_t, t, target


#: Damping modes for :meth:`EkfacFactors.damped_inverse_flat`.
DAMPING_MODES = ("global", "blockshrink")

#: Sampling schemes for :meth:`EkfacScorer.sample_gradient` (query and train
#: side alike). ``"iid"`` is the reference implementation's stream.
SAMPLINGS = ("iid", "stratified_antithetic")


def _stratified_antithetic_draws(shape, mc: int, gen: torch.Generator, *, hflip: bool):
    """``(flip[mc], t[mc], eps[mc, *shape])`` on CPU: ``t_i = (i + U_i)/mc`` (one
    draw per stratum) and ``eps`` in antithetic pairs ``(η, −η)``. Both keep the
    estimator unbiased (each t_i is still marginally U(0,1); η and −η have the
    same law) while cancelling the odd part of the integrand across pairs and
    the between-stratum variance of t."""
    flip = (torch.rand(mc, generator=gen) < 0.5) if hflip else torch.zeros(mc, dtype=torch.bool)
    t = (torch.arange(mc, dtype=torch.float32) + torch.rand(mc, generator=gen)) / mc
    half = (mc + 1) // 2
    eta = torch.randn((half, *shape), generator=gen)
    eps = torch.stack([eta, -eta], dim=1).reshape(2 * half, *shape)[:mc]
    return flip, t, eps


def _materialize(images: torch.Tensor, process: GenerativeProcess, flip, t, eps, *,
                 device: str, with_target: bool,
                 flipped: Optional[torch.Tensor] = None):
    """Same contract as :func:`_draw_batch` but from pre-drawn (flip, t, ε)."""
    alt = images.flip(-1) if flipped is None else flipped
    x1 = torch.where(flip.view(-1, 1, 1, 1).to(images.device), alt, images)
    x1, t, eps = x1.to(device), t.to(device), eps.to(device)
    x_t = process.interpolate(x1, eps, t)
    target = process.target(x1, eps, t) if with_target else None
    return x_t, t, target


def _pseudo_grad_output(shape, gen: torch.Generator, device: str) -> torch.Tensor:
    """Would-be gradient w.r.t. the model output for the MC-Fisher.

    ``η·√(2/D)`` per datum (η ~ N(0, I), D the per-datum output numel) — the
    curvlinops MSELoss(reduction="mean") sampling convention, so the expected
    outer product is the per-datum loss Hessian ``(2/D)·I``.
    """
    D = int(torch.tensor(shape[1:]).prod())
    eta = torch.randn(shape, generator=gen)
    return (eta * math.sqrt(2.0 / D)).to(device)


#: Per-draw scalar readouts for :meth:`EkfacScorer.sample_gradient` (``(k, ...)``
#: output -> ``(k,)``). ``"loss"`` is handled inline (needs the target).
READOUTS = {
    "loss": None,
    "mean": lambda pred: pred.flatten(1).mean(dim=1),          # featurize.f_mean
    "msl2": lambda pred: pred.flatten(1).pow(2).mean(dim=1),   # featurize.f_msl2 (L_Square)
}


def _mc_generator(seed: int, phase: str, index: int) -> torch.Generator:
    """Deterministic per-(phase, logical-unit) CPU generator.

    Keyed by a stable hash of the phase name so factor/eigenvalue/score draws
    never collide; independent of batch size and execution order.
    """
    tag = sum(ord(c) * 1009 ** i for i, c in enumerate(phase)) % 1_000_003
    g = torch.Generator()
    g.manual_seed(((seed * 1_000_003 + tag) * 1_000_003 + index) % (2 ** 64))
    return g


def _replica_generator(
    seed: int, phase: str, index: int, replica_key: MCReplicaKey | None,
) -> torch.Generator:
    """Return the legacy stream or an explicit domain-separated pilot stream."""
    if replica_key is None:
        return _mc_generator(seed, phase, index)
    expected_axis = "query" if phase == "ekfac_meas" else "train"
    if replica_key.axis != expected_axis:
        raise ValueError(
            f"replica key axis {replica_key.axis!r} conflicts with phase {phase!r}; "
            f"expected {expected_axis!r}"
        )
    if replica_key.logical_index != index:
        raise ValueError(
            f"replica key global index {replica_key.logical_index} != call index {index}"
        )
    return torch.Generator().manual_seed(replica_key.torch_seed())


# --------------------------------------------------------------------------- #
# fitting engine (hooks)
# --------------------------------------------------------------------------- #

class _HookState:
    """Per-backward bookkeeping shared by the two fitting phases."""

    def __init__(self) -> None:
        self.inputs: dict[str, torch.Tensor] = {}


class EkfacFitter:
    """Two-pass EK-FAC fit: (1) covariance factors -> eigenbases, (2) corrected
    eigenvalues in that basis. Both passes stream the training set with fresh
    MC draws; accumulation is float64 for numerical safety."""

    def __init__(self, model: nn.Module, process: GenerativeProcess, *,
                 device: str = "cuda", curvature_kind: str = "ggn_model") -> None:
        if curvature_kind not in {"ggn_model", "empirical_fisher"}:
            raise ValueError("curvature_kind must be ggn_model or empirical_fisher")
        self.model = model
        self.process = process
        self.device = device
        self.modules = kfac_target_modules(model)
        self.curvature_kind = curvature_kind

    # ---- pass 1: covariance factors --------------------------------------
    def fit_factor_pass(self, images: torch.Tensor, labels: Optional[torch.Tensor], *,
                        epochs: int, batch_size: int, seed: int, hflip: bool,
                        flipped: Optional[torch.Tensor] = None,
                        on_progress=None) -> tuple[dict, dict, int]:
        """Accumulate A/B covariances over ``epochs`` MC draws per image.

        Returns ``(A, B, n_draws)`` with A[name] (d_in, d_in), B[name]
        (d_out, d_out) as float64 CPU tensors, already normalized (A by
        draws·M, B by draws). Accumulators live on the GPU in float64 (~2 GB
        for the 36M UNet) — shipping per-batch partial sums to the CPU would
        dominate wall-clock — and move to the CPU once at the end.
        """
        A = {n: torch.zeros(_in_features(m), _in_features(m), dtype=torch.float64,
                            device=self.device) for n, m in self.modules.items()}
        Bcov = {n: torch.zeros(_out_features(m), _out_features(m), dtype=torch.float64,
                               device=self.device) for n, m in self.modules.items()}
        n_pos = {n: 0 for n in self.modules}
        state = _HookState()
        handles = self._install_hooks(state, mode="factors", A=A, B=Bcov, n_pos=n_pos)
        self.model.eval()
        N = images.shape[0]
        total = epochs * math.ceil(N / batch_size)
        done = 0
        try:
            for e in range(epochs):
                for start in range(0, N, batch_size):
                    stop = min(start + batch_size, N)
                    gen = _mc_generator(seed, "ekfac_factors", e * N + start)
                    self._pseudo_backward(images[start:stop],
                                          None if labels is None else labels[start:stop],
                                          gen, hflip=hflip,
                                          flipped=None if flipped is None else flipped[start:stop])
                    done += 1
                    if on_progress is not None and (done % 50 == 0 or done == total):
                        on_progress(done=done, total=total, phase="factors")
        finally:
            for h in handles:
                h.remove()
        n_draws = epochs * N
        for n in self.modules:
            M = max(1, n_pos[n] // n_draws)          # positions per draw
            A[n] = (A[n] / float(n_draws * M)).cpu()
            Bcov[n] = (Bcov[n] / float(n_draws)).cpu()
        return A, Bcov, n_draws

    @staticmethod
    def eigenbases(A: dict, Bcov: dict) -> dict[str, dict]:
        """Eigendecompose the (symmetric PSD) factors -> per-layer Q_A / Q_B."""
        out: dict[str, dict] = {}
        for name in A:
            _, QA = torch.linalg.eigh(A[name])
            _, QB = torch.linalg.eigh(Bcov[name])
            out[name] = {"Q_A": QA.to(torch.float32), "Q_B": QB.to(torch.float32)}
        return out

    # ---- pass 2: corrected eigenvalues ------------------------------------
    def fit_eigenvalue_pass(self, images: torch.Tensor, labels: Optional[torch.Tensor],
                            bases: dict[str, dict], *, epochs: int, batch_size: int,
                            seed: int, hflip: bool,
                            flipped: Optional[torch.Tensor] = None,
                            on_progress=None) -> dict[str, dict]:
        """Refit the diagonal in the Kronecker eigenbasis (George et al., 2018).

        Λ_W = E[(Q_Bᵀ G Q_A)²], Λ_b = E[(Q_Bᵀ g_b)²] over fresh MC draws — the
        expectation of the squared eigen-coordinates of the per-datum pseudo-loss
        gradient, i.e. the exact diagonal of the MC-Fisher in that basis.
        """
        dev = self.device
        QA = {n: bases[n]["Q_A"].to(dev) for n in self.modules}
        QB = {n: bases[n]["Q_B"].to(dev) for n in self.modules}
        lam_w = {n: torch.zeros(_out_features(m), _in_features(m), dtype=torch.float64,
                                device=dev) for n, m in self.modules.items()}
        lam_b = {n: (torch.zeros(_out_features(m), dtype=torch.float64, device=dev)
                     if m.bias is not None else None)
                 for n, m in self.modules.items()}
        state = _HookState()
        handles = self._install_hooks(state, mode="eig", QA=QA, QB=QB,
                                      lam_w=lam_w, lam_b=lam_b)
        self.model.eval()
        N = images.shape[0]
        total = epochs * math.ceil(N / batch_size)
        done = 0
        try:
            for e in range(epochs):
                for start in range(0, N, batch_size):
                    stop = min(start + batch_size, N)
                    gen = _mc_generator(seed, "ekfac_eig", e * N + start)
                    self._pseudo_backward(images[start:stop],
                                          None if labels is None else labels[start:stop],
                                          gen, hflip=hflip,
                                          flipped=None if flipped is None else flipped[start:stop])
                    state.inputs.clear()
                    done += 1
                    if on_progress is not None and (done % 50 == 0 or done == total):
                        on_progress(done=done, total=total, phase="eigenvalues")
        finally:
            for h in handles:
                h.remove()
        n_draws = float(epochs * N)
        out: dict[str, dict] = {}
        for n in self.modules:
            out[n] = {
                "Q_A": bases[n]["Q_A"].cpu(),
                "Q_B": bases[n]["Q_B"].cpu(),
                # squares are >= 0; clamp guards eigh round-off only
                "lam_w": (lam_w[n] / n_draws).clamp_min_(0).to(torch.float32).cpu(),
                "lam_b": (None if lam_b[n] is None
                          else (lam_b[n] / n_draws).clamp_min_(0).to(torch.float32).cpu()),
            }
        return out

    # ---- shared internals -------------------------------------------------
    def _pseudo_backward(self, images: torch.Tensor, labels: Optional[torch.Tensor],
                         gen: torch.Generator, *, hflip: bool,
                         flipped: Optional[torch.Tensor] = None) -> None:
        """Forward + backward of the MC-Fisher pseudo-loss for one batch."""
        empirical = self.curvature_kind == "empirical_fisher"
        x_t, t, target = _draw_batch(images, self.process, gen, hflip=hflip,
                                    device=self.device, with_target=empirical, flipped=flipped)
        lb = None if labels is None else labels.to(self.device)
        v = self.model(x_t, t, lb)
        if empirical:
            # Sum independent per-datum mean MSEs, NOT a batch-mean gradient.
            # Hooks square per-datum gradients; both passes divide by n_draws.
            loss = (v - target).square().flatten(1).mean(1).sum()
        else:
            g_out = _pseudo_grad_output(v.shape, gen, self.device)
            loss = (v * g_out).sum()
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        self.model.zero_grad(set_to_none=True)

    def _install_hooks(self, state: _HookState, *, mode: str, **acc) -> list:
        handles = []
        for name, mod in self.modules.items():
            handles.append(mod.register_forward_hook(self._fwd_hook(name, state, mode, acc)))
        return handles

    def _fwd_hook(self, name: str, state: _HookState, mode: str, acc: dict):
        mod = self.modules[name]

        def hook(module, args, output):
            x = args[0]
            if mode == "factors":
                with torch.no_grad():
                    a = _input_to_positions(mod, x.detach())
                    acc["A"][name] += torch.einsum("bmi,bmj->ij", a, a).double()
                    acc["n_pos"][name] += a.shape[0] * a.shape[1]
            else:
                state.inputs[name] = x.detach()

            def bwd(grad):
                with torch.no_grad():
                    b = _grad_to_positions(mod, grad.detach())
                    if mode == "factors":
                        acc["B"][name] += torch.einsum("bmi,bmj->ij", b, b).double()
                    else:
                        self._accumulate_eig(name, state.inputs.pop(name), b, acc)
                return grad

            if output.requires_grad:
                output.register_hook(bwd)
            return None

        return hook

    def _accumulate_eig(self, name: str, x: torch.Tensor, b: torch.Tensor, acc: dict) -> None:
        mod = self.modules[name]
        a = _input_to_positions(mod, x)
        QA, QB = acc["QA"][name], acc["QB"][name]
        B = a.shape[0]
        di, do = a.shape[-1], b.shape[-1]
        step = max(1, _EINSUM_BUDGET // max(1, do * di))
        for s in range(0, B, step):
            G = torch.einsum("bmo,bmi->boi", b[s:s + step], a[s:s + step])  # per-datum grads
            rot = QB.T @ G @ QA
            acc["lam_w"][name] += rot.square().sum(dim=0).double()
        if acc["lam_b"][name] is not None:
            gb = b.sum(dim=1)                                    # (B, d_out)
            acc["lam_b"][name] += (gb @ QB).square().sum(dim=0).double()


# --------------------------------------------------------------------------- #
# per-sample gradients and eigen-coordinates (scoring side)
# --------------------------------------------------------------------------- #

class EkfacScorer:
    """Per-sample MC gradients of the training loss / measurement, rotated into
    the EK-FAC eigenbasis for damped inner products.

    score(i, q) at damping λ decomposes as ⟨u_i, u_q / (Λ + λ)⟩ with u the
    eigen-coordinates — so one streamed train pass supports a whole damping
    grid at the cost of an elementwise scale + matvec per value.
    """

    def __init__(self, model: nn.Module, process: GenerativeProcess,
                 factors: EkfacFactors, *, device: str = "cuda") -> None:
        self.model = model
        self.process = process
        self.device = device
        self.modules = kfac_target_modules(model)
        factors.assert_matches(self.modules)
        self.factors = factors.to(device)
        self.params = target_parameters(self.modules)

    def sample_gradient(self, image: torch.Tensor, label, *, mc: int, chunk: int,
                        seed: int, phase: str, index: int, hflip: bool,
                        train_mode: bool, readout: str = "loss",
                        sampling: str = "iid",
                        flipped: Optional[torch.Tensor] = None,
                        replica_key: MCReplicaKey | None = None) -> list[torch.Tensor]:
        """MC-averaged gradient of a per-example readout, per target parameter.

        One image, ``mc`` (flip, t, ε) draws in chunks of ``chunk``; the value
        differentiated is the mean over draws of the per-draw readout:

        * ``"loss"`` — per-draw (output-mean) MSE against the flow/diffusion
          target: the reference's measurement / train-loss convention (EK-FAC IF).
        * ``"mean"`` — per-draw spatial mean of the model output, ``mean(v)``:
          the FMAS readout of ``featurize.py`` (``f_mean``) under the curvature
          preconditioner instead of the projected Gram kernel (X1, "FMAS-EK").
        * ``"msl2"`` — per-draw ``mean(v²)``: the D-TRAK L_Square readout under
          the same preconditioner ("D-TRAK-EK").

        No per-draw normalisation on any readout (the projected featurizer
        L2-normalises per time point; here magnitudes are kept, as in the
        reference). ``train_mode`` enables dropout (reference computes
        train-side gradients in train mode, query side eval).
        """
        if readout not in READOUTS:
            raise ValueError(f"unknown ekfac readout {readout!r}; choose from {sorted(READOUTS)}")
        if sampling not in SAMPLINGS:
            raise ValueError(f"unknown ekfac sampling {sampling!r}; choose from {SAMPLINGS}")
        was_training = self.model.training
        self.model.train(train_mode)
        gen = _replica_generator(seed, phase, index, replica_key)
        total = [torch.zeros_like(p) for p in self.params]
        img = image.unsqueeze(0)
        # Dropout (train_mode) draws from the GLOBAL torch RNG, which the
        # per-sample CPU generators do not cover. Fork it and seed it from the
        # same (seed, phase, index) key so a sample's gradient is a pure function
        # of that key — the property the repeat-scored σ̂ path (task brief D3)
        # and any rerun rely on. fork_rng restores the caller's RNG state.
        devices = [torch.device(self.device)] if str(self.device).startswith("cuda") \
            and torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(gen.initial_seed()) % (2 ** 63 - 1))
                self._accumulate(total, img, gen, mc=mc, chunk=chunk, label=label,
                                 hflip=hflip, readout=readout, sampling=sampling,
                                 flipped=flipped)
        finally:
            self.model.train(was_training)
        return total

    def _accumulate(self, total, img, gen, *, mc, chunk, label, hflip, readout="loss",
                    sampling="iid", flipped=None):
        pre = None
        if sampling == "stratified_antithetic":
            # Zero-cost variance reduction (ANALYSIS §9.4-1): draw i takes t from
            # stratum [i/mc, (i+1)/mc) and ε in antithetic pairs (ε, −ε). Pre-drawn
            # once per sample so chunking cannot change the estimator.
            pre = _stratified_antithetic_draws(img.shape[1:], mc, gen, hflip=hflip)
        for start in range(0, mc, chunk):
            k = min(chunk, mc - start)
            batch = img.expand(k, -1, -1, -1)
            fbatch = None if flipped is None else \
                flipped.unsqueeze(0).expand(k, -1, -1, -1)
            if pre is None:
                # The draw sequence (flip, t, ε) is identical for every readout, so
                # the three readouts of one (seed, phase, index) see the same MC
                # points; the target is drawn regardless to keep that stream fixed.
                x_t, t, target = _draw_batch(batch, self.process, gen, hflip=hflip,
                                             device=self.device, with_target=True,
                                             flipped=fbatch)
            else:
                flip, tt, eps = (a[start:start + k] for a in pre)
                x_t, t, target = _materialize(batch, self.process, flip, tt, eps,
                                              device=self.device, with_target=True,
                                              flipped=fbatch)
            lb = None
            if label is not None:
                lb = torch.full((k,), int(label), dtype=torch.long, device=self.device)
            pred = self.model(x_t, t, lb)
            if readout == "loss":
                loss = F.mse_loss(pred, target) * (k / mc)          # unchanged EK-FAC IF path
            else:
                # mean over draws of the per-draw readout, same (k/mc) chunk weighting
                loss = READOUTS[readout](pred).mean() * (k / mc)
            grads = torch.autograd.grad(loss, self.params)
            for tot, g in zip(total, grads):
                tot += g.detach()

    def to_eigencoords(self, grads: list[torch.Tensor]) -> torch.Tensor:
        """Rotate per-parameter gradients into the eigenbasis; flatten (P,)."""
        parts = []
        i = 0
        for name in self.factors.layer_order():
            mod = self.modules[name]
            ly = self.factors.layers[name]
            G = grads[i].reshape(_out_features(mod), _in_features(mod))
            parts.append((ly["Q_B"].T @ G @ ly["Q_A"]).flatten())
            i += 1
            if mod.bias is not None:
                parts.append(ly["Q_B"].T @ grads[i])
                i += 1
        return torch.cat(parts)
