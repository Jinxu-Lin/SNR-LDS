"""Per-sample projected-gradient featurization (ports the legacy ``DTrakFM``).

For each sample, over a grid of timesteps: take the gradient of the
direct-output functional w.r.t. parameters (``vmap(grad(...))``), L2-normalize,
(time-weighted) average across timesteps, and random-project to ``proj_dim``.

The featurizer is parameterised by three protocol axes so the vertical axis can
vary them (aggregation over *structural* protocols, not just projection):

* ``t_set`` (𝒯) — the actual timestep values (not just the count);
* ``time_weight`` (w) — per-timestep weight in the average (uniform/transport/linear);
* ``output_mask`` (Λ) — which output region enters the direct-output functional.

The **default** (``t_set=None`` → ``arange(T)/T``, ``uniform``, ``full``) is
numerically identical to the legacy featurizer (parity-preserving).

The ``CudaProjector`` (fast_jl) must be constructed before any vmap/cuDNN call —
so the projector is created in ``__init__``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from balds.schema.generative import GenerativeProcess

# Time-weight functions w(t) (uniform reproduces the legacy uniform average).
_WEIGHT_FNS = {
    "uniform": lambda t: 1.0,
    "transport": lambda t: (1.0 - t) ** 2,
    "linear": lambda t: 1.0 - t,
}


def _random_draw(seed: int, draw: int, start: int, batch_imgs: torch.Tensor):
    """``(t_vec, noise)`` for one draw of a batch — per-sample CPU generators
    keyed by ``(seed, draw, global sample index)`` (dispatcher rule: randomness
    belongs to the logical unit, never to batch size / loop order)."""
    bsz = batch_imgs.shape[0]
    t_vec = torch.empty(bsz)
    noise = torch.empty(batch_imgs.shape)
    for j in range(bsz):
        g = torch.Generator().manual_seed(((seed * 1000 + draw) * 1_000_003 + start + j)
                                          % (2 ** 63 - 1))
        t_vec[j] = torch.rand(1, generator=g)
        noise[j] = torch.randn(batch_imgs.shape[1:], generator=g)
    return (t_vec.to(device=batch_imgs.device, dtype=batch_imgs.dtype),
            noise.to(device=batch_imgs.device, dtype=batch_imgs.dtype))


def _accumulate(grad_box: list, bsz: int, *, normalize: bool, w_t: float, emb):
    """Flatten one timestep's per-sample gradients into ONE fp32 ``(B, P)``
    buffer and accumulate it into ``emb``, in place. Returns the new ``emb``.

    Shared by :meth:`GradFeaturizer.featurize` and
    :meth:`GradFeaturizer.featurize_states` so the ordinary and the trajectory
    featurizer cannot drift into computing different gradients (or, once, into
    one of them being the only one that fits in 24 GB).

    Memory contract, and why the grads arrive in a one-element LIST rather than
    directly: on the SD3.5+LoRA platform P = 95,551,488, so one ``(B, P)`` fp32
    buffer is 1.42 GiB at B=4 against a card whose whole idle capacity is
    24,210 MiB. The list is popped and deleted here, which makes this the ONLY
    reference to the grad dict and frees it before the accumulation runs — with
    a plain argument the caller's own name would keep it alive and the loop
    would hold three live buffers instead of two.

    Numerics (TASK_FEATURIZE_MEM 2026-09-06, unchanged here):
    * pre-allocating and ``copy_``-ing each parameter block is the same exact
      upcast ``torch.cat(...).float()`` does (bf16/fp16 -> fp32 is lossless) and
      the blocks land in the same order, so the result is bit-identical;
    * ``div_``/``mul_``/``add_`` are the same element-wise ops in the same order
      with the same rounding, they only skip the output buffer. Verified
      old-vs-new on CPU across seven recipe variants plus the production
      cifar2_5k model (P=35.7M, T=100, B=16, ``torch.equal`` on the full emb),
      and on CUDA under ``torch.use_deterministic_algorithms`` (max|delta| = 0).
    * Do NOT fold the scale into ``emb.add_(flat, alpha=w_t)``: the fused form
      may contract to an FMA and change the last bit.
    """
    per_grads = grad_box.pop()
    _P = sum(v.numel() // bsz for v in per_grads.values())
    flat = torch.empty(bsz, _P, dtype=torch.float32,
                       device=next(iter(per_grads.values())).device)
    _off = 0
    for v in per_grads.values():
        _n = v.numel() // bsz
        flat[:, _off:_off + _n].copy_(v.reshape(bsz, _n))
        _off += _n
    del per_grads, v
    if normalize:
        flat.div_(torch.norm(flat, dim=-1, keepdim=True) + 1e-8)
    flat.mul_(w_t)
    # first pass takes ownership of flat; later passes accumulate into emb and
    # let flat die at the next iteration's allocation
    return flat if emb is None else emb.add_(flat)


def _build_output_mask(name: str, shape, device) -> torch.Tensor:
    """Flat 0/1 output mask over ``(C, H, W)``; ``full`` selects everything."""
    c, h, w = int(shape[0]), int(shape[1]), int(shape[2])
    if name == "full":
        m = torch.ones(c, h, w)
    elif name == "center":                     # central 16x16 spatial region
        m = torch.zeros(c, h, w); m[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 1.0
    elif name == "border":                     # complementary outer border
        m = torch.ones(c, h, w); m[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.0
    else:
        raise ValueError(f"unknown output_mask '{name}'")
    return m.reshape(-1).to(device)


class GradFeaturizer:
    """D-TRAK / TRAK featurizer for a flow/diffusion model.

    ``loss_type="mean"`` is D-TRAK (functional = masked ``v.mean()``, Q = I);
    ``loss_type="mse"`` is TRAK (the ``(v - target)`` residual enters the gradient).
    """

    def __init__(self, model, process: GenerativeProcess, *, proj_dim: int = 4096,
                 proj_seed: int = 0, T: int = 10, t_set: Optional[Sequence[float]] = None,
                 time_weight: str = "uniform", output_mask: str = "full",
                 batch_size: int = 16, loss_type: str = "mean", device: str = "cuda",
                 projection: str = "auto", normalize: bool = True,
                 t_sampling: str = "grid") -> None:
        """``normalize`` — L2-normalise each time point's gradient before the
        average (legacy/D-TRAK convention; keeps every t at equal magnitude but
        erases ``|r_t|``, the only magnitude a loss gradient carries).
        ``t_sampling`` — ``"grid"``: the fixed ``time_grid(T)`` with one noise per
        point (legacy); ``"random"``: ``T`` independent ``(t, ε)`` draws per sample,
        ``t ~ U(0,1)``, seeded per (seed, draw, sample index) so the result does not
        depend on batch size or order. X2 (_Data/reports/fmas_vs_ekfac_2026-08-27) uses
        ``normalize=False, t_sampling="random"`` on the L_Simple readout."""
        self.model = model
        self.process = process
        self.device = "cuda:0" if device == "cuda" else device  # set_device needs an index
        self.proj_dim = proj_dim
        self.t_set = [float(t) for t in t_set] if t_set is not None else None
        self.T = len(self.t_set) if self.t_set is not None else T
        self.time_weight = time_weight
        self.output_mask = output_mask
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if time_weight not in _WEIGHT_FNS:
            raise ValueError(f"unknown time_weight '{time_weight}'")
        if t_sampling not in ("grid", "random"):
            raise ValueError(f"unknown t_sampling '{t_sampling}' (grid|random)")
        if t_sampling == "random" and time_weight != "uniform":
            raise ValueError("t_sampling='random' draws t ~ U(0,1); a non-uniform "
                             "time_weight is a different (importance-weighted) estimator")
        self.normalize = bool(normalize)
        self.t_sampling = t_sampling
        # construct the projector FIRST (CUDA-stream ordering requirement)
        from .projection import build_projector
        self._proj, self.projector_kind = build_projector(
            projection, grad_dim=self.n_params, proj_dim=proj_dim, seed=proj_seed,
            batch_size=batch_size, device=device,
        )

    @torch.no_grad()
    def _timesteps(self):
        if self.t_set is not None:
            return self.t_set
        return [float(t) for t in self.process.time_grid(self.T, "featurize")]

    def featurize(self, dataset, *, seed: int = 42, max_samples: Optional[int] = None,
                  log_every: int = 500, on_progress=None) -> torch.Tensor:
        """Return ``(N, proj_dim)`` CPU float32 projected gradient features.

        ``on_progress(done=..., total=...)``, if given, is called at the same
        cadence as the stdout counter plus once at completion (no IO here)."""
        import torch.utils.data as td

        N = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        params, buffers, grad_fn = self._grad_fn(dataset.images[0].shape)
        w_fn = _WEIGHT_FNS[self.time_weight]

        images = torch.stack([dataset.images[i] for i in range(N)])
        labels = torch.tensor([
            dataset.labels[i].item() if isinstance(dataset.labels[i], torch.Tensor)
            else int(dataset.labels[i]) for i in range(N)
        ], dtype=torch.long)
        loader = td.DataLoader(td.TensorDataset(images, labels),
                               batch_size=self.batch_size, shuffle=False)

        timesteps = self._timesteps() if self.t_sampling == "grid" else [None] * self.T
        features = torch.zeros(N, self.proj_dim, dtype=torch.float32)
        start = 0
        for batch_imgs, batch_labels in loader:
            bsz = batch_imgs.shape[0]
            batch_imgs = batch_imgs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            emb = None
            w_sum = 0.0
            for t_idx, t_val in enumerate(timesteps):
                if self.t_sampling == "grid":
                    torch.manual_seed(seed * 1000 + t_idx)
                    torch.cuda.manual_seed_all(seed * 1000 + t_idx)
                    noise = torch.randn_like(batch_imgs)
                    # process-owned interpolation/target (float t keeps the CFM path
                    # bit-identical to the legacy inline expression; DDPM q-samples)
                    x_t = self.process.interpolate(batch_imgs, noise, t_val)
                    t_vec = torch.full((bsz,), t_val, device=self.device, dtype=batch_imgs.dtype)
                    t_for_target = t_val
                    w_t = float(w_fn(t_val))
                else:
                    # random (t, ε) per sample, keyed by (seed, draw, global index):
                    # a pure function of the sample, independent of batch grouping
                    t_vec, noise = _random_draw(seed, t_idx, start, batch_imgs)
                    x_t = self.process.interpolate(batch_imgs, noise, t_vec)
                    t_for_target = t_vec
                    w_t = 1.0
                if self.loss_type == "mse":
                    u = self.process.target(batch_imgs, noise, t_for_target).to(batch_imgs.dtype)
                    box = [grad_fn(params, buffers, x_t, t_vec, batch_labels, u)]
                else:
                    box = [grad_fn(params, buffers, x_t, t_vec, batch_labels)]
                # the flatten + normalise + weighted accumulation, in place and
                # in one place (see _accumulate for the memory + parity contract)
                emb = _accumulate(box, bsz, normalize=self.normalize, w_t=w_t, emb=emb)
                w_sum += w_t
            emb.div_(w_sum)                                     # uniform => emb / T (parity)
            projected = self._proj.project(emb, model_id=0)
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device=self.device)
            features[start:start + bsz] = projected.detach().cpu()
            start += bsz
            if log_every and start % max(1, log_every) < self.batch_size:
                print(f"  featurize: {start}/{N}")
                if on_progress is not None and start < N:
                    on_progress(done=start, total=N)
        if on_progress is not None:
            on_progress(done=N, total=N)
        return features


    def _grad_fn(self, image_shape):
        """``(params, buffers, per-sample-grad fn)`` for this featurizer's readout.

        Shared by :meth:`featurize` and :meth:`featurize_states` so the two cannot
        drift into computing different gradients — the trajectory variant differs
        only in where ``x_t`` comes from, and a second copy of these closures is
        exactly how "Journey-TRAK" would quietly become a different estimator.
        """
        from torch.func import grad, vmap, functional_call

        self.model.eval()
        params = {k: v.detach() for k, v in self.model.named_parameters() if v.requires_grad}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        mask = _build_output_mask(self.output_mask, image_shape, self.device)
        mask_sum = mask.sum()

        # Output-function readouts (== D-TRAK's `--f` options; the score whose
        # ∇_θ is the gradient feature). `mse` = L_Simple (needs the target u);
        # the rest are target-free. Masked forms reduce to the D-TRAK definition
        # under a full mask (mask≡1, mask_sum=d).
        def f_mean(params, buffers, x_t, t_s, c_s):          # "das": mean(ε)
            v = functional_call(self.model, (params, buffers),
                                args=(x_t.unsqueeze(0), t_s.unsqueeze(0), c_s.unsqueeze(0)))
            return (v.reshape(-1) * mask).sum() / mask_sum

        def f_msl2(params, buffers, x_t, t_s, c_s):          # "dtrak": mean(ε²) = L_Square
            v = functional_call(self.model, (params, buffers),
                                args=(x_t.unsqueeze(0), t_s.unsqueeze(0), c_s.unsqueeze(0)))
            return (v.reshape(-1) ** 2 * mask).sum() / mask_sum

        def f_l1(params, buffers, x_t, t_s, c_s):            # "l1norm": Σ|ε| (D-TRAK: no /d)
            v = functional_call(self.model, (params, buffers),
                                args=(x_t.unsqueeze(0), t_s.unsqueeze(0), c_s.unsqueeze(0)))
            return (v.reshape(-1).abs() * mask).sum()

        def f_l2(params, buffers, x_t, t_s, c_s):            # "l2norm": sqrt(Σε²) (D-TRAK: no /d)
            v = functional_call(self.model, (params, buffers),
                                args=(x_t.unsqueeze(0), t_s.unsqueeze(0), c_s.unsqueeze(0)))
            return ((v.reshape(-1) ** 2 * mask).sum() + 1e-12).sqrt()

        def f_mse(params, buffers, x_t, t_s, c_s, u):        # "trak": mean((ε-u)²) = L_Simple
            v = functional_call(self.model, (params, buffers),
                                args=(x_t.unsqueeze(0), t_s.unsqueeze(0), c_s.unsqueeze(0)))
            r = (v.reshape(-1) - u.reshape(-1)) ** 2
            return (r * mask).sum() / mask_sum

        _READOUTS = {"mean": f_mean, "msl2": f_msl2, "l1": f_l1, "l2": f_l2}
        if self.loss_type == "mse":
            grad_fn = vmap(grad(f_mse), in_dims=(None, None, 0, 0, 0, 0))
        elif self.loss_type in _READOUTS:
            grad_fn = vmap(grad(_READOUTS[self.loss_type]), in_dims=(None, None, 0, 0, 0))
        else:
            raise ValueError(f"unknown loss_type '{self.loss_type}'")
        return params, buffers, grad_fn

    def featurize_states(self, states, t_values, labels, *, log_every: int = 100,
                         on_progress=None) -> torch.Tensor:
        """Journey-TRAK ([e]): features from GIVEN ``(x_t, t)`` pairs.

        ``states`` is ``(N, P, C, H, W)`` — for each sample, the P intermediate
        states of its own generation trajectory — and ``t_values`` the matching
        ``(P,)`` times. The one difference from :meth:`featurize` is where ``x_t``
        comes from: there it is a clean image noised to time t, here it is the
        actual latent the sampler passed through. That is the whole method — a
        point on the ODE path is not distributed like an independently noised
        image, so attributing over the journey asks a different question.

        ``mse``/L_Simple is refused: its target needs the ``(x_0, noise)`` pair
        that produced ``x_t``, and a trajectory latent has no such pair. Silently
        substituting one would produce a number that is not L_Simple.
        """
        if self.loss_type == "mse":
            raise ValueError(
                "the L_Simple (mse) readout needs the (x_0, noise) pair behind x_t, "
                "which a trajectory latent does not have. Use a target-free readout "
                "(das/dtrak/l1norm/l2norm) for the journey features.")
        states = torch.as_tensor(states)
        N, P = states.shape[0], states.shape[1]
        if len(t_values) != P:
            raise ValueError(f"{P} trajectory states but {len(t_values)} times")
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        params, buffers, grad_fn = self._grad_fn(states.shape[2:])
        w_fn = _WEIGHT_FNS[self.time_weight]
        labels = torch.as_tensor(labels, dtype=torch.long)

        features = torch.zeros(N, self.proj_dim, dtype=torch.float32)
        for start in range(0, N, self.batch_size):
            stop = min(start + self.batch_size, N)
            bsz = stop - start
            batch = states[start:stop].to(self.device)              # (b, P, C, H, W)
            batch_labels = labels[start:stop].to(self.device)
            emb, w_sum = None, 0.0
            for p, t_val in enumerate(float(t) for t in t_values):
                x_t = batch[:, p]
                t_vec = torch.full((bsz,), t_val, device=self.device, dtype=x_t.dtype)
                w_t = float(w_fn(t_val))
                # SAME accumulation as `featurize` -- one function, so the journey
                # cannot silently become a different estimator, and so the LoRA
                # platform's 1.42 GiB-per-buffer arithmetic holds here too
                emb = _accumulate([grad_fn(params, buffers, x_t, t_vec, batch_labels)],
                                  bsz, normalize=self.normalize, w_t=w_t, emb=emb)
                w_sum += w_t
            emb.div_(w_sum)
            features[start:stop] = self._proj.project(emb, model_id=0).detach().cpu()
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device=self.device)
            if log_every and stop % max(1, log_every) < self.batch_size:
                print(f"  journey featurize: {stop}/{N}")
                if on_progress is not None and stop < N:
                    on_progress(done=stop, total=N)
        if on_progress is not None:
            on_progress(done=N, total=N)
        return features


class GroupedGradFeaturizer(GradFeaturizer):
    """Retain one shared-projection query vector per parameter tensor group."""

    def __init__(self, *args, groups: Optional[dict[str, Sequence[str]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        from .parameter_weighting import parameter_slices

        trainable = [(name, parameter) for name, parameter in self.model.named_parameters()
                     if parameter.requires_grad]
        tensor_names, tensor_slices = parameter_slices(trainable)
        if groups is None:
            self.group_names, self.group_slices = tensor_names, tensor_slices
            return
        offsets = {name: slc for name, slc in zip(tensor_names, tensor_slices)}
        used, named_slices = set(), []
        for group_name, members_value in groups.items():
            members = list(members_value)
            if not members or any(name not in offsets for name in members):
                raise ValueError(f"group {group_name!r} has unknown/no parameters")
            if any(name in used for name in members):
                raise ValueError("custom parameter groups overlap")
            ordered = sorted((offsets[name].start, offsets[name].stop, name) for name in members)
            if any(ordered[i][1] != ordered[i + 1][0] for i in range(len(ordered) - 1)):
                raise ValueError(f"group {group_name!r} is not contiguous in parameter order")
            named_slices.append((slice(ordered[0][0], ordered[-1][1]), str(group_name)))
            used.update(members)
        if used != set(tensor_names):
            raise ValueError("custom parameter groups must cover every trainable parameter")
        named_slices.sort(key=lambda item: item[0].start)
        self.group_slices = [item[0] for item in named_slices]
        self.group_names = [item[1] for item in named_slices]

    def featurize_grouped(self, dataset, *, seed: int = 42,
                          max_samples: Optional[int] = None, log_every: int = 100,
                          on_progress=None) -> torch.Tensor:
        """Return CPU float32 ``(N, groups, projection)`` query features."""
        return torch.cat(list(self.iter_grouped(
            dataset, seed=seed, max_samples=max_samples, log_every=log_every,
            on_progress=on_progress)), dim=0)

    def iter_grouped(self, dataset, *, seed: int = 42,
                     max_samples: Optional[int] = None, log_every: int = 100,
                     on_progress=None):
        """Yield original MC batches without restarting/reindexing their RNG."""
        import torch.utils.data as td
        from .parameter_weighting import project_parameter_groups

        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        params, buffers, grad_fn = self._grad_fn(dataset.images[0].shape)
        weight_fn = _WEIGHT_FNS[self.time_weight]
        images = torch.stack([dataset.images[i] for i in range(n)])
        labels = torch.tensor([int(dataset.labels[i]) for i in range(n)], dtype=torch.long)
        loader = td.DataLoader(td.TensorDataset(images, labels),
                               batch_size=self.batch_size, shuffle=False)
        timesteps = self._timesteps() if self.t_sampling == "grid" else [None] * self.T
        start = 0
        for batch_images, batch_labels in loader:
            batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
            batch_size = batch_images.shape[0]
            accumulated, weight_sum = None, 0.0
            for draw, time_value in enumerate(timesteps):
                if self.t_sampling == "grid":
                    torch.manual_seed(seed * 1000 + draw)
                    torch.cuda.manual_seed_all(seed * 1000 + draw)
                    noise = torch.randn_like(batch_images)
                    state = self.process.interpolate(batch_images, noise, time_value)
                    times = torch.full((batch_size,), time_value, device=self.device,
                                       dtype=batch_images.dtype)
                    target_time, weight = time_value, float(weight_fn(time_value))
                else:
                    times, noise = _random_draw(seed, draw, start, batch_images)
                    state = self.process.interpolate(batch_images, noise, times)
                    target_time, weight = times, 1.0
                if self.loss_type == "mse":
                    target = self.process.target(batch_images, noise, target_time).to(batch_images.dtype)
                    gradient = grad_fn(params, buffers, state, times, batch_labels, target)
                else:
                    gradient = grad_fn(params, buffers, state, times, batch_labels)
                accumulated = _accumulate([gradient], batch_size, normalize=self.normalize,
                                          w_t=weight, emb=accumulated)
                weight_sum += weight
            accumulated.div_(weight_sum)
            projected = project_parameter_groups(accumulated, self.group_slices, self._proj)
            yield projected.detach().cpu()
            start += batch_size
            if log_every and start % max(1, log_every) < self.batch_size:
                print(f"  grouped featurize: {start}/{n}")
                if on_progress is not None and start < n:
                    on_progress(done=start, total=n)
        if on_progress is not None:
            on_progress(done=n, total=n)
