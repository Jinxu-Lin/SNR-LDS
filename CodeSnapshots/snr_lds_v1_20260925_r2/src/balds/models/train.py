"""Process-agnostic training loop.

Returns the trained model + loss history; it does **not** write to disk (the app
use-case persists via the store, so the Business layer stays IO-free). Works for
any registered :class:`~balds.schema.generative.GenerativeProcess`: for ``cfm``
the target is the velocity, for ``ddpm`` the added noise — the process supplies
``interpolate``/``target``/``per_sample_loss``.

Recipe knobs (DAS-aligned formal recipe, adjudicated 2026-08-07): AdamW weight
decay, linear-warmup + cosine LR schedule, random horizontal flip, UNet dropout,
and periodic mid-training checkpoints via ``on_checkpoint``. Every knob defaults
to the pre-recipe behaviour (plain-Adam-equivalent, no warmup, no flip, no
dropout, final-only checkpoint), so legacy callers are unaffected.
Unconditional training: pass ``num_classes=None`` — no class embedding is built,
labels are ignored, and ``p_uncond`` is irrelevant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from balds.schema.registry import PROCESSES
from .unet import UNetCFM


def train_model(
    process_name: str,
    dataset,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    num_classes: int | None,
    p_uncond: float = 0.1,
    device: str = "cuda",
    base_ch: int = 128,
    log_every: int = 1000,
    compile_model: bool = True,
    weight_decay: float = 0.0,
    warmup_frac: float = 0.0,
    dropout: float = 0.0,
    hflip: bool = False,
    checkpoint_steps: tuple[int, ...] = (),
    on_checkpoint=None,
    on_progress=None,
    model: nn.Module | None = None,
    amp_dtype: torch.dtype = torch.float16,
    grad_accum: int = 1,
    process=None,
) -> tuple[nn.Module, list[dict]]:
    """Train a (class-conditional or unconditional) model for ``process_name``.

    ``num_classes=None`` trains unconditionally (labels ignored, no CFG dropout).
    ``on_checkpoint(step, model)``, if given, is called at each step listed in
    ``checkpoint_steps`` (mid-training checkpoints for TracInCP/GAS); the caller
    persists — this function performs no IO. ``on_progress(**fields)`` is called
    at step 1 (liveness) and every ``log_every`` steps.

    ``model``: pass a pre-built module (e.g. the SD3.5+LoRA adapter, whose base
    is frozen) to train it instead of constructing a ``UNetCFM``; ``base_ch``/
    ``dropout`` are then ignored. The optimizer covers ``model.parameters()``
    either way — frozen parameters receive no gradient and AdamW skips them.
    """
    process = process if process is not None else PROCESSES.get(process_name)
    if getattr(process, "name", process_name) != process_name:
        raise ValueError(f"process instance {getattr(process, 'name', None)!r} does not "
                         f"match requested process_name={process_name!r}")
    torch.manual_seed(seed)
    if model is None:
        model = UNetCFM(base_ch=base_ch, num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    # AdamW with weight_decay=0 is mathematically identical to Adam (the
    # decoupled decay term vanishes), so the legacy default path is unchanged.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = int(round(warmup_frac * steps))
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                               end_factor=1.0, total_iters=warmup_steps),
             torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps - warmup_steps))],
            milestones=[warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda")
    run_model = torch.compile(model) if compile_model else model

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    it = iter(loader)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    conditional = num_classes is not None
    null_class = num_classes if conditional else None
    ckpt_set = set(int(s) for s in checkpoint_steps)
    run_model.train()
    history: list[dict] = []

    for step in range(1, steps + 1):
        # One optimizer step = grad_accum micro-batches. `steps` therefore always
        # counts OPTIMIZER steps, so the epoch budget (steps = epochs*N/effective
        # batch) means the same thing whether or not the effective batch had to be
        # split. grad_accum=1 leaves the loop byte-for-byte as it was.
        optimizer.zero_grad()
        for micro in range(grad_accum):
            try:
                x1, labels = next(it)
            except StopIteration:
                it = iter(loader)
                x1, labels = next(it)
            x1 = x1.to(device)
            B = x1.shape[0]
            if hflip:
                flip = torch.rand(B, device=device) < 0.5
                x1 = torch.where(flip.view(-1, 1, 1, 1), x1.flip(-1), x1)
            if conditional:
                labels = labels.to(device).clone()
                labels[torch.rand(B, device=device) < p_uncond] = null_class
            else:
                labels = None

            noise = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            x_t = process.interpolate(x1, noise, t)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred = run_model(x_t, t, labels)
                loss = nn.functional.mse_loss(pred, process.target(x1, noise, t))
            # mean-of-means: micro-batches are equal-sized (drop_last), so scaling
            # each by 1/grad_accum reproduces the full batch's mean loss exactly
            scaler.scale(loss / grad_accum).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step in ckpt_set and on_checkpoint is not None:
            on_checkpoint(step, model)

        if step % log_every == 0 or (step == 1 and on_progress is not None):
            entry = {"step": step, "loss": float(loss.item()),
                     "lr": scheduler.get_last_lr()[0]}
            if step % log_every == 0:          # history semantics unchanged
                history.append(entry)
            if on_progress is not None:
                N = len(dataset)
                on_progress(step=step, total_steps=steps, loss=entry["loss"],
                            lr=entry["lr"], samples_seen=step * batch_size,
                            epoch=round(step * batch_size / max(N, 1), 2))
    return model, history
