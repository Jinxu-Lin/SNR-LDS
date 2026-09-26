"""SD3.5-Medium + LoRA — the ArtBench latent-flow platform ([o], researcher
decision 2026-08-07: base = SD3.5 Medium, LoRA rank 128, 256x256, style-name
prompt conditioning; recipe mirrored from the DAS original ArtBench scripts).

Why this slots straight into the existing pipeline: SD3.5 *is* rectified-flow /
conditional flow matching in VAE latent space, so the platform runs under the
ordinary ``process="cfm"`` seam — same interpolation ``x_t = (1-t)·ε + t·x1``,
same target ``x1 - ε``, same uniform-t training draw, same Euler sampler. The
only translation lives in :class:`Sd3LoraAdapter`: SD3's native convention is
time-reversed and sign-flipped (σ = noise fraction = ``1 - t``; the transformer
takes ``timestep = σ·1000`` and predicts ``ε - x0``), so the adapter converts
time and negates the output, exposing the exact ``(x_t, t, class_label)``
signature every stage (train loop, featurizer's vmap, GT losses, e_n) already
speaks. Class labels are style indices; the adapter holds one precomputed
prompt embedding per style as buffers, so no text encoder is ever loaded after
the one-off ``balds latents`` stage.

Checkpoints store ONLY the LoRA weights (the 2.5B base is frozen and pinned by
name in the checkpoint); ``diffusers`` is imported lazily so the package stays
importable without it.

LoRA parity with DAS (train_text_to_image_lora.py, read 2026-08-17 from
the DAS ArtBench10 reference implementation): rank 128 hardcoded, down-projection init
``normal(std=1/rank)``, up-projection zeros, scale = alpha/rank with
alpha = rank (diffusers ``LoRAAttnProcessor`` convention).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """``base(x) + (alpha/rank) · B(A(x))`` with the base Linear frozen.

    Parameter names carry the ``lora_`` prefix so :func:`lora_state_dict` can
    slice a checkpoint by name alone.

    ``lora_A``/``lora_B`` are real ``nn.Linear(bias=False)`` submodules — not
    bare Parameters — so EK-FAC's ``kfac_target_modules`` sees the delta path
    as two serial linear layers and its standard hooks capture exactly the
    K-FAC quantities of TASK_EKFAC_LORA §2: layer A gets the cast input x and
    ``g_a = scale·Bᵀg_y``; layer B gets ``a = A x`` and ``g_b = scale·g_y``
    (the scale multiply sits OUTSIDE both submodules, so autograd routes it
    into the captured output-gradients — never multiply it in by hand). The
    frozen base Linear stays invisible to the curvature (requires_grad filter).
    State-dict cost of the refactor: ``lora_A`` → ``lora_A.weight`` (decided
    2026-09-02 while no LoRA production checkpoint exists yet).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1.0 / rank)   # diffusers LoRALinearLayer init
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.base.weight, self.base.bias)
        delta = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype)))
        return out + self.scale * delta.to(out.dtype)


def inject_lora(root: nn.Module, target_suffixes: Sequence[str], *, rank: int,
                alpha: float) -> int:
    """Replace every ``nn.Linear`` whose qualified name ends with a target
    suffix by a :class:`LoRALinear`; freeze everything else. Returns the number
    of wrapped layers (0 would mean the suffix list is wrong for this model —
    callers must treat that as an error, not a no-op)."""
    for p in root.parameters():
        p.requires_grad_(False)
    targets = list(dict.fromkeys(str(s) for s in target_suffixes))
    if not targets:
        raise ValueError("LoRA target suffixes cannot be empty")
    counts = {s: 0 for s in targets}
    n = 0
    for name, mod in list(root.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            matched = [s for s in targets if full == s or full.endswith("." + s)]
            if isinstance(child, nn.Linear) and matched:
                setattr(mod, child_name, LoRALinear(child, rank, alpha))
                for suffix in matched:
                    counts[suffix] += 1
                n += 1
    missing = [s for s, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"LoRA target suffixes matched no Linear layer: {missing}; "
                         f"the target list is wrong for this architecture")
    return n


def reset_lora_(model: nn.Module, seed: int) -> None:
    """Re-initialise every LoRA adapter from an explicit seed (A ~ N(0, 1/r²),
    B = 0). The training loop seeds the global RNG for data order and noise,
    but the adapter is built BEFORE that — so subset retrains and reruns must
    pin the init here or the trained LoRA is not reproducible."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.weight.copy_(
                    torch.randn(m.lora_A.weight.shape, generator=g) / m.rank)
                m.lora_B.weight.zero_()


def lora_state_dict(model: nn.Module) -> dict:
    """Only the LoRA parameters (float32 CPU) — the whole trainable state."""
    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()
            if "lora_" in k}


def upgrade_lora_state(state: dict, model: nn.Module) -> dict:
    """Map a pre-a4c1e6b LoRA state (``...lora_A`` / ``...lora_B`` raw
    Parameters, 2026-08-17 format) onto the submodule keys the current
    ``LoRALinear`` expects (``...lora_A.weight`` / ``...lora_B.weight``).

    The task book assumed no LoRA checkpoint predated the refactor; the
    ArtBench-2 main model (08-30) does. Shapes are identical between the two
    formats (``(rank, in)`` / ``(out, rank)``), so this is a pure rename; a
    transposed tensor is caught by the shape check rather than loaded silently.
    """
    want = {k: v.shape for k, v in model.state_dict().items() if "lora_" in k}
    out = {}
    for k, v in state.items():
        nk = k
        if k not in want and (k.endswith(".lora_A") or k.endswith(".lora_B")):
            nk = k + ".weight"
        if nk in want and tuple(v.shape) != tuple(want[nk]):
            if tuple(v.T.shape) == tuple(want[nk]):
                v = v.T.contiguous()
            else:
                raise ValueError(f"LoRA tensor {k}: shape {tuple(v.shape)} does not match "
                                 f"the model's {tuple(want[nk])}")
        out[nk] = v
    return out


def load_lora_state(model: nn.Module, state: dict) -> None:
    state = upgrade_lora_state(state, model)
    missing = [k for k in model.state_dict() if "lora_" in k and k not in state]
    if missing:
        raise ValueError(f"LoRA checkpoint is missing {len(missing)} tensors "
                         f"(first: {missing[0]}) — rank/targets mismatch?")
    model.load_state_dict(state, strict=False)


class Sd3LoraAdapter(nn.Module):
    """SD3 transformer behind the pipeline's ``(x_t, t, class_label)`` seam.

    * time: our ``t`` is the data fraction (t=1 ⇒ clean); SD3's σ is the noise
      fraction and its timestep input is ``σ·1000`` — so ``σ = 1 - t``;
    * output: SD3 predicts ``ε - x0``; our target is ``x1 - ε`` — negate;
    * conditioning: ``class_label`` indexes the per-style prompt-embedding
      buffers (styles keep their original ArtBench indices, so a 2-style
      subset platform shares the same 10-row table).

    Pure tensor ops end to end, so ``torch.func`` vmap/grad featurization works
    unchanged, with only the LoRA parameters trainable.
    """

    def __init__(self, transformer: nn.Module, prompt_embeds: torch.Tensor,
                 pooled_embeds: torch.Tensor) -> None:
        super().__init__()
        self.transformer = transformer
        self.register_buffer("prompt_embeds", prompt_embeds)   # (S, L, D)
        self.register_buffer("pooled_embeds", pooled_embeds)   # (S, P)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                class_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        if class_label is None:
            raise ValueError("the SD3.5 platform is prompt-conditional; every "
                             "stage must pass style indices as class_label")
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        sigma = (1.0 - t).to(x_t.device)
        # timestep stays float32: SD3's time_proj does its sinusoid in fp32 and
        # casts afterwards; a bf16 timestep would quantise sigma·1000 by ~±2.
        timestep = sigma.float() * 1000.0
        pe = self.prompt_embeds[class_label]
        pp = self.pooled_embeds[class_label]
        out = self.transformer(
            hidden_states=x_t.to(self.prompt_embeds.dtype),
            encoder_hidden_states=pe,
            pooled_projections=pp,
            timestep=timestep,
            return_dict=False,
        )[0]
        return -out.float()                       # ε - x0  ->  x1 - ε


# --------------------------------------------------------------------------- #
# heavy loaders (diffusers/transformers imported lazily)
# --------------------------------------------------------------------------- #

def load_sd3_transformer(base_model: str, *, device: str, dtype: torch.dtype):
    from diffusers import SD3Transformer2DModel
    tf = SD3Transformer2DModel.from_pretrained(base_model, subfolder="transformer",
                                               torch_dtype=dtype)
    tf.requires_grad_(False)
    return tf.to(device).eval()


def build_lora_adapter(base_model: str, prompt_embeds: torch.Tensor,
                       pooled_embeds: torch.Tensor, *, rank: int, alpha: float,
                       targets: Sequence[str], device: str,
                       dtype: torch.dtype = torch.bfloat16,
                       lora_state: Optional[dict] = None) -> Sd3LoraAdapter:
    """Frozen base + (fresh | loaded) LoRA + prompt buffers, on ``device``.

    LoRA parameters are kept float32 regardless of the base dtype (the DAS /
    diffusers mixed-precision convention: half base, fp32 adapters).
    """
    tf = load_sd3_transformer(base_model, device=device, dtype=dtype)
    n = inject_lora(tf, targets, rank=rank, alpha=alpha)
    if n == 0:
        raise ValueError(f"LoRA targets {list(targets)} matched no Linear layer "
                         f"in {base_model} — wrong suffix list for this architecture")
    adapter = Sd3LoraAdapter(tf, prompt_embeds.to(device, dtype),
                             pooled_embeds.to(device, dtype)).to(device)
    for k, p in adapter.named_parameters():
        if "lora_" in k:
            p.data = p.data.float()
    if lora_state is not None:
        load_lora_state(adapter, lora_state)
    adapter.eval()
    return adapter


def load_vae(base_model: str, *, device: str):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae",
                                        torch_dtype=torch.float32)
    vae.requires_grad_(False)
    return vae.to(device).eval()


@torch.no_grad()
def encode_images(vae, images: torch.Tensor, *, batch_size: int = 16,
                  device: str = "cuda") -> torch.Tensor:
    """Images in [-1, 1] -> scaled latent MEANS (deterministic by design).

    Using the posterior mean rather than a sample keeps every downstream
    consumer (GT losses, featurization, e_n) a deterministic function of the
    dataset — the property the whole seeding discipline is built on. The
    encoder noise it discards is negligible against diffusion-time noise.
    """
    sf = float(vae.config.scaling_factor)
    sh = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    out = []
    for s in range(0, images.shape[0], batch_size):
        x = images[s:s + batch_size].to(device=device, dtype=torch.float32)
        posterior = vae.encode(x).latent_dist
        out.append(((posterior.mean - sh) * sf).cpu())
    return torch.cat(out)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, *, batch_size: int = 16,
                   device: str = "cuda") -> torch.Tensor:
    """Scaled latents -> images in [-1, 1] (for visualisation only)."""
    sf = float(vae.config.scaling_factor)
    sh = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    out = []
    for s in range(0, latents.shape[0], batch_size):
        z = latents[s:s + batch_size].to(device=device, dtype=torch.float32)
        out.append(vae.decode(z / sf + sh).sample.clamp(-1, 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def compute_prompt_embeds(base_model: str, prompts: Sequence[str], *,
                          device: str = "cuda",
                          max_sequence_length: int = 256):
    """Per-prompt SD3 embeddings WITHOUT the T5 tower.

    Mirrors ``StableDiffusion3Pipeline.encode_prompt`` with
    ``text_encoder_3=None`` (diffusers 0.37 source): per CLIP tower take
    ``hidden_states[-2]`` (77 tokens) + the projected pooled embedding,
    concat the two towers channel-wise (2048), zero-pad to the transformer's
    ``joint_attention_dim`` and append a zero T5 block of
    ``max_sequence_length`` tokens. Returns ``(prompt_embeds (S, L, D),
    pooled (S, P))`` on the CPU in float32.
    """
    from diffusers import SD3Transformer2DModel
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    joint_dim = int(SD3Transformer2DModel.load_config(
        base_model, subfolder="transformer")["joint_attention_dim"])

    towers = []
    pooled = []
    for tok_sub, enc_sub in (("tokenizer", "text_encoder"),
                             ("tokenizer_2", "text_encoder_2")):
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder=tok_sub)
        encoder = CLIPTextModelWithProjection.from_pretrained(
            base_model, subfolder=enc_sub, torch_dtype=torch.float32).to(device).eval()
        ids = tokenizer(list(prompts), padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        enc_out = encoder(ids, output_hidden_states=True)
        towers.append(enc_out.hidden_states[-2].cpu())     # (S, 77, d_tower)
        pooled.append(enc_out[0].cpu())                    # projected pooled
        del encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    clip_embeds = torch.cat(towers, dim=-1)                          # (S, 77, 2048)
    clip_embeds = F.pad(clip_embeds, (0, joint_dim - clip_embeds.shape[-1]))
    t5_block = torch.zeros(clip_embeds.shape[0], max_sequence_length, joint_dim)
    return torch.cat([clip_embeds, t5_block], dim=-2), torch.cat(pooled, dim=-1)
