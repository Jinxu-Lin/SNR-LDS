"""SD3.5+LoRA latent-platform plumbing ([o]) — the app-layer half.

The platform reuses the ordinary stage machinery (train loop, Euler sampler,
GT losses, featurizer, score/evaluate) because the adapter speaks the same
``(x_t, t, class_label)`` seam and the process is literally ``cfm`` in latent
space. What this module owns is everything that differs:

* the one-off ``balds latents`` encode stage (VAE latents for both horizontal
  orientations + per-style prompt embeddings — the only stage that ever loads
  text encoders or touches 256x256 pixels);
* dataset access: every later stage reads the cached latents, never the raw
  image folder;
* model lifecycle: checkpoints hold ONLY LoRA weights; the frozen 2.5B base is
  re-materialized from ``artbench.base_model`` at load time;
* the train/generate/subset-retrain drivers with the DAS-parity recipe knobs
  (read 2026-08-17 from the DAS ArtBench10 reference implementation run_train.sh: 100 epochs,
  effective batch 64, lr 3e-4, wd 1e-6, cosine, rank 128, random flip,
  prompt "a {style} painting").

``pipeline.py``'s use-cases branch into these functions per
``datasets.<name>.platform`` (lazy import there; this module imports pipeline,
not the other way round at module level).
"""
from __future__ import annotations
import time
from typing import Optional
import torch
from balds.schema.artifact import ArtifactKind as K
from balds.schema.logging import get_logger, set_context
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset, LatentPairDataset
from balds.schema.registry import PROCESSES
from balds.models import ddim_sample, ddpm_sample, euler_solve, train_model
from balds.models.sd3 import build_lora_adapter, compute_prompt_embeds, decode_latents, encode_images, load_vae, lora_state_dict, reset_lora_
from balds.artifacts.progress import ProgressCSV, progress_path
from .config import resolve_Q
from .common import _balanced_labels, _load_ds, _pool_noise
log = get_logger('balds.latent')
PLATFORM_UNET = 'unet'
PLATFORM_SD35 = 'sd35_lora'
PLATFORMS = {PLATFORM_SD35: {'backend': 'sd3', 'process': 'cfm', 'latent_shape': [16, 32, 32], 'sampler': 'euler', 'steps_key': 'gen_ode_steps', 'config_key': 'artbench'}}

def platform_of(cfg, dataset: str) -> str:
    return (cfg.get('datasets', {}).get(dataset, {}) or {}).get('platform', PLATFORM_UNET)

def is_latent_platform(cfg, dataset: str) -> bool:
    return platform_of(cfg, dataset) in PLATFORMS

def platform_spec(cfg, dataset: str) -> dict:
    platform = platform_of(cfg, dataset)
    if platform not in PLATFORMS:
        raise ValueError(f"'{dataset}' is not a registered latent platform (got datasets.{dataset}.platform={platform!r})")
    return {'platform': platform, **PLATFORMS[platform]}

def _resolve_base_model(cfg, name: str) -> str:
    """Prefer a snapshot under ``<data_root>/models/`` over the HF hub id.

    A locally materialized base model makes the whole platform work with no
    network and no HF auth on the machine that runs it — which matters because
    SD3.5 is a gated repo, so an executor box without a token cannot fetch it at
    all. Resolved by data_root rather than hardcoding an absolute path, so the
    same config works on every machine that has the model synced.

    Also applied to the path a CHECKPOINT carries (2026-09-07): a checkpoint
    records the base it was trained against, and the ones in the library record
    it as ``_Data/models/stable-diffusion-3.5-medium`` — a path relative to the
    cwd, from before ``data_root`` was anchored to the repo root. Used verbatim
    it resolves only when the command happens to be launched from the repo root,
    and fails with a "couldn't connect to huggingface.co" error everywhere else
    (a smoke run against an alternate ``data_root``, another machine, a
    worktree). A stored path is a claim about a filesystem, so it gets
    re-resolved on the machine reading it; an absolute path or an existing
    directory still passes through untouched, so the production launch resolves
    to exactly what it resolved to before.
    """
    import os
    if os.path.isabs(name) or os.path.isdir(name):
        return name
    root = (cfg.get('storage', {}) or {}).get('data_root', '_Data')
    local = os.path.join(root, 'models', name.split('/')[-1])
    return local if os.path.isdir(local) else name

def artbench_cfg(cfg) -> dict:
    a = dict(cfg.get('artbench', {}) or {})
    a.setdefault('base_model', 'stabilityai/stable-diffusion-3.5-medium')
    a['base_model'] = _resolve_base_model(cfg, str(a['base_model']))
    a.setdefault('base_dtype', 'bfloat16')
    a.setdefault('prompt_template', 'a {style} painting')
    a.setdefault('lora', {})
    a['lora'] = {'rank': 128, 'alpha': 128, 'targets': ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out'], **(a['lora'] or {})}
    a.setdefault('epochs', 100)
    a.setdefault('batch_size', 64)
    a.setdefault('lr', 0.0003)
    a.setdefault('weight_decay', 1e-06)
    a.setdefault('warmup_steps', 500)
    a.setdefault('latent_shape', [16, 32, 32])
    a.setdefault('encode_batch_size', 16)
    a.setdefault('featurize_batch_size', 1)
    a.setdefault('gen_ode_steps', 100)
    a.setdefault('compile', False)
    return a

def platform_cfg(cfg, dataset: str) -> dict:
    platform_spec(cfg, dataset)
    return artbench_cfg(cfg)

def platform_process(cfg, dataset: str, *, device: str='cpu'):
    platform_spec(cfg, dataset)
    return (PROCESSES.get('cfm'), {})

def _require_latent_shape(acfg: dict, verified: Optional[dict]) -> None:
    """Validate configured latent geometry when VAE verification provides it."""
    want = (verified or {}).get('latent_shape')
    if want is None:
        return
    have = [int(v) for v in acfg['latent_shape']]
    if have != [int(v) for v in want]:
        raise ValueError(f'latent_shape {have} does not match the verified VAE geometry {[int(v) for v in want]} at the platform resolution; refusing to encode or sample at another latent size')

def _require_platform_process(cfg, dataset: str, process: str) -> dict:
    spec = platform_spec(cfg, dataset)
    if process != spec['process']:
        raise ValueError(f"latent platform {spec['platform']} requires process={spec['process']!r}, got {process!r}")
    return spec

def style_prompts(cfg) -> list[str]:
    from balds.data.artbench import ARTBENCH_STYLES
    tpl = artbench_cfg(cfg)['prompt_template']
    return [tpl.format(style=s.replace('_', ' ').lower()) for s in ARTBENCH_STYLES]

def platform_prompts(cfg, dataset: str) -> list[str]:
    from balds.data.artbench import ARTBENCH_STYLES
    tpl = platform_cfg(cfg, dataset)['prompt_template']
    return [tpl.format(style=s.replace('_', ' ').lower()) for s in ARTBENCH_STYLES]

def _load_split(store, dataset: str, split: str, *, flip: bool=False):
    spec = RunSpec(dataset=dataset)
    blob = store.load(K.LATENTS, spec, split=split, **{'flip': True} if flip else {})
    return (blob['latents'].float(), blob['labels'])

def latent_train_plain(store, dataset: str) -> ImageDataset:
    z, y = _load_split(store, dataset, 'train')
    return ImageDataset(z, y)

def latent_test(store, dataset: str) -> ImageDataset:
    z, y = _load_split(store, dataset, 'test')
    return ImageDataset(z, y)

def latent_train_pair(store, dataset: str) -> LatentPairDataset:
    z, y = _load_split(store, dataset, 'train')
    zf, _ = _load_split(store, dataset, 'train', flip=True)
    return LatentPairDataset(z, zf, y)

def encode_query_images(cfg, dataset: str, images_u8, labels, *, device: str):
    """VAE-encode generated review queries with the platform's own VAE.

    Inject queries are derived artifacts and therefore never enter the shared
    ``LATENTS`` cache.  They still use the exact deterministic posterior-mean
    rule used by :class:`EncodeUseCase`, just at feature/EK-FAC consumption
    time.
    """
    pspec = platform_spec(cfg, dataset)
    acfg = platform_cfg(cfg, dataset)
    vae = load_vae(acfg['base_model'], device=device)
    images = torch.as_tensor(images_u8).float().div(127.5).sub(1.0)
    z = encode_images(vae, images, batch_size=int(acfg['encode_batch_size']), device=device).float()
    del vae
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ImageDataset(z, torch.as_tensor(labels).long())

def _lora_checkpoint(adapter, acfg: dict, *, step: int, process: str, platform: str=PLATFORM_SD35, verified: Optional[dict]=None) -> dict:
    out = {'platform': platform, 'model_type': process, 'conditional': True, 'step': step, 'base_model': acfg['base_model'], 'lora_rank': int(acfg['lora']['rank']), 'lora_alpha': float(acfg['lora']['alpha']), 'lora_targets': list(acfg['lora']['targets']), 'lora': lora_state_dict(adapter)}
    if verified:
        out['base_verification'] = dict(verified)
    return out

def build_latent_model(store, cfg, dataset: str, ckpt: Optional[dict], *, device: str):
    """(Re)build the adapter: frozen base + prompt buffers + LoRA (fresh if
    ``ckpt`` is None). The checkpoint's own rank/targets/base win over config so
    an old artifact keeps meaning what it meant when trained."""
    pspec = platform_spec(cfg, dataset)
    acfg = platform_cfg(cfg, dataset)
    pe = store.load(K.PROMPT_EMBEDS, RunSpec(dataset=dataset))
    dtype = getattr(torch, str(acfg['base_dtype']))
    if ckpt is not None and ckpt.get('platform') != pspec['platform']:
        raise ValueError(f"checkpoint platform {ckpt.get('platform')!r} does not match dataset platform {pspec['platform']!r}")
    if ckpt is None:
        return build_lora_adapter(acfg['base_model'], pe['prompt_embeds'], pe['pooled_embeds'], rank=int(acfg['lora']['rank']), alpha=float(acfg['lora']['alpha']), targets=list(acfg['lora']['targets']), device=device, dtype=dtype)
    return build_lora_adapter(_resolve_base_model(cfg, str(ckpt['base_model'])), pe['prompt_embeds'], pe['pooled_embeds'], rank=int(ckpt['lora_rank']), alpha=float(ckpt['lora_alpha']), targets=list(ckpt['lora_targets']), device=device, dtype=dtype, lora_state=ckpt['lora'])

class EncodeUseCase:
    """The platform's stage zero; the only stage that reads 256x256 pixels."""

    def __init__(self, store, cfg, *, device: str='cuda:0') -> None:
        self.store, self.cfg, self.device = (store, cfg, device)

    def run(self, dataset: str, *, force: bool=False) -> dict:
        pspec = platform_spec(self.cfg, dataset)
        spec = RunSpec(dataset=dataset)
        set_context(run_id=spec.digest()[:6], trace_id=f'latents:{dataset}')
        acfg = platform_cfg(self.cfg, dataset)
        want = [('train', False), ('train', True), ('test', False)]
        have = all((self.store.exists(K.LATENTS, spec, split=s, **{'flip': True} if f else {}) for s, f in want)) and self.store.exists(K.PROMPT_EMBEDS, spec)
        if have and (not force):
            log.info('latents + prompt embeds exist, skipping (use --force)')
            return {'skipped': True}
        train_ds, test_ds = _load_ds(self.cfg, dataset)
        vae = load_vae(acfg['base_model'], device=self.device)
        verified = {}
        bs = int(acfg['encode_batch_size'])
        prog = ProgressCSV(progress_path(self.cfg['storage']['data_root'], f'latents_{dataset}'))
        out: dict = {}

        def _encode(ds, *, flip: bool, tag: str, split: str):
            key = {'split': split, **({'flip': True} if flip else {})}
            if self.store.exists(K.LATENTS, spec, **key) and (not force):
                out[tag] = 'skipped'
                return
            chunks = []
            N = len(ds)
            for s in range(0, N, bs):
                batch = torch.stack([ds[i][0] for i in range(s, min(s + bs, N))])
                if flip:
                    batch = batch.flip(-1)
                chunks.append(encode_images(vae, batch, batch_size=bs, device=self.device).half())
                if s // bs % 20 == 0 or s + bs >= N:
                    prog.log(done=min(s + bs, N), total=N, phase=tag)
            z = torch.cat(chunks)
            want = (verified or {}).get('latent_shape')
            if want is not None and list(z.shape[1:]) != [int(v) for v in want]:
                raise ValueError(f'encoded latents have shape {list(z.shape[1:])}, expected the verified geometry {list(want)}')
            blob = {'latents': z, 'labels': ds.labels, 'flip': flip, 'split': split, 'base_model': acfg['base_model'], 'platform': pspec['platform']}
            if verified:
                blob['base_verification'] = dict(verified)
            self.store.save(K.LATENTS, spec, blob, **key)
            out[tag] = list(z.shape)
        _encode(train_ds, flip=False, tag='train', split='train')
        _encode(train_ds, flip=True, tag='train_flip', split='train')
        _encode(test_ds, flip=False, tag='test', split='test')
        if not self.store.exists(K.PROMPT_EMBEDS, spec) or force:
            del vae
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            prompts = platform_prompts(self.cfg, dataset)
            pe, pooled = compute_prompt_embeds(acfg['base_model'], prompts, device=self.device)
            prompt_blob = {'prompt_embeds': pe, 'pooled_embeds': pooled, 'prompts': prompts, 'base_model': acfg['base_model']}
            self.store.save(K.PROMPT_EMBEDS, spec, prompt_blob)
            out['prompt_embeds'] = list(pe.shape)
        return out

def _train_knobs(acfg: dict, n_samples: int) -> dict:
    """Recipe knobs, with the effective batch split into micro-batches.

    ``batch_size`` is the DAS recipe's EFFECTIVE batch (64, from bs32 x 2 GPUs)
    and is what the epoch budget is computed against — it must not change with
    the hardware. A single 24GB card cannot hold it for SD3.5-M: measured
    2026-08-18, batch 64 and even 32 OOM, 16 survives one step and then dies once
    AdamW's states for the 95.6M LoRA parameters are allocated. So the effective
    batch is reached by accumulation instead, which is mathematically the same
    update and leaves the recipe intact.
    """
    eff = int(acfg['batch_size'])
    micro = int(acfg.get('micro_batch_size') or eff)
    micro = min(micro, eff)
    if eff % micro:
        raise ValueError(f"artbench.batch_size ({eff}) must be a multiple of micro_batch_size ({micro}) — otherwise the effective batch is not the recipe's batch")
    steps = max(1, round(float(acfg['epochs']) * n_samples / eff))
    amp_name = str(acfg.get('base_dtype', 'bfloat16'))
    amp_dtype = getattr(torch, amp_name)
    return {'steps': steps, 'batch_size': micro, 'grad_accum': eff // micro, 'effective_batch': eff, 'lr': float(acfg['lr']), 'weight_decay': float(acfg['weight_decay']), 'warmup_frac': float(acfg['warmup_steps']) / steps if steps else 0.0, 'amp_dtype': amp_dtype}

def run_latent_train(store, cfg, dataset: str, seed: int, *, process: str, device: str, force: bool=False) -> dict:
    pspec = _require_platform_process(cfg, dataset, process)
    spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=True)
    set_context(run_id=spec.digest()[:6], trace_id=f'train:{dataset}:{process}')
    if store.exists(K.CHECKPOINT, spec) and (not force):
        log.info('checkpoint exists, skipping')
        return {'skipped': True}
    acfg = platform_cfg(cfg, dataset)
    proc, verified = platform_process(cfg, dataset, device=device)
    pair = latent_train_pair(store, dataset)
    knobs = _train_knobs(acfg, len(pair))
    steps = knobs['steps']
    fracs = (cfg.get('train', {}) or {}).get('checkpoint_fracs', []) or []
    ckpt_steps = sorted({int(round(f * steps)) for f in fracs if 0 < f < 1})
    model = build_latent_model(store, cfg, dataset, None, device=device)
    reset_lora_(model, seed)
    prog = ProgressCSV(progress_path(cfg['storage']['data_root'], f'train_{dataset}_{process}_seed{seed}'))
    log.info('LoRA finetune %s on %s: steps=%d effective_batch=%d (micro %d x accum %d) N=%d lr=%g', acfg['base_model'], dataset, steps, knobs['effective_batch'], knobs['batch_size'], knobs['grad_accum'], len(pair), knobs['lr'])

    def _save_mid(step, mid_model):
        store.save(K.CHECKPOINT, spec, _lora_checkpoint(mid_model, acfg, step=step, process=process, platform=pspec['platform'], verified=verified), step=step)
        log.info('mid LoRA checkpoint saved at step %d/%d', step, steps)
    model, history = train_model(process, pair, steps=steps, batch_size=knobs['batch_size'], lr=knobs['lr'], seed=seed, num_classes=int(cfg['model']['num_classes']), p_uncond=0.0, device=device, compile_model=bool(acfg['compile']), model=model, weight_decay=knobs['weight_decay'], warmup_frac=knobs['warmup_frac'], amp_dtype=knobs['amp_dtype'], grad_accum=knobs['grad_accum'], process=proc, hflip=False, checkpoint_steps=tuple(ckpt_steps), on_checkpoint=_save_mid, on_progress=prog.log)
    store.save(K.CHECKPOINT, spec, _lora_checkpoint(model, acfg, step=steps, process=process, platform=pspec['platform'], verified=verified))
    store.save(K.LOSS_HISTORY, spec, {'model_type': process, 'steps': steps, 'conditional': True, 'p_uncond': 0.0, 'platform': pspec['platform'], **({'base_verification': verified} if verified else {}), 'history': history})
    return {'steps': steps, 'mid_checkpoints': ckpt_steps, 'final_loss': history[-1]['loss'] if history else None}

def run_latent_generate(store, cfg, dataset: str, seed: int, *, process: str, device: str, Q: Optional[int]=None, ode_steps: Optional[int]=None, gen_seed: Optional[int]=None, force: bool=False, artifact_name: Optional[str]=None) -> dict:
    pspec = _require_platform_process(cfg, dataset, process)
    spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=True)
    set_context(run_id=spec.digest()[:6], trace_id=f'generate:{dataset}:{process}')
    acfg = platform_cfg(cfg, dataset)
    classes = cfg['datasets'][dataset]['classes']
    Q = Q or resolve_Q(cfg, dataset)
    sample_steps = int(ode_steps or acfg[pspec['steps_key']])
    generation_key = {'name': artifact_name} if artifact_name is not None else {}
    if store.exists(K.GENERATION, spec, **generation_key) and (not force):
        have = int(store.load(K.GENERATION, spec, **generation_key)['samples'].shape[0])
        if have != Q:
            raise ValueError(f'existing generation has Q={have}, requested Q={Q}; path does not encode Q — pass --force to regenerate')
        log.info('generations exist (Q=%d), skipping', Q)
        return {'skipped': True}
    gen_seed = gen_seed if gen_seed is not None else int(cfg['lds'].get('gen_seed_base', 123)) + seed
    labels = _balanced_labels(classes, Q)
    generation_started = time.perf_counter()
    model = build_latent_model(store, cfg, dataset, store.load(K.CHECKPOINT, spec), device=device)
    shape = [int(v) for v in acfg['latent_shape']]
    torch.manual_seed(gen_seed)
    x0 = torch.randn(Q, *shape)
    with torch.no_grad():
        if pspec['sampler'] == 'euler':
            latents = euler_solve(model, x0, steps=sample_steps, device=device, class_label=labels.to(device)).cpu()
            verified = {}
        else:
            proc, verified = platform_process(cfg, dataset, device=device)
            _require_latent_shape(acfg, verified)
            latents = ddim_sample(model, proc._schedule(device), x0, steps=sample_steps, eta=0.0, class_label=labels.to(device), device=device).cpu()
    vae = load_vae(acfg['base_model'], device=device)
    images = decode_latents(vae, latents, device=device)
    images_u8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    del vae
    blob = {'samples': latents, 'images_u8': images_u8, 'labels': labels, 'Q': Q, 'gen_seed': gen_seed, 'conditional': True, 'platform': pspec['platform'], 'sampler': pspec['sampler'], 'steps': sample_steps, 'artifact_name': artifact_name or 'samples', 'timing_seconds': {'model_load_generation_and_decode': time.perf_counter() - generation_started}}
    if pspec['sampler'] == 'euler':
        blob['ode_steps'] = sample_steps
    if verified:
        blob['base_verification'] = dict(verified)
    store.save(K.GENERATION, spec, blob, **generation_key)
    log.info('generated %d latent samples (+decoded previews), gen_seed=%d', Q, gen_seed)
    return {'Q': Q, 'gen_seed': gen_seed, 'sampler': pspec['sampler'], 'steps': sample_steps, 'shape': list(latents.shape)}

def run_latent_pool(store, cfg, dataset: str, seed: int, *, process: str, device: str, n: int, tag: Optional[str]=None, sampler: Optional[str]=None, steps: Optional[int]=None, eta: float=0.0, gen_seed: Optional[int]=None, batch: Optional[int]=None, force: bool=False, store_latents: bool=False) -> dict:
    """Generate a 256px screening pool through a latent platform's own sampler."""
    pspec = _require_platform_process(cfg, dataset, process)
    if n <= 0:
        raise ValueError('--pool must be positive')
    acfg = platform_cfg(cfg, dataset)
    sampler = str(sampler or pspec['sampler'])
    allowed = {'cfm': {'euler'}, 'ddpm': {'ddim', 'ancestral'}}
    if sampler not in allowed[process]:
        raise ValueError(f'sampler {sampler!r} is incompatible with latent process {process!r}')
    proc, verified = platform_process(cfg, dataset, device=device)
    _require_latent_shape(acfg, verified)
    schedule = proc._schedule(device) if process == 'ddpm' else None
    default_steps = int(acfg[pspec['steps_key']])
    steps = int(steps or (schedule.T if sampler == 'ancestral' else default_steps))
    if sampler == 'ancestral' and steps != schedule.T:
        raise ValueError(f'ancestral sampling always uses the full {schedule.T} steps')
    if sampler != 'ddim' and float(eta) != 0.0:
        raise ValueError('eta is defined only for DDIM')
    batch = int(batch or cfg.get('inject', {}).get('pool', {}).get('batch_size', 512))
    gen_seed = int(gen_seed if gen_seed is not None else cfg.get('inject', {}).get('pool', {}).get('gen_seed', 20260914))
    tag = tag or f'{sampler}{steps}_n{n}'
    spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=True)
    from balds.artifacts.addressing import relpath
    relpath(K.GEN_POOL, spec, tag=tag)
    if store.exists(K.GEN_POOL, spec, tag=tag) and (not force):
        raise FileExistsError(f'generation pool tag {tag!r} already exists; use --force')
    labels = _balanced_labels(cfg['datasets'][dataset]['classes'], n)
    ckpt = store.load(K.CHECKPOINT, spec)
    checkpoint_sha = store.describe(K.CHECKPOINT, spec)['sha256']
    model = build_latent_model(store, cfg, dataset, ckpt, device=device)
    vae = load_vae(acfg['base_model'], device=device)
    images, latent_parts = ([], [])
    shape = tuple((int(v) for v in acfg['latent_shape']))
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        x_T = _pool_noise(range(start, stop), gen_seed=gen_seed, shape=shape)
        class_label = labels[start:stop].to(device)
        if sampler == 'euler':
            latents = euler_solve(model, x_T, steps=steps, device=device, class_label=class_label)
        elif sampler == 'ddim':
            latents = ddim_sample(model, schedule, x_T, steps=steps, eta=float(eta), class_label=class_label, device=device)
        else:
            latents = ddpm_sample(model, schedule, tuple(x_T.shape), x_T=x_T, class_label=class_label, device=device)
        decoded = decode_latents(vae, latents, device=device)
        images.append(((decoded + 1.0) * 127.5).clamp(0, 255).to(torch.uint8))
        if store_latents:
            latent_parts.append(latents.detach().half().cpu())
    payload = {'images_u8': torch.cat(images), 'labels': labels, 'n': n, 'gen_seed': gen_seed, 'sampler': sampler, 'steps': steps, 'eta': float(eta), 'guidance': 0.0, 'checkpoint_sha256': checkpoint_sha, 'code_version': getattr(store, 'code_version', 'dev'), 'conditional': True, 'p_uncond': 0.0, 'platform': pspec['platform'], **({'base_verification': verified} if verified else {})}
    if store_latents:
        payload['latents'] = torch.cat(latent_parts)
    store.save(K.GEN_POOL, spec, payload, tag=tag, upstream=(checkpoint_sha,))
    return {'tag': tag, 'n': n, 'shape': list(payload['images_u8'].shape), 'sampler': sampler, 'steps': steps, 'gen_seed': gen_seed, 'platform': pspec['platform']}

def run_latent_train_subsets(store, cfg, dataset: str, seed: int, *, process: str, device: str, rank: int=0, world: int=1, replica: int=0, masks=None, ckpt_kind=K.SUBSET_CHECKPOINT, key_fn=None, progress_tag: Optional[str]=None, seed_offset: int=0, model_factory=None) -> dict:
    """Retrain one LoRA per KEEP mask; the mask source is a parameter.

    Default (``masks=None``): the dataset's LDS subset masks under
    ``SUBSET_CHECKPOINT`` — byte-identical to before the parameter existed.
    The AN-2 counterfactual chain passes its removal sets, a checkpoint kind
    and a key builder instead; everything else (per-item idempotent skip,
    rank/world sharding, one base materialisation per shard, progress CSV,
    seeded LoRA init) is shared, not copied.
    """
    pspec = _require_platform_process(cfg, dataset, process)
    spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=True)
    rkey = {'replica': replica} if replica else {}
    acfg = platform_cfg(cfg, dataset)
    proc, verified = platform_process(cfg, dataset, device=device)
    if masks is None:
        masks = store.load(K.SUBSET_MASKS, spec)
    if key_fn is None:
        key_fn = lambda m: {'m': m, **rkey}
    build = model_factory or (lambda: build_latent_model(store, cfg, dataset, None, device=device))
    pair = latent_train_pair(store, dataset)
    tasks = [m for m in range(len(masks)) if m % world == rank]
    rep_tag = f'_r{replica}' if replica else ''
    tag = progress_tag or f'subsets_{dataset}_{process}_seed{seed}{rep_tag}'
    prog = ProgressCSV(progress_path(cfg['storage']['data_root'], f'{tag}_rank{rank}of{world}'))
    log.info('LoRA-retraining %d/%d masks (rank %d/%d, replica %d, kind %s)', len(tasks), len(masks), rank, world, replica, ckpt_kind.value)
    model = None
    for i, m in enumerate(tasks):
        if store.exists(ckpt_kind, spec, **key_fn(m)):
            prog.log(done=i + 1, total=len(tasks), subset_m=m, final_loss='', skipped=1)
            continue
        if model is None:
            model = build()
        sub = pair.masked(masks[m])
        knobs = _train_knobs(acfg, len(sub))
        retrain_seed = seed + m + 1 + replica * 100003 + int(seed_offset)
        reset_lora_(model, retrain_seed)
        model, hist = train_model(process, sub, steps=knobs['steps'], batch_size=knobs['batch_size'], lr=knobs['lr'], seed=retrain_seed, num_classes=int(cfg['model']['num_classes']), p_uncond=0.0, device=device, compile_model=bool(acfg['compile']), model=model, weight_decay=knobs['weight_decay'], warmup_frac=knobs['warmup_frac'], amp_dtype=knobs['amp_dtype'], grad_accum=knobs['grad_accum'], process=proc, hflip=False)
        store.save(ckpt_kind, spec, _lora_checkpoint(model, acfg, step=knobs['steps'], process=process, platform=pspec['platform'], verified=verified), **key_fn(m))
        prog.log(done=i + 1, total=len(tasks), subset_m=m, final_loss=hist[-1]['loss'] if hist else '', skipped=0)
    return {'trained': len(tasks), 'replica': replica}
