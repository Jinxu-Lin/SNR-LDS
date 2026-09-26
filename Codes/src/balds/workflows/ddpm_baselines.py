"""Portable adapters for the paper's DDPM reference and four-checkpoint families.

Numerical kernels come from GradFeaturizer and the normal method registry. The
adapter only fixes the source image/query axes and keeps the retrained trajectory
separate from the imported DDPM. It never substitutes aggregate results for scores.
"""
from pathlib import Path

import numpy as np
import torch

from balds.artifacts.addressing import relpath
from balds.artifacts.das_archive import read_split_indices, resolve_root
from balds.artifacts.e3c import atomic, read_json
from balds.attribution import GradFeaturizer
from balds.attribution.embed import embed
from balds.data.base import ImageDataset, balanced_query_indices
from balds.models import build_model
from balds.schema.artifact import ArtifactKind as K
from balds.schema.estimator import FeatureSet
from balds.schema.registry import METHODS, PROCESSES
from balds.schema.runspec import RunSpec
from .common import _load_ds
from .container import Container
from .usecases import EvaluateUseCase, ScoreUseCase

STEPS = (2000, 4000, 6000, 8000)
CHECKPOINT_DATASET = 'cifar2_das_retrain_20260924'
REFERENCE_METHODS = {
    'embedding': ('pixel_dot', 'pixel_cos', 'clip_dot', 'clip_cos'),
    'gradient': ('trak_T100', 'grad_dot_T100', 'grad_cos_T100', 'relative_if_T100', 'renorm_if_T100'),
}
CHECKPOINT_METHODS = ('tracincp_T100', 'gas_T100')
FEATURE_RECIPE = dict(loss_type='mse', T=100, projection='cuda_jl', proj_dim=4096,
                      proj_seed=0, seed=42, batch_size=1, normalize=True, t_sampling='grid')


def checked(value, shape):
    value = torch.as_tensor(value)
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f'invalid tensor: {tuple(value.shape)}, expected finite {shape}')
    return value


def panels_from_manifest(path):
    """Resolve selected IDs against the response axis, preserving their labels."""
    panels = read_json(path)
    if isinstance(panels, dict):
        panels = panels['panels']
    selected, columns = {}, {}
    for track in ('gen', 'val'):
        matches = [p for p in panels if p['panel_id'] == f'cifar2_das_s42_{track}']
        if len(matches) != 1:
            raise ValueError(f'manifest must contain one cifar2_das_s42_{track} panel')
        panel = dict(matches[0])
        ids, response_ids = panel['query_ids'], panel['response_query_ids']
        if len(ids) != 100 or len(set(ids)) != 100 or len(set(response_ids)) != len(response_ids):
            raise ValueError('DDPM panels require 100 unique queries and unambiguous response IDs')
        if len(panel['train_ids']) != 5000 or len(set(panel['train_ids'])) != 5000:
            raise ValueError('DDPM panel must identify all 5000 training rows')
        if panel.get('n_subsets', 64) != 64:
            raise ValueError('paper DDPM panels require the fixed first 64 subset masks')
        try:
            columns[track] = [response_ids.index(q) for q in ids]
        except ValueError as exc:
            raise ValueError('selected query ID missing from response axis') from exc
        selected[track] = panel
    return selected, columns


def context(data_root, family):
    if family not in ('reference', 'checkpoints'):
        raise ValueError('family must be reference or checkpoints')
    c = Container({'storage.data_root': str(Path(data_root).resolve()),
                   'lds.M_by_dataset.cifar2_das': 64, 'lds.Q_by_dataset.cifar2_das': 100}, device='cpu')
    specs = {track: RunSpec(dataset='cifar2_das' if family == 'reference' else CHECKPOINT_DATASET,
                           process='ddpm', conditional=family == 'reference', seed=42,
                           query_type=track) for track in ('gen', 'val')}
    return c, specs


def datasets(c, columns):
    train, test = _load_ds(c.cfg, 'cifar2_das')
    if columns['val'] != list(balanced_query_indices(test.labels, 100)):
        raise ValueError('validation columns differ from the accepted balanced100 image selection')
    if columns['gen'] != list(range(100)):
        raise ValueError('generated queries must be the original first100 DDPM images')
    gen = c.store.load(K.GENERATION, RunSpec(dataset='cifar2_das', process='ddpm',
                                            conditional=True, seed=42))
    return {'train': train,
            'gen': ImageDataset(gen['samples'][columns['gen']], gen['labels'][columns['gen']]),
            'val': ImageDataset(test.images[columns['val']], test.labels[columns['val']])}


def relative_path(root, name):
    path, root = Path(name), Path(root).resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('artifact path must remain inside the selected data root')
    return str(resolved.relative_to(root))


def _identity(store, spec, recipe, **key):
    if store.exists(K.FEATURE_META, spec, **key):
        if store.load(K.FEATURE_META, spec, **key) != recipe:
            raise ValueError('existing feature identity differs; choose a clean artifact root')
    else:
        if any(store.exists(kind, candidate, **key) for kind, candidate in (
                (K.TRAIN_FEATURES, spec), (K.QUERY_FEATURES, spec),
                (K.QUERY_FEATURES, spec.with_(query_type='val')))):
            raise ValueError('existing feature values lack a compatible identity')
        store.save(K.FEATURE_META, spec, recipe, **key)


def features(data_root, manifest, *, family, line='gradient', device='cuda:0',
             steps=STEPS, checkpoint_root='results/ddpm_gaps_20260923/retrain_j2'):
    c, specs = context(data_root, family)
    panels, columns = panels_from_manifest(manifest)
    ds = datasets(c, columns)
    if line not in REFERENCE_METHODS:
        raise ValueError('line must be embedding or gradient')
    if family == 'checkpoints' and (not steps or any(step not in STEPS for step in steps)):
        raise ValueError(f'checkpoint feature steps must be a nonempty subset of {STEPS}')
    checkpoint_root = relative_path(data_root, checkpoint_root)
    idx_train, _ = read_split_indices(resolve_root(c.cfg['datasets']['raw_dirs']['das_archive']))
    jobs = [(s, 'trak_T100') for s in steps] if family == 'checkpoints' else [
        (None, name) for name in (('pixel', 'clip') if line == 'embedding' else ('trak_T100',))]
    produced = []
    for step, feat in jobs:
        key = {'feat': feat, **({'step': step} if step is not None else {})}
        dim = {'pixel': 3072, 'clip': 512, 'trak_T100': 4096}[feat]
        recipe = {'protocol': 'ddpm-paper-baselines-v1', 'family': family, 'feat': feat,
                  'original_train_indices': list(map(int, idx_train)),
                  'query_ids': {t: panels[t]['query_ids'] for t in panels}, 'query_columns': columns}
        if feat == 'trak_T100':
            recipe.update(FEATURE_RECIPE)
        else:
            recipe.update(dim=dim, encoder='openai/clip-vit-base-patch32' if feat == 'clip' else 'pixel [0,1]',
                          source_train=f'featurize/{feat}/cifar10_v2/seed_42/train_features.pt')
        if step is not None:
            recipe.update(checkpoint=f'{checkpoint_root}/step_{step}.pt', step=step)
        _identity(c.store, specs['gen'], recipe, **key)
        fz = None
        for split in ('train', 'gen', 'val'):
            spec = specs['val' if split == 'val' else 'gen']
            kind = K.TRAIN_FEATURES if split == 'train' else K.QUERY_FEATURES
            shape = (5000 if split == 'train' else 100, dim)
            if c.store.exists(kind, spec, **key):
                checked(c.store.load(kind, spec, **key), shape)
                continue
            if feat != 'trak_T100':
                if split == 'train':
                    source = torch.load(Path(data_root) / recipe['source_train'], map_location='cpu', weights_only=False)
                    value = checked(source, (50000, dim))[idx_train].clone()
                    if feat == 'pixel' and not torch.equal(value, embed('pixel', ds['train'].images, device='cpu')):
                        raise ValueError('pixel embedding training rows disagree with the DAS archive order')
                else:
                    value = embed(feat, ds[split].images, device='cpu', batch_size=16)
            else:
                if fz is None:
                    ckpt = (c.store.load(K.CHECKPOINT, spec) if step is None else
                            torch.load(Path(data_root) / recipe['checkpoint'], map_location='cpu', weights_only=False))
                    if step is not None and (ckpt['step'] != step or ckpt['conditional'] is not False):
                        raise ValueError('checkpoint does not identify the selected unconditional retrained step')
                    kwargs = {k: v for k, v in FEATURE_RECIPE.items() if k != 'seed'}
                    fz = GradFeaturizer(build_model(ckpt, device=device), PROCESSES.get('ddpm'), device=device, **kwargs)
                value = fz.featurize(ds[split], seed=42, log_every=50)
            c.store.save(kind, spec, checked(value, shape), **key)
            produced.append(relpath(kind, spec, **key))
        del fz
    return {'family': family, 'produced': produced, 'steps': list(steps) if family == 'checkpoints' else None}


class QueryGTView:
    """Select exact archived response columns without replacing the full matrix."""
    def __init__(self, store, columns, panels):
        self.store, self.columns, self.panels = store, columns, panels

    def __getattr__(self, name):
        return getattr(self.store, name)

    def load(self, kind, spec, **key):
        value = self.store.load(kind, spec, **key)
        if kind == K.GT_MATRIX:
            expected = len(self.panels[spec.query_type]['response_query_ids'])
            if value.shape[1] != expected:
                raise ValueError('archived response width disagrees with manifest response IDs')
            return value[:, self.columns[spec.query_type]]
        return value

    def save(self, kind, spec, value, **key):
        if kind == K.SCORES_META:
            value = dict(value, query_indices=self.columns[spec.query_type],
                         query_ids=self.panels[spec.query_type]['query_ids'], n_subsets=64,
                         protocol='ddpm-paper-baselines-v1')
        return self.store.save(kind, spec, value, **key)


def checkpoint_scores(train, query, method, *, steps=STEPS):
    """Uniform mean over the four matched checkpoints, with no final-only fallback."""
    if tuple(steps) != STEPS or len(train) != 4 or len(query) != 4:
        raise ValueError("all four ordered checkpoint pairs (2000,4000,6000,8000) are required")
    if method not in CHECKPOINT_METHODS:
        raise ValueError("checkpoint family only supports TracInCP and GAS")
    for g, q in zip(train, query):
        if g.ndim != 2 or q.ndim != 2 or g.shape[1] != q.shape[1]:
            raise ValueError("checkpoint train/query feature axes disagree")
        if g.shape != train[0].shape or q.shape != query[0].shape:
            raise ValueError("checkpoint pairs must retain the same image axes")
        if not torch.isfinite(g).all() or not torch.isfinite(q).all():
            raise ValueError("checkpoint features must be finite")
    feats = FeatureSet(train[-1], None, 'trak_T100', ckpt_grads=train,
                       ckpt_query=query, ckpt_steps=STEPS)
    return METHODS.get(method).score(feats, query[-1], 0., device='cpu')


def score(data_root, manifest, output, *, family, line='gradient',
          checkpoint_root='results/ddpm_gaps_20260923/retrain_j2'):
    c, specs = context(data_root, family)
    panels, columns = panels_from_manifest(manifest)
    checkpoint_root = relative_path(data_root, checkpoint_root)
    methods = CHECKPOINT_METHODS if family == 'checkpoints' else REFERENCE_METHODS[line]
    view = QueryGTView(c.store, columns, panels)
    if family == 'checkpoints':
        for step in STEPS:
            meta = c.store.load(K.FEATURE_META, specs['gen'], feat='trak_T100', step=step)
            expected_feature = dict(FEATURE_RECIPE, step=step,
                                    query_ids={t: panels[t]['query_ids'] for t in panels})
            if any(meta.get(k) != v for k, v in expected_feature.items()):
                raise ValueError('checkpoint feature metadata disagrees with the requested recipe/query IDs')
    train = ([checked(c.store.load(K.TRAIN_FEATURES, specs['gen'], feat='trak_T100', step=step),
                      (5000, 4096)) for step in STEPS] if family == 'checkpoints' else None)
    for track, panel in panels.items():
        if family == 'checkpoints':
            query = [checked(c.store.load(K.QUERY_FEATURES, specs[track], feat='trak_T100', step=step),
                             (100, 4096)) for step in STEPS]
        items = []
        for method in methods:
            spec = specs[track].with_(method=method)
            expected = {'protocol': 'ddpm-paper-baselines-v1', 'family': family,
                        'query_ids': panel['query_ids'], 'query_indices': columns[track],
                        'method': method, 'source_dataset': 'cifar2_das'}
            if family == 'checkpoints':
                expected.update(checkpoint_steps=list(STEPS), aggregation='uniform_checkpoint_mean',
                                checkpoint_paths=[f'{checkpoint_root}/step_{step}.pt' for step in STEPS],
                                retrained_trajectory=True)
            if c.store.exists(K.SCORES, spec):
                if not c.store.exists(K.SCORES_META, spec):
                    raise ValueError('existing score lacks its scientific identity')
                meta = c.store.load(K.SCORES_META, spec)
                if any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError('existing score identity differs; choose a clean artifact root')
            else:
                if family == 'checkpoints':
                    values = checkpoint_scores(train, query, method)
                    c.store.save(K.SCORES, spec, checked(values, (5000, 100)).numpy())
                    meta = expected
                else:
                    ScoreUseCase(view, c.cfg, device='cpu').run(method, 'cifar2_das', 42,
                        query_type=track, process='ddpm', conditional=True)
                    meta = {**c.store.load(K.SCORES_META, spec), **expected}
                c.store.save(K.SCORES_META, spec, meta)
            checked(c.store.load(K.SCORES, spec), (5000, 100))
            if family == 'reference':
                metrics = EvaluateUseCase(view, c.cfg).run(method, 'cifar2_das', 42,
                    query_type=track, process='ddpm', conditional=True)
                atomic(Path(output) / 'full' / f'{method}_{track}.json', metrics)
            items.append({'method': method, 'scores': relpath(K.SCORES, spec),
                          'train_ids': panel['train_ids'], 'query_ids': panel['query_ids'],
                          'score_space': 'native', 'identity': meta})
        panel['methods'] = items
        for key in ('masks', 'response'):
            panel[key] = relative_path(data_root, panel[key])
    path = Path(output) / 'benchmark_manifest.json'
    atomic(path, list(panels.values()))
    return {'family': family, 'methods': list(methods), 'manifest': str(path),
            'evaluation': 'run balds batch on the produced manifest for SNR-LDS'}
