"""Application orchestration for the contamination-injection benchmark."""
from __future__ import annotations
import csv
import hashlib
import io
import logging
import time
import numpy as np
import torch
from balds.schema.artifact import ArtifactKind as K
from balds.schema.identity import canonical_sha256
from balds.schema.runspec import RunSpec
from balds.data.inject import detector_training_data
from balds.evaluation.detector import predict_proba, predictor_for, pretrained_weights_info, screen_probabilities, train_detector
from .common import _load_ds
log = logging.getLogger(__name__)
class InjectUseCase:
    """Build and file deterministic injected-dataset metadata."""

    def __init__(self, store, cfg, *, device: str='cuda:0') -> None:
        self.store, self.cfg = (store, cfg)
        self.device = device

    @staticmethod
    def _validate_meta(value, dataset: str) -> None:
        expected = dataset
        if value.get('dataset') != expected:
            raise ValueError('injection metadata dataset identity mismatch')
        if len(value.get('pairs', {})) != 10:
            raise ValueError('injection metadata must contain ten host/concept pairs')
        if len(str(value.get('train_sha256', ''))) != 64:
            raise ValueError('injection metadata lacks a train_sha256')

    def build(self, dataset: str, *, force: bool=False) -> dict:
        inject_cfg = self.cfg.get('inject', {}) or {}
        if dataset not in inject_cfg:
            raise ValueError(f'no inject dataset config for {dataset!r}')
        train, _test = _load_ds(self.cfg, dataset)
        meta = getattr(train, 'inject_meta', None)
        if not isinstance(meta, dict):
            raise ValueError(f'dataset {dataset!r} did not provide inject metadata')
        self._validate_meta(meta, dataset)
        spec = RunSpec(dataset=dataset)
        if self.store.exists(K.INJECT_META, spec) and (not force):
            existing = self.store.load(K.INJECT_META, spec)
            if existing == meta:
                return self._summary(meta, skipped=True)
            raise FileExistsError(f'refusing to overwrite differing inject_meta for {dataset!r}; use --force')
        self.store.save_atomic(K.INJECT_META, spec, meta, replace=force, validator=lambda value: self._validate_meta(value, dataset))
        return self._summary(meta, skipped=False)

    @staticmethod
    def _summary(meta: dict, *, skipped: bool) -> dict:
        pairs = []
        for host_name, row in sorted(meta['pairs'].items(), key=lambda item: item[1]['host_label']):
            if 'concept_name' in row:
                pairs.append({'host': host_name, 'concept': row['concept_name'], 'tier': row['tier'], 'replaced': len(row['replaced_cifar10_indices']), 'injected': len(row['injected_cifar100_train_ids']), 'detector_train': len(row['detector_cifar100_train_ids']), 'detector_test': len(row['detector_cifar100_test_ids'])})
            else:
                pairs.append({'host': host_name, 'concept': row['foreign_style'], 'tier': row['tier'], 'replaced': len(row['foreign_rows']), 'injected': len(row['foreign_ids']), 'dedup_count': int(row['dedup_count'])})
        return {'dataset': meta['dataset'], 'builder_version': meta['builder_version'], 'train_sha256': meta['train_sha256'], 'pairs': pairs, 'skipped': skipped}

    @staticmethod
    def _tensor_sha256(images: torch.Tensor, labels: torch.Tensor) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(images.cpu().numpy(), dtype=np.uint8).tobytes())
        digest.update(np.ascontiguousarray(labels.cpu().numpy(), dtype=np.int64).tobytes())
        return digest.hexdigest()

    def detector_train(self, dataset: str, *, force: bool=False, detector_tag: str | None=None, detector_config: str='detector') -> dict:
        """Train and file a detector.

        CIFAR recipes come from ``inject.<detector_config>`` (``arch``,
        ``data_mode`` isolated|all_real, ``holdout_frac``, ...). Anything but the
        legacy recipe (``detector`` block, isolated data, ResNet-18) must carry a
        ``detector_tag``, so ``results/inject/<ds>/detector.pt`` is never
        overwritten; a tagged detector is filed at ``detector_<tag>.pt``.
        """
        spec = RunSpec(dataset=dataset)
        det_key = {'tag': detector_tag} if detector_tag else {}
        detector_cfg = dict(self.cfg['inject'][detector_config])
        data_mode = str(detector_cfg.get('data_mode', 'isolated'))
        arch = str(detector_cfg.get('arch', 'resnet18_cifar'))
        legacy = detector_config == 'detector' and data_mode == 'isolated' and (arch == 'resnet18_cifar')
        if not legacy and (not detector_tag):
            raise ValueError(f'inject.{detector_config} (arch={arch}, data_mode={data_mode}) needs --detector-tag, so results/inject/{dataset}/detector.pt is never overwritten')
        if self.store.exists(K.INJECT_DETECTOR, spec, **det_key) and (not force):
            return {'dataset': dataset, 'detector_tag': detector_tag, 'skipped': True}
        meta = self.store.load(K.INJECT_META, spec)
        extra: dict = {}
        from balds.data.inject import DETECTOR_DATA_MODES
        if data_mode not in DETECTOR_DATA_MODES:
            raise ValueError(f'unknown detector data_mode {data_mode!r}; choose from {sorted(DETECTOR_DATA_MODES)}')
        inject_spec = self.cfg['inject'][dataset]
        from .common import _raw_dir
        images, labels, class_names, source = DETECTOR_DATA_MODES[data_mode](_raw_dir(self.cfg, dataset), inject_spec, meta)
        seed = int(detector_cfg.get('seed', 0))
        state, report = train_detector(images, labels, num_classes=20, cfg=detector_cfg, device=self.device, seed=seed)
        train_sha = self._tensor_sha256(images, labels)
        embedding_model = None
        if detector_tag:
            pretrained = pretrained_weights_info(arch, detector_cfg.get('weights'))
            if pretrained is not None:
                from balds.artifacts import sha256_file
                pretrained['sha256'] = sha256_file(pretrained['path'])
            extra = {'detector_tag': detector_tag, 'detector_config': detector_config, 'data_mode': data_mode, 'pretrained': pretrained}
        payload = {'state_dict': state, 'report': report, 'class_names': class_names, 'pairs_sha256': canonical_sha256(meta['pairs']), 'train_set_sha256': train_sha, 'config': detector_cfg, 'source': source, 'arch': arch, 'embedding_model': embedding_model, 'code_version': getattr(self.store, 'code_version', 'dev'), **extra}
        self.store.save_atomic(K.INJECT_DETECTOR, spec, payload, replace=force, **det_key)
        return {'dataset': dataset, 'detector_tag': detector_tag, 'n': len(labels), 'class_names': class_names, 'train_set_sha256': train_sha, 'report': report, 'detector': self.store.describe(K.INJECT_DETECTOR, spec, **det_key)['relpath'], 'skipped': False}
    _BARE_CROSS_SCREENS = frozenset({('cifar10_v2', 'cifar10_inj4')})

    @classmethod
    def _candidates_detector_key(cls, dataset: str, detector_dataset: str) -> dict:
        """``{}`` (bare address) for a self-screen or a legacy cross-screen, else
        ``{"detector": detector_dataset}``, so the address never depends on run order."""
        if detector_dataset == dataset or (dataset, detector_dataset) in cls._BARE_CROSS_SCREENS:
            return {}
        return {'detector': detector_dataset}

    def screen(self, dataset: str, seed: int, *, process: str, tag: str, detector_dataset: str | None=None, threshold: float | None=None, conditional: bool=True, cf: str | None=None, detector_tag: str | None=None) -> dict:
        threshold = float(threshold if threshold is not None else self.cfg['inject'].get('screen_threshold', 0.9))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('screen threshold must lie in [0,1]')
        pool_spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        detector_dataset = detector_dataset or dataset
        detector_spec = RunSpec(dataset=detector_dataset)
        pool_key = {'tag': tag, **({'cf': cf} if cf else {})}
        candidates_key = {**pool_key, **self._candidates_detector_key(dataset, detector_dataset)}
        det_key = {'tag': detector_tag} if detector_tag else {}
        if detector_tag:
            candidates_key['detector_tag'] = detector_tag
        pool = self.store.load(K.GEN_POOL, pool_spec, **pool_key)
        detector = self.store.load(K.INJECT_DETECTOR, detector_spec, **det_key)
        batch = int(self.cfg['inject']['pool'].get('batch_size', 512))
        if detector.get('arch', 'resnet18_cifar') == 'clip_linear':
            from balds.evaluation.clip_probe import predict_linear_probe
            from balds.attribution.embed import ClipEmbedder
            if detector.get('embedding_model') != ClipEmbedder.model_id:
                raise ValueError('CLIP detector embedding-model identity mismatch')
            images = torch.as_tensor(pool['images_u8']).float().div(127.5).sub(1.0)
            embeddings = ClipEmbedder()(images, device=self.device, batch_size=batch)
            probabilities = predict_linear_probe(detector['state_dict'], embeddings, batch=batch, device=self.device)
        else:
            probabilities = predict_proba(detector['state_dict'], pool['images_u8'], batch=batch, device=self.device, **self._predict_kwargs(detector))
        result = screen_probabilities(probabilities, pool['labels'], threshold=threshold, num_hosts=10)
        detector_sha = self.store.describe(K.INJECT_DETECTOR, detector_spec, **det_key)['sha256']
        pool_sha = self.store.describe(K.GEN_POOL, pool_spec, **pool_key)['sha256']
        payload = {'threshold': threshold, 'detector_sha256': detector_sha, 'pool_sha256': pool_sha, **result}
        if self.store.exists(K.INJECT_CANDIDATES, pool_spec, **candidates_key):
            existing = self.store.load(K.INJECT_CANDIDATES, pool_spec, **candidates_key)
            if existing == payload:
                return {**payload, 'skipped': True, 'candidates': self.store.describe(K.INJECT_CANDIDATES, pool_spec, **candidates_key)['relpath']}
            raise FileExistsError('candidate artifact exists with different inputs or threshold')
        self.store.save_atomic(K.INJECT_CANDIDATES, pool_spec, payload, upstream=(detector_sha, pool_sha), **candidates_key)
        return {**payload, 'skipped': False, 'candidates': self.store.describe(K.INJECT_CANDIDATES, pool_spec, **candidates_key)['relpath']}

    @staticmethod
    def _predict_kwargs(detector: dict) -> dict:
        """``predict_proba`` arguments for a non-legacy arch (none for ResNet-18)."""
        arch = str(detector.get('arch', 'resnet18_cifar'))
        if arch == 'resnet18_cifar':
            return {}
        cfg = detector.get('config') or {}
        return {'arch': arch, 'input_size': int(cfg.get('input_size', 224)), 'amp': bool(cfg.get('amp', False))}

    @staticmethod
    def _sheet(images_u8: torch.Tensor, captions: list[str], title: str) -> bytes:
        from PIL import Image, ImageDraw
        native = int(images_u8.shape[-1])
        cell, cols, rows, title_h = (256, 5, 5, 24) if native >= 256 else (128, 10, 10, 24)
        canvas = Image.new('RGB', (cols * cell, title_h + rows * cell), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), title, fill='black')
        capacity = cols * rows
        for index, (tensor, caption) in enumerate(zip(images_u8[:capacity], captions[:capacity])):
            array = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8, copy=False)
            image = Image.fromarray(array).resize((cell, cell), resample=Image.Resampling.NEAREST)
            left, top = (index % cols * cell, title_h + index // cols * cell)
            canvas.paste(image, (left, top))
            draw.rectangle((left, top, left + cell - 1, top + 13), fill='black')
            draw.text((left + 2, top + 1), caption, fill='white')
        output = io.BytesIO()
        canvas.save(output, format='PNG')
        return output.getvalue()

    @staticmethod
    def _sheet_capacity(images_u8: torch.Tensor) -> int:
        return 25 if int(images_u8.shape[-1]) >= 256 else 100

    def review_export(self, dataset: str, seed: int, *, process: str, tag: str | None, per_concept: int | None=None, random_n: int | None=None, conditional: bool=True, mine_tag: str | None=None) -> dict:
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        if mine_tag:
            if random_n is not None:
                raise ValueError('--random samples a pool; a mined set is reviewed in full')
            return self._review_export_mined(spec, dataset, mine_tag, per_concept)
        pool = self.store.load(K.GEN_POOL, spec, tag=tag)
        if random_n is not None:
            n = min(int(random_n), int(pool['n']))
            if n <= 0:
                raise ValueError('--random must be positive')
            rng = np.random.RandomState(0)
            selected = np.sort(rng.choice(int(pool['n']), n, replace=False))
            sheets = 0
            sheet_capacity = self._sheet_capacity(pool['images_u8'])
            for start in range(0, n, sheet_capacity):
                ids = selected[start:start + sheet_capacity]
                png = self._sheet(pool['images_u8'][ids], [f'pool {int(i)}' for i in ids], 'random pool sanity')
                self.store.save(K.INJECT_REVIEW_SHEET, spec, png, tag=tag, host='random', sheet=sheets)
                sheets += 1
            return {'random': n, 'sheets': sheets, 'csv': None}
        candidates = self.store.load(K.INJECT_CANDIDATES, spec, tag=tag)
        meta = self.store.load(K.INJECT_META, RunSpec(dataset=dataset))
        maximum = int(self.cfg['inject']['review'].get('per_concept_cap', 150))
        cap = int(per_concept if per_concept is not None else maximum)
        if cap <= 0 or cap > maximum:
            raise ValueError(f'--per-concept must lie in [1,{maximum}]')
        rows = []
        sheet_count = 0
        for host in range(10):
            paired = [row for row in candidates['flagged'] if int(row['host_label']) == host and int(row['pred_concept_host']) == host]
            paired.sort(key=lambda row: (-float(row['prob']), int(row['pool_index'])))
            paired = paired[:cap]
            host_name = next((name for name, row in meta['pairs'].items() if int(row['host_label']) == host))
            pair = meta['pairs'][host_name]
            concept = pair.get('concept_name', pair.get('foreign_style'))
            sheet_capacity = self._sheet_capacity(pool['images_u8'])
            for sheet_index, start in enumerate(range(0, len(paired), sheet_capacity)):
                group = paired[start:start + sheet_capacity]
                ids = [int(row['pool_index']) for row in group]
                filename = f'sheet_{host}_{sheet_index}.png'
                png = self._sheet(pool['images_u8'][ids], [f"{cell}: {float(row['prob']):.3f}" for cell, row in enumerate(group)], f'{host_name}←{concept}')
                self.store.save(K.INJECT_REVIEW_SHEET, spec, png, tag=tag, host=host, sheet=sheet_index)
                sheet_count += 1
                for cell, row in enumerate(group):
                    rows.append({'sheet': filename, 'cell': cell, 'pool_index': int(row['pool_index']), 'host_label': host, 'host_name': host_name, 'concept': concept, 'prob': f"{float(row['prob']):.9g}", 'decision': ''})
        output = io.StringIO(newline='')
        fields = ['sheet', 'cell', 'pool_index', 'host_label', 'host_name', 'concept', 'prob', 'decision']
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        self.store.save(K.INJECT_REVIEW_TABLE, spec, output.getvalue(), tag=tag)
        return {'rows': len(rows), 'sheets': sheet_count, 'review': self.store.describe(K.INJECT_REVIEW_TABLE, spec, tag=tag)['relpath']}

    def queries(self, dataset: str, seed: int, *, process: str, tag: str | None, review: str, version: str, reviewer: str | None=None, conditional: bool=True, force: bool=False, mine_tag: str | None=None) -> dict:
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        if self.store.exists(K.INJECT_QUERIES, spec) and (not force):
            raise FileExistsError('inject queries already exist; use --force for a new review')
        review_rows = self.store.load_external(review)
        review_desc = self.store.describe_external(review)
        allowed = {'', 'accept', 'reject', 'unsure'}
        seen = set()
        accepted = []
        reviewed_counts = {str(host): 0 for host in range(10)}
        accepted_counts = {str(host): 0 for host in range(10)}
        extra: dict = {}
        if mine_tag:
            mined = self.store.load(K.INJECT_MINED, spec, tag=mine_tag)
            mined_sha = self.store.describe(K.INJECT_MINED, spec, tag=mine_tag)['sha256']
            n_mined = len(mined['sources'])
            for row in review_rows:
                decision = str(row.get('decision', '')).strip().lower()
                if decision not in allowed:
                    raise ValueError(f"unknown review decision {row.get('decision')!r}")
                index, host = (int(row['candidate_index']), int(row['host_label']))
                if host not in range(10):
                    raise ValueError(f'review host_label must be in [0,9], got {host}')
                if index < 0 or index >= n_mined:
                    raise ValueError('review CSV contains a candidate index outside the mined set')
                if str(row.get('candidate_id', '')) != str(mined['sources'][index]) or int(mined['host_labels'][index]) != host:
                    raise ValueError('review CSV row does not match its mined candidate')
                if index in seen:
                    raise ValueError(f'duplicate candidate_index {index} in review CSV')
                seen.add(index)
                reviewed_counts[str(host)] += 1
                if decision == 'accept':
                    accepted.append((host, index, float(mined['probs'][index, 10 + host])))
                    accepted_counts[str(host)] += 1
            images_source = mined['images_u8']
            detector_sha = mined['recipe']['detector_sha256']
            pool_tag = f'mine:{mine_tag}'
            upstream = (review_desc['sha256'], detector_sha, mined_sha)
        else:
            pool = self.store.load(K.GEN_POOL, spec, tag=tag)
            candidate = self.store.load(K.INJECT_CANDIDATES, spec, tag=tag)
            candidate_rows = {int(row['pool_index']): row for row in candidate['flagged']}
            for row in review_rows:
                decision = str(row.get('decision', '')).strip().lower()
                if decision not in allowed:
                    raise ValueError(f"unknown review decision {row.get('decision')!r}")
                pool_index, host = (int(row['pool_index']), int(row['host_label']))
                if host not in range(10):
                    raise ValueError(f'review host_label must be in [0,9], got {host}')
                if pool_index < 0 or pool_index >= int(pool['n']):
                    raise ValueError('review CSV contains a pool index outside the selected pool')
                source = candidate_rows.get(pool_index)
                if source is None or int(source['host_label']) != host or int(source['pred_concept_host']) != host or (int(pool['labels'][pool_index]) != host):
                    raise ValueError('review CSV row does not match a paired screened candidate')
                if pool_index in seen:
                    raise ValueError(f'duplicate pool_index {pool_index} in review CSV')
                seen.add(pool_index)
                reviewed_counts[str(host)] += 1
                if decision == 'accept':
                    accepted.append((host, pool_index, float(row['prob'])))
                    accepted_counts[str(host)] += 1
            images_source = pool['images_u8']
            detector_sha = candidate['detector_sha256']
            pool_tag = tag
            upstream = (review_desc['sha256'], candidate['detector_sha256'], candidate['pool_sha256'])
        accepted.sort(key=lambda value: (value[0], value[1]))
        host_labels = torch.tensor([host for host, _index, _prob in accepted], dtype=torch.long)
        pool_indices = torch.tensor([index for _host, index, _prob in accepted], dtype=torch.long)
        probs = torch.tensor([prob for _host, _index, prob in accepted], dtype=torch.float32)
        images = images_source[pool_indices] if len(pool_indices) else torch.empty((0, *tuple(images_source.shape[1:])), dtype=torch.uint8)
        if mine_tag:
            extra = {'mine_tag': mine_tag, 'candidate_ids': [str(mined['sources'][int(i)]) for i in pool_indices]}
        split = torch.ones(len(accepted), dtype=torch.int8)
        rng = np.random.RandomState(int(self.cfg['inject']['split'].get('seed', 0)))
        val_frac = float(self.cfg['inject']['split'].get('val_frac', 0.2))
        for host in range(10):
            rows = torch.nonzero(host_labels == host, as_tuple=False).flatten().numpy()
            n_val = int(round(val_frac * len(rows)))
            if n_val:
                split[rng.choice(rows, n_val, replace=False)] = 0
        query_sha = self._tensor_sha256(images, host_labels)
        payload = {'images_u8': images, 'host_labels': host_labels, 'concept_hosts': host_labels.clone(), 'pool_indices': pool_indices, 'probs': probs, 'split': split, 'review_version': str(version), 'reviewer': reviewer, 'review_csv_sha256': review_desc['sha256'], 'detector_sha256': detector_sha, 'pool_tag': pool_tag, 'n_reviewed_per_concept': reviewed_counts, 'n_accepted_per_concept': accepted_counts, 'queries_sha256': query_sha, 'code_version': getattr(self.store, 'code_version', 'dev'), **extra}
        self.store.save_atomic(K.INJECT_QUERIES, spec, payload, replace=force, upstream=upstream)
        return {'Q': len(accepted), 'val': int((split == 0).sum()), 'test': int((split == 1).sum()), 'queries_sha256': query_sha}
    _MINE_IDENTITY = ('dataset', 'process', 'seed', 'detector_dataset', 'detector_tag', 'detector_sha256', 'checkpoint_sha256', 'sampler', 'steps', 'eta', 'gen_seed', 'noise_key', 'keep_rule', 'chunk', 'include_pool')

    @staticmethod
    def _validate_mined(value) -> None:
        n = len(value['sources'])
        if tuple(value['images_u8'].shape) != (n, 3, 32, 32) or len(value['host_labels']) != n or tuple(value['probs'].shape) != (n, 20):
            raise ValueError('mined set arrays are misaligned')

    def _host_classes(self, dataset: str, classes) -> tuple[list[int], dict[int, str]]:
        meta = self.store.load(K.INJECT_META, RunSpec(dataset=dataset))
        names = {int(row['host_label']): name for name, row in meta['pairs'].items()}
        if classes is None or classes == '':
            order = sorted(names)
        else:
            tokens = classes.split(',') if isinstance(classes, str) else list(classes)
            by_name = {name: label for label, name in names.items()}
            order = []
            for token in (str(v).strip() for v in tokens):
                if token.lstrip('-').isdigit() and int(token) in names:
                    order.append(int(token))
                elif token in by_name:
                    order.append(by_name[token])
                else:
                    raise ValueError(f'unknown host class {token!r}; use a label 0-9 or {sorted(by_name)}')
            if len(set(order)) != len(order):
                raise ValueError('--classes lists a class twice')
        return (order, names)

    def mine(self, dataset: str, seed: int, *, process: str, mine_tag: str, detector_tag: str | None=None, detector_dataset: str | None=None, target: int | None=None, chunk: int | None=None, keep_threshold: float | None=None, max_per_class: int | None=None, include_pool: str | None=None, classes=None, batch: int | None=None, conditional: bool=True) -> dict:
        """Class-by-class mining of "wrong" generations.

        For each class in order: skip it once it holds ``target`` kept images;
        otherwise generate ``chunk`` images with that class label, keep those whose
        detector probability for class ``10 + c`` reaches ``keep_threshold``, and
        persist the kept set and progress atomically; stop at ``target`` or after
        ``max_per_class`` generated images (status incomplete). Rejected images are
        never written. Noise is keyed by class and row under the mining gen_seed, so
        an interrupted run resumes at its next chunk and ends identical to an
        uninterrupted one. ``include_pool`` first screens a filed pool of the same
        checkpoint and recipe with the same detector.
        """
        from balds.models import DDPMSchedule, build_model
        from .common import _mine_noise, _sample_pool_rows
        mcfg = dict(self.cfg['inject'].get('mine') or {})
        target = int(target if target is not None else mcfg.get('target', 150))
        chunk = int(chunk if chunk is not None else mcfg.get('chunk', 5000))
        keep_threshold = float(keep_threshold if keep_threshold is not None else mcfg.get('keep_threshold', 0.5))
        max_per_class = int(max_per_class if max_per_class is not None else mcfg.get('max_per_class', 300000))
        batch = int(batch if batch is not None else mcfg.get('batch_size', 512))
        if target <= 0 or chunk <= 0 or max_per_class <= 0 or (batch <= 0):
            raise ValueError('--target, --chunk, --max-per-class and --batch must be positive')
        if not 0.0 <= keep_threshold <= 1.0:
            raise ValueError('--keep-threshold must lie in [0,1]')
        if not conditional:
            raise ValueError('mining generates class by class and needs a class-conditional model')
        sampler = str(mcfg.get('sampler', 'ddim'))
        steps = int(mcfg.get('steps', 50))
        eta = float(mcfg.get('eta', 0.0))
        gen_seed = int(mcfg.get('gen_seed', 20260915))
        order, names = self._host_classes(dataset, classes)
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=True)
        from balds.artifacts.addressing import relpath
        rel = relpath(K.INJECT_MINED, spec, tag=mine_tag)
        detector_dataset = detector_dataset or dataset
        det_spec = RunSpec(dataset=detector_dataset)
        det_key = {'tag': detector_tag} if detector_tag else {}
        detector = self.store.load(K.INJECT_DETECTOR, det_spec, **det_key)
        if str(detector.get('arch', 'resnet18_cifar')) == 'clip_linear':
            raise ValueError('mining screens 32 px CIFAR generations; a CLIP detector does not apply')
        detector_sha = self.store.describe(K.INJECT_DETECTOR, det_spec, **det_key)['sha256']
        checkpoint_sha = self.store.describe(K.CHECKPOINT, spec)['sha256']
        recipe = {'dataset': dataset, 'process': process, 'seed': int(seed), 'detector_dataset': detector_dataset, 'detector_tag': detector_tag, 'detector_sha256': detector_sha, 'checkpoint_sha256': checkpoint_sha, 'sampler': sampler, 'steps': steps, 'eta': eta, 'gen_seed': gen_seed, 'noise_key': '(class + 1) * 10**9 + row', 'keep_rule': {'probability': 'p[10 + class]', 'threshold': keep_threshold}, 'chunk': chunk, 'include_pool': include_pool}
        exists = self.store.exists(K.INJECT_MINED, spec, tag=mine_tag)
        if exists:
            state = self.store.load(K.INJECT_MINED, spec, tag=mine_tag)
            differing = [key for key in self._MINE_IDENTITY if state['recipe'].get(key) != recipe[key]]
            if differing:
                raise FileExistsError(f'mining tag {mine_tag!r} exists with a different {differing}; use a new --mine-tag')
        else:
            state = {'recipe': recipe, 'images_u8': torch.empty((0, 3, 32, 32), dtype=torch.uint8), 'host_labels': torch.empty(0, dtype=torch.long), 'probs': torch.empty((0, 20), dtype=torch.float32), 'sources': [], 'progress': {}, 'pool': None}

        def append(images, probs, host, sources):
            state['images_u8'] = torch.cat([state['images_u8'], images.to(torch.uint8)])
            state['host_labels'] = torch.cat([state['host_labels'], torch.full((len(sources),), int(host), dtype=torch.long)])
            state['probs'] = torch.cat([state['probs'], probs.to(torch.float32)])
            state['sources'] = list(state['sources']) + list(sources)

        def persist():
            nonlocal exists
            payload = {**state, 'target': target, 'max_per_class': max_per_class, 'classes': order, 'code_version': getattr(self.store, 'code_version', 'dev')}
            self.store.save_atomic(K.INJECT_MINED, spec, payload, replace=exists, upstream=(detector_sha, checkpoint_sha), validator=self._validate_mined, tag=mine_tag)
            exists = True
        predictor = predictor_for(detector, device=self.device)
        if include_pool and state['pool'] is None:
            pool = self.store.load(K.GEN_POOL, spec, tag=include_pool)
            pool_recipe = (pool.get('checkpoint_sha256'), pool.get('sampler'), int(pool.get('steps', -1)), float(pool.get('eta', -1.0)))
            if pool_recipe != (checkpoint_sha, sampler, steps, eta):
                raise ValueError(f"pool {include_pool!r} was generated with checkpoint/sampler/steps/eta {pool_recipe}, not this mining recipe's {(checkpoint_sha, sampler, steps, eta)}")
            pool_probs = predictor(pool['images_u8'], batch=batch)
            pool_labels = torch.as_tensor(pool['labels'], dtype=torch.long)
            kept_per_class = {}
            for host in range(10):
                rows = torch.nonzero((pool_labels == host) & (pool_probs[:, 10 + host] >= keep_threshold), as_tuple=False).flatten()
                append(pool['images_u8'][rows], pool_probs[rows], host, [f'pool:{include_pool}:{int(r)}' for r in rows])
                kept_per_class[str(host)] = int(len(rows))
            state['pool'] = {'tag': include_pool, 'n': int(pool['n']), 'sha256': self.store.describe(K.GEN_POOL, spec, tag=include_pool)['sha256'], 'gen_seed': pool.get('gen_seed'), 'kept_per_class': kept_per_class}
            persist()
            log.info('mine %s: screened pool %s (%d images), kept per class %s', mine_tag, include_pool, int(pool['n']), kept_per_class)
        model = schedule = None
        for host in order:
            progress = state['progress'].setdefault(str(host), {'chunks': 0, 'generated': 0, 'kept_mined': 0, 'status': 'pending'})
            while True:
                kept_total = int((state['host_labels'] == host).sum())
                if kept_total >= target:
                    status = 'done'
                elif progress['generated'] >= max_per_class:
                    status = 'incomplete'
                else:
                    status = None
                if status is not None:
                    if progress['status'] != status:
                        progress['status'] = status
                        persist()
                    break
                if model is None:
                    model = build_model(self.store.load(K.CHECKPOINT, spec), device=self.device)
                    if process == 'ddpm':
                        schedule = DDPMSchedule(T=self.cfg.get('ddpm', {}).get('T', 1000)).to(self.device)
                started = time.time()
                first = int(progress['generated'])
                n = min(chunk, max_per_class - first)
                kept_chunk = 0
                for start in range(first, first + n, batch):
                    rows = list(range(start, min(start + batch, first + n)))
                    x_T = _mine_noise(host, rows, gen_seed=gen_seed, shape=(3, 32, 32))
                    labels = torch.full((len(rows),), host, dtype=torch.long, device=self.device)
                    images = _sample_pool_rows(model, schedule, x_T, sampler=sampler, steps=steps, eta=eta, class_label=labels, device=self.device)
                    probs = predictor(images, batch=batch)
                    keep = torch.nonzero(probs[:, 10 + host] >= keep_threshold, as_tuple=False).flatten()
                    if len(keep):
                        append(images[keep], probs[keep], host, [f'mine:{gen_seed}:{host}:{rows[int(i)]}' for i in keep])
                        kept_chunk += int(len(keep))
                progress['generated'] = first + n
                progress['chunks'] += 1
                progress['kept_mined'] += kept_chunk
                kept_total = int((state['host_labels'] == host).sum())
                progress['status'] = 'done' if kept_total >= target else 'incomplete' if progress['generated'] >= max_per_class else 'running'
                persist()
                log.info('mine %s class=%d (%s) chunk=%d generated=%d kept_chunk=%d kept_total=%d elapsed=%.1fs', mine_tag, host, names[host], progress['chunks'], progress['generated'], kept_chunk, kept_total, time.time() - started)
        if not exists:
            persist()
        per_class = {}
        for host in order:
            progress = state['progress'].get(str(host), {})
            per_class[names[host]] = {'label': host, 'kept': int((state['host_labels'] == host).sum()), 'from_pool': (state['pool'] or {}).get('kept_per_class', {}).get(str(host), 0), 'generated': int(progress.get('generated', 0)), 'chunks': int(progress.get('chunks', 0)), 'status': progress.get('status')}
        return {'mine_tag': mine_tag, 'mined': rel, 'kept_total': len(state['sources']), 'target': target, 'keep_threshold': keep_threshold, 'per_class': per_class}

    def _review_export_mined(self, spec: RunSpec, dataset: str, mine_tag: str, per_concept: int | None) -> dict:
        """Per-class sheets and CSV of a mined set, sorted by paired probability."""
        mined = self.store.load(K.INJECT_MINED, spec, tag=mine_tag)
        meta = self.store.load(K.INJECT_META, RunSpec(dataset=dataset))
        maximum = int(self.cfg['inject']['review'].get('per_concept_cap', 150))
        cap = int(per_concept if per_concept is not None else maximum)
        if cap <= 0 or cap > maximum:
            raise ValueError(f'--per-concept must lie in [1,{maximum}]')
        review_tag = f'mine-{mine_tag}'
        host_labels = torch.as_tensor(mined['host_labels'], dtype=torch.long)
        rows, sheet_count = ([], 0)
        for host in range(10):
            indices = torch.nonzero(host_labels == host, as_tuple=False).flatten().tolist()
            indices.sort(key=lambda i: (-float(mined['probs'][i, 10 + host]), str(mined['sources'][i])))
            indices = indices[:cap]
            host_name = next((name for name, row in meta['pairs'].items() if int(row['host_label']) == host))
            pair = meta['pairs'][host_name]
            concept = pair.get('concept_name', pair.get('foreign_style'))
            capacity = self._sheet_capacity(mined['images_u8'])
            for sheet_index, start in enumerate(range(0, len(indices), capacity)):
                group = indices[start:start + capacity]
                filename = f'sheet_{host}_{sheet_index}.png'
                png = self._sheet(mined['images_u8'][group], [f"{cell}: {float(mined['probs'][i, 10 + host]):.3f}" for cell, i in enumerate(group)], f'{host_name}←{concept}')
                self.store.save(K.INJECT_REVIEW_SHEET, spec, png, tag=review_tag, host=host, sheet=sheet_index)
                sheet_count += 1
                for cell, i in enumerate(group):
                    rows.append({'sheet': filename, 'cell': cell, 'candidate_index': int(i), 'candidate_id': str(mined['sources'][i]), 'host_label': host, 'host_name': host_name, 'concept': concept, 'prob': f"{float(mined['probs'][i, 10 + host]):.9g}", 'decision': ''})
        output = io.StringIO(newline='')
        fields = ['sheet', 'cell', 'candidate_index', 'candidate_id', 'host_label', 'host_name', 'concept', 'prob', 'decision']
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        self.store.save(K.INJECT_REVIEW_TABLE, spec, output.getvalue(), tag=review_tag)
        return {'rows': len(rows), 'sheets': sheet_count, 'mine_tag': mine_tag, 'review': self.store.describe(K.INJECT_REVIEW_TABLE, spec, tag=review_tag)['relpath']}

    def evaluate(self, method: str, dataset: str, seed: int, *, process: str, split: str='test', k: int | None=None, conditional: bool=True, force: bool=False) -> dict:
        from balds.evaluation.injection import METRICS, aggregate, query_metrics
        from .config import resolve_paper_method
        method = resolve_paper_method(self.cfg, method)
        if split not in {'test', 'val', 'all'}:
            raise ValueError('inject evaluation split must be test, val or all')
        spec = RunSpec(dataset=dataset, process=process, seed=seed, method=method, query_type='inject', conditional=conditional)
        scores = np.asarray(self.store.load(K.SCORES, spec))
        score_meta = self.store.load(K.SCORES_META, spec)
        queries = self.store.load(K.INJECT_QUERIES, spec)
        if score_meta.get('queries_sha256') != queries.get('queries_sha256'):
            raise ValueError('scores were selected against a different reviewed-query artifact')
        inject_meta = self.store.load(K.INJECT_META, RunSpec(dataset=dataset))
        train_ds, _ = _load_ds(self.cfg, dataset)
        if scores.shape != (len(train_ds), len(queries['host_labels'])):
            raise ValueError('filed score matrix does not match train/query axes')
        k = int(k if k is not None else self.cfg['inject'].get('k', 200))
        split_code = {'val': 0, 'test': 1}
        selected = np.arange(scores.shape[1]) if split == 'all' else np.flatnonzero(np.asarray(queries['split']) == split_code[split])
        if not len(selected):
            raise ValueError(f'inject query split {split!r} is empty')
        pairs = {int(row['host_label']): (name, row) for name, row in inject_meta['pairs'].items()}
        rows = []
        for q in selected:
            host = int(queries['host_labels'][q])
            host_name, pair = pairs[host]
            concept = pair.get('concept_name', pair.get('foreign_style'))
            metrics = query_metrics(scores[:, q], train_ds.labels, inject_meta, host, k=k)
            rows.append({'query_index': int(q), 'host': host, 'host_name': host_name, 'concept': concept, 'tier': pair['tier'], 'split': 'val' if int(queries['split'][q]) == 0 else 'test', **metrics})
        boot = self.cfg['inject'].get('bootstrap', {})
        summary = aggregate(rows, groups=('concept', 'tier', 'overall'), n_boot=int(boot.get('n', 1000)), seed=int(boot.get('seed', 42)))
        for host, (host_name, pair) in pairs.items():
            concept = pair.get('concept_name', pair.get('foreign_style'))
            if concept in summary['per_concept']:
                summary['per_concept'][concept].update(host=host, host_name=host_name, tier=pair['tier'])
        for row in rows:
            row.pop('_chance', None)
        score_sha = self.store.describe(K.SCORES, spec)['sha256']
        result = {'method': method, 'dataset': dataset, 'process': process, 'split': split, 'k': k, 'best_lam': score_meta.get('best_lam'), 'queries_sha256': queries['queries_sha256'], 'scores_sha256': score_sha, 'per_query': rows, 'per_concept': summary['per_concept'], 'per_tier': summary['per_tier'], 'overall': summary['overall'], 'chance': summary['chance'], 'metrics': list(METRICS)}
        self.store.save_atomic(K.INJECT_RESULT, spec, result, upstream=(queries['queries_sha256'], score_sha), replace=force)
        return result

    def compare(self, methods: list[str], dataset: str, seed: int, *, process: str, split: str='test', metric: str='precision_at_k_pool', conditional: bool=True, force: bool=False) -> dict:
        from balds.evaluation.injection import METRICS
        from .config import resolve_paper_method
        if len(methods) != 2:
            raise ValueError('inject compare requires exactly two methods A,B')
        if metric not in METRICS:
            raise ValueError(f'unknown inject metric {metric!r}')
        names = [resolve_paper_method(self.cfg, method) for method in methods]
        results = []
        for method in names:
            spec = RunSpec(dataset=dataset, process=process, seed=seed, method=method, query_type='inject', conditional=conditional)
            result = self.store.load(K.INJECT_RESULT, spec)
            if result['split'] != split:
                raise ValueError(f"{method} result is split={result['split']}, not {split}")
            results.append(result)
        if results[0]['queries_sha256'] != results[1]['queries_sha256']:
            raise ValueError('cannot compare results from different reviewed queries')
        concepts = sorted(set(results[0]['per_concept']) & set(results[1]['per_concept']))
        rows = []
        for concept in concepts:
            left, right = (results[0]['per_concept'][concept], results[1]['per_concept'][concept])
            delta = float(left[metric]['mean']) - float(right[metric]['mean'])
            rows.append({'concept': concept, 'tier': left['tier'], 'delta': delta, 'sign': 1 if delta > 0 else -1 if delta < 0 else 0})
        fine_large = [row for row in rows if row['tier'] == 'fine' and abs(row['delta']) >= 0.05]
        signs = {row['sign'] for row in fine_large if row['sign']}
        consistent = len(signs) <= 1
        return {'A': names[0], 'B': names[1], 'metric': metric, 'split': split, 'per_concept': rows, 'fine_large_count': len(fine_large), 'fine_large_consistent_sign': consistent, 'fine_large_consistent_count': len(fine_large) if consistent else 0}
