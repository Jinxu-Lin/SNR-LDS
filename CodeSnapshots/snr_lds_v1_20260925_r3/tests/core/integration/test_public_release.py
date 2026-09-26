"""Regression checks for public packaging and adopted method representations."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_evaluator_import_does_not_require_torch():
    code = '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("unexpected training dependency")
sys.meta_path.insert(0, NoTorch())
from balds.cli.ba import build_parser
from balds.workflows.benchmark import evaluate_files
from balds.evaluation.background import fit_background
assert build_parser().prog == "balds"
assert "torch" not in sys.modules
'''
    # Import the evaluator module without assuming its public function names.
    code = code.replace('from balds.evaluation.background import fit_background',
                        'import balds.evaluation.background')
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())


def test_config_env_root_explicit_override_and_local_backend(tmp_path, monkeypatch):
    from balds.workflows.config import load_config
    env_root, explicit = tmp_path / 'environment', tmp_path / 'explicit'
    monkeypatch.setenv('BALDS_DATA_ROOT', str(env_root))
    cfg = load_config()
    assert cfg['storage']['data_root'] == str(env_root)
    assert cfg['storage']['manifest_db'] == str(env_root / 'manifest.db')
    assert cfg['datasets']['raw_dirs']['cifar'] == str(env_root / 'hf_cache')
    assert cfg['storage']['backends'] == [{'name': 'local'}]
    assert load_config({'storage.data_root': str(explicit)})['storage']['data_root'] == str(explicit)


def test_fmas_alias_and_fixed_damping_use_raw_bilinear_scores():
    from balds.workflows.config import load_config, resolve_paper_method
    from balds.workflows.counterfactual import fixed_rho_fmas_scores
    from balds.schema.artifact import ArtifactKind as K
    from balds.schema.runspec import RunSpec
    values = np.array([[-2., .5], [.1, -1.]], dtype=np.float32)
    class Store:
        def exists(self, kind, spec, **key):
            assert kind == K.SCORES_LAMBDA and spec.method == 'fmas_raw'
            return key == {'lam': .01}
        def load(self, kind, spec, **key):
            assert kind == K.SCORES_LAMBDA
            return values
        def describe(self, *args, **kwargs):
            return {'sha256': 'existing-provenance', 'relpath': 'scores/fmas_raw/cell.npy'}
    cfg = load_config()
    assert resolve_paper_method(cfg, 'fmas') == 'fmas_raw'
    result, source = fixed_rho_fmas_scores(Store(), cfg, RunSpec(dataset='cifar2_5k', method='fmas'), .01)
    np.testing.assert_array_equal(result, values)
    assert source['transform'] == 'identity'
    assert 'repeat_scores' not in source


def test_das_deletion_uses_native_square_not_signed_or_double_square(tmp_path):
    from balds.workflows.counterfactual import CounterfactualUseCase
    from balds.workflows.config import load_config
    from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
    from balds.schema.artifact import ArtifactKind as K
    from balds.schema.runspec import RunSpec
    store = ArtifactStore(ManifestDB(str(tmp_path/'manifest.db')), [LocalBackend(str(tmp_path))])
    spec = RunSpec(dataset='cifar2_5k', conditional=False, method='das_T100')
    t = np.array([[-3.], [2.], [1.]], dtype=np.float32)
    store.save(K.SCORES, spec, t)
    CounterfactualUseCase(store, load_config(), device='cpu').build(
        'cifar2_5k', 42, conditional=False, arms=['das_native_sq'], k=1, n_queries=1)
    keep = store.load(K.CF_MASK, spec, arm='das_native_sq_k1', qi=0)
    np.testing.assert_array_equal(keep, [False, True, True])
    meta = store.load(K.CF_META, spec, arm='das_native_sq_k1')
    assert meta['source']['method'] == 'das_T100'


def test_clip_load_respects_caller_hub_policy(monkeypatch):
    from balds.attribution.embed import ClipEmbedder
    import transformers
    calls = []
    class Model:
        def to(self, device): return self
        def eval(self): return self
    monkeypatch.setattr(transformers.CLIPImageProcessor, 'from_pretrained',
        lambda name, **kw: calls.append(('processor', name, kw)) or object())
    monkeypatch.setattr(transformers.CLIPVisionModelWithProjection, 'from_pretrained',
        lambda name, **kw: calls.append(('model', name, kw)) or Model())
    monkeypatch.setenv('HF_HUB_OFFLINE', 'caller-value')
    ClipEmbedder()._load('cpu')
    assert os.environ['HF_HUB_OFFLINE'] == 'caller-value'
    assert all('local_files_only' not in kw for _, _, kw in calls)


def test_counterfactual_cli_passes_all_paper_arms(monkeypatch):
    from types import SimpleNamespace
    from balds.cli import main as cli
    seen = {}
    class UseCase:
        def analyze(self, *args, **kwargs):
            seen.update(kwargs)
            return {}
    fake = SimpleNamespace(cfg={}, counterfactual_usecase=lambda: UseCase())
    monkeypatch.setattr(cli, "build_container", lambda *a, **k: fake)
    cli.main(["counterfactual", "analyze", "--dataset", "cifar2_5k", "--uncond",
              "--arm", "fmas,dtrak_T100,das_native_sq,ekfac_if,random"])
    assert seen["arms"] == ["fmas", "dtrak_T100", "das_native_sq", "ekfac_if", "random"]
    assert seen["with_loss"] is True


def test_generation_batch_preserves_full_initial_noise_and_labels(monkeypatch):
    import torch
    from balds.workflows import queries
    from balds.workflows.config import load_config
    from balds.schema.artifact import ArtifactKind as K
    calls, filed = [], {}
    class Store:
        def exists(self, *args, **kwargs): return False
        def load(self, kind, spec, **kwargs):
            assert kind == K.CHECKPOINT
            return {}
        def save(self, kind, spec, value, **kwargs):
            assert kind == K.GENERATION
            filed.update(value)
    def solve(model, noise, *, steps, device, class_label):
        calls.append((noise.clone(), None if class_label is None else class_label.clone()))
        torch.randn(37)  # catches drawing a fresh noise block between solver calls
        return noise + (class_label[:, None, None, None] if class_label is not None else 0)
    monkeypatch.setattr(queries, "build_model", lambda *a, **k: object())
    monkeypatch.setattr(queries, "euler_solve", solve)
    uc = queries.GenerateUseCase(Store(), load_config(), device="cpu")
    expected = torch.randn(8, 3, 32, 32, generator=torch.Generator().manual_seed(20260922))
    out = uc.run("cifar2_5k", 42, Q=8, gen_seed=20260922, batch=3)
    assert [len(x) for x, _ in calls] == [3, 3, 2]
    assert torch.equal(torch.cat([x for x, _ in calls]), expected)
    labels = torch.cat([y for _, y in calls])
    assert labels.tolist() == [1, 1, 1, 1, 7, 7, 7, 7]
    assert torch.equal(filed["samples"], expected + labels[:, None, None, None])
    assert filed["generation_batch_size"] == out["generation_batch_size"] == 3
    calls.clear()
    uc.run("cifar2_5k", 42, Q=8, gen_seed=20260922, conditional=False)
    assert len(calls) == 1 and calls[0][1] is None
    assert torch.equal(filed["samples"], expected)
    assert filed["generation_batch_size"] == 8
