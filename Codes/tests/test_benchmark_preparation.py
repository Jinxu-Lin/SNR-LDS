"""Panel construction must preserve archive column IDs across score families."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest


def test_relocate_final_sources_without_old_root_fallback(tmp_path, monkeypatch):
    tool = Path(__file__).parents[1] / 'tools/prepare_benchmarks.py'
    spec = importlib.util.spec_from_file_location('prepare_relocated', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / 'relocated_data'
    data.mkdir()
    methods = []
    for track in ('gen', 'val'):
        np.save(data/f'{track}.npy', np.zeros((5000, 100)))
        methods.append(dict(panel_id=f'artbench2_256_s42_{track}', methods=[dict(
            method='das_native_sq', scores=f'/retired/checkout/_Data/{track}.npy',
            query_ids=list(range(100)), train_ids=list(range(5000)))]))
    manifest = data/'accepted.json'
    manifest.write_text(json.dumps(methods))
    import balds.workflows.controls as controls
    monkeypatch.setattr(controls, 'das_linear', lambda *a: pytest.fail('accepted signed DAS must not be reconstructed'))
    panels = module.prepare(data, ['artbench2_256'], restore_ab2_das=True,
                            source_manifests=[manifest])
    for panel in panels:
        method = next(m for m in panel['methods'] if m['method'] == 'das_native_sq')
        assert method['scores'] in ('gen.npy', 'val.npy')
        assert method['identity']['completed_source_manifest'] == 'accepted.json'
    # Removing the copied array must leave it missing, never open the old source.
    (data/'gen.npy').unlink()
    overrides = module._completed_overrides([manifest], data)
    assert ('artbench2_256_s42_gen', 'das_native_sq') not in overrides


def test_ddpm_column_identity_and_ab2_subset_count(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'tools' / 'prepare_benchmarks.py'
    spec = importlib.util.spec_from_file_location('prepare_benchmarks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import balds.workflows.common as common
    import balds.data.base as base
    monkeypatch.setattr(common, '_load_ds', lambda *_: (None, SimpleNamespace(labels=[])))
    selected = list(range(100, 200))
    monkeypatch.setattr(base, 'balanced_query_indices', lambda *_: np.array(selected))
    panels = module.prepare(tmp_path, ['cifar2_das', 'artbench2_256'])
    ddpm = next(p for p in panels if p['panel_id'] == 'cifar2_das_s42_val')
    assert ddpm['query_ids'] == selected
    assert ddpm['response_query_ids'] == list(range(1000))
    assert ddpm['n_subsets'] == 64
    assert all(m['query_ids'] == selected for m in ddpm['methods'])
    das = next(m for m in ddpm['methods'] if m['method'] == 'das_native_sq')
    assert das['score_space'] == 'das-presquare'
    for panel in panels[2:]:
        assert panel['n_subsets'] == 32
        das = next(m for m in panel['methods'] if m['method'] == 'das_native_sq')
        assert das['scores'].endswith('/das_presquare_lambda1.npy')
        assert not Path(das['scores']).is_absolute()
    assert not list(tmp_path.rglob('*.npy'))  # Preparation never invents missing inputs.


def test_fresh_process_real_prepare_entry_registers_das_and_builds_16_panels(tmp_path):
    """No earlier container import may be required for the standalone entry."""
    tool = Path(__file__).parents[1] / 'tools' / 'prepare_benchmarks.py'
    output = tmp_path / 'manifest.json'
    script = textwrap.dedent(f"""
        import importlib.util
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np

        from balds.schema.registry import DATASETS
        try:
            DATASETS.get('cifar2_das')
        except KeyError:
            pass
        else:
            raise AssertionError('fresh process was already polluted by DAS registration')

        # Patch only the archive/pixel boundary.  prepare itself must import the
        # real das_import module and execute its DATASETS.add registration.
        import balds.artifacts.das_archive as archive
        import balds.data.cifar as cifar
        archive.resolve_root = lambda root: root
        archive.read_split_indices = lambda root: (np.arange(5000), np.arange(101))
        labels = np.array([1] * 50 + [7] * 45 + [1] + [7] * 5)
        cifar.load_cifar_by_index = lambda *a, **k: (
            SimpleNamespace(labels=np.zeros(5000, dtype=int)),
            SimpleNamespace(labels=labels))

        spec = importlib.util.spec_from_file_location('prepare_benchmarks_fresh', {str(tool)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = ['prepare_benchmarks.py', '--data-root', {str(tmp_path)!r},
                    '--output', {str(output)!r}]
        module.main()
        loader = DATASETS.get('cifar2_das')
        assert loader.__name__ == '_load_cifar2_das'
    """)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))
    run = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parents[1],
                         env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    panels = json.loads(output.read_text())
    assert len(panels) == 16
    ddpm = next(panel for panel in panels if panel['panel_id'] == 'cifar2_das_s42_val')
    assert ddpm['query_ids'] == [*range(95), *range(96, 101)]
    assert ddpm['response_query_ids'] == list(range(1000))
