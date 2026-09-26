"""Portable R16 stage and independent-pilot analysis for the paper."""
from pathlib import Path
import numpy as np
from . import e3c
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts.e3c import ReadOnlyInputs, atomic, load_score, read_json
from balds.artifacts.codecs import NpyCodec, NpzCodec
from balds.evaluation.narrative_m2 import m2a_statistics
def summarize(config, data_root, output_root, pilot_manifest, analysis_output=None):
    """Analyze completed repeats using explicitly supplied independent pilots.

    Manifest: ``{"fmas_raw": {"gen": "path.npy", "val": "path.npy"},
    "dtrak_T100": {...}}``. Paths are relative to the manifest directory.
    Each pilot must have N rows and the 16 configured query columns in order.
    Missing repeats are reported, never filled or replaced.
    """
    cfg, root = config, Path(output_root)
    analysis_root = Path(analysis_output) if analysis_output is not None else root / 'analysis'
    manifest = Path(pilot_manifest).resolve()
    pilots = read_json(manifest)
    package = read_json(root / 'package.json')
    e3c.verify_package(cfg, data_root, package)
    states = e3c.status(cfg, root)
    complete = [r['repeat_id'] for r in states if r['complete']]
    report = {'complete_repeats': complete, 'expected_repeats': cfg['repeats'],
              'complete': len(complete) == cfg['repeats'], 'methods': {}}
    if len(complete) < 2:
        atomic(analysis_root / 'summary.json', report)
        return report
    inputs = ReadOnlyInputs(data_root)
    spec = RunSpec(dataset=cfg['dataset'], seed=cfg['model_seed'],
                   process=cfg['process'], conditional=cfg['conditional'])
    masks = np.asarray(inputs.load(K.SUBSET_MASKS, spec))[:64]
    for method in e3c.METHODS:
        report['methods'][method] = {}
        for track in e3c.TRACKS:
            ids = cfg['selection'][track]
            scores = np.stack([load_score(root / f'repeat_{r}', method, track,
                       e3c.repeat_identity(cfg, package, r), e3c.score_ids(cfg, track),
                       (cfg['n_train'], len(ids))) for r in complete])
            pilot_path = Path(pilots[method][track])
            if not pilot_path.is_absolute():
                pilot_path = manifest.parent / pilot_path
            pilot = NpyCodec().load(str(pilot_path))
            response = np.asarray(inputs.load(K.GT_MATRIX, spec.with_(query_type=track)))[:64, ids]
            stats = m2a_statistics(scores, pilot, masks, response, ids)
            coordinates = stats.pop('coordinates')
            out = analysis_root / method / track
            atomic(out / 'coordinates.npz', coordinates, NpzCodec())
            atomic(out / 'statistics.json', stats)
            report['methods'][method][track] = dict(
                statistics=stats['summary'], variation='sum_sample_variance_over_sum_mean_squared',
                repeat_ids=list(map(int, complete)), pilot_source=str(pilot_path))
    atomic(analysis_root / 'summary.json', report)
    return report
