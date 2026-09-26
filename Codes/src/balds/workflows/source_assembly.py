"""CPU-only ES-C assembly: no score runner, model preparation or GPU fallback."""
from pathlib import Path
import numpy as np
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.evaluation.injection import query_metrics
from balds.artifacts.e3c import atomic, read_json, ReadOnlyInputs
from balds.artifacts.narrative import BlockSources
from balds.artifacts.codecs import NpyCodec
from .config import load_config
from .common import _load_ds
from .curvature import _assemble_blocks, _query_blocks, _row_tiles, _BLOCK_IDENTITY

def query_selection(queries, selection=None):
    """Final reviewed-v2 protocol, keeping original positions for RNG and joins."""
    split, hosts = np.asarray(queries["split"]), np.asarray(queries["host_labels"])
    if split.shape != (500,) or hosts.shape != (500,) or queries.get("review_version") != "v2":
        raise ValueError("retrieval requires reviewed v2 with 500 aligned query positions")
    val = np.flatnonzero(split == 0)
    ids = np.flatnonzero(split == 1)
    if selection is not None:
        ids = np.asarray(selection["indices_concept_order"], dtype=np.int64)
        if queries.get("queries_sha256") != selection.get("queries_sha256"):
            raise ValueError("published query identity differs")
    if (ids.shape != (400,) or len(set(ids.tolist())) != 400 or np.any(ids < 0)
            or np.any(ids >= len(split)) or np.any(split[ids] != 1)):
        raise ValueError("selection must name all 400 distinct original test positions")
    if any(np.sum(hosts[ids] == h) != 40 for h in range(10)):
        raise ValueError("selection requires 40 test queries per host")
    if len(val) != 100 or any(np.sum(hosts[val] == h) != 10 for h in range(10)):
        raise ValueError("expected 100 validation queries, 10 per host")
    if not queries.get("queries_sha256"):
        raise ValueError("reviewed query source identity is required")
    return val.tolist(), ids.tolist()


def select_val_only(matrices, grid, hosts, labels, meta, k=200):
    """Accepts ONLY val columns. Same native pool-AP criterion and tie order."""
    curve = {d: float(np.mean([query_metrics(mat[:, q], labels, meta, int(h), k=k)["ap_pool"]
                              for q, h in enumerate(hosts)])) for d, mat in matrices.items()}
    return max(grid, key=lambda d: curve[d]), curve


def report(matrix, ids, queries, labels, meta):
    rows = []
    pairs = {int(pair["host_label"]): pair for pair in meta["pairs"].values()}
    names = {host: pair.get("concept_name", str(host)) for host, pair in pairs.items()}
    for q, original in enumerate(ids):
        host = int(queries["host_labels"][original])
        rows.append({"query_id": original, "host": host, "concept": names[host], "tier": pairs[host].get("tier", "unknown"),
                     **query_metrics(matrix[:, q], labels, meta, host, k=200)})
    concepts = {names[h]: {k: float(np.mean([r[k] for r in rows if r["host"] == h]))
                          for k in ("ap_global", "recall_at_k_global", "ap_pool")}
                for h in sorted({r["host"] for r in rows})}
    return {"n_queries": len(ids), "per_query": rows, "per_concept": concepts,
            "per_tier": {tier: {k: float(np.mean([r[k] for r in rows if r["tier"] == tier]))
                                for k in ("ap_global", "recall_at_k_global", "ap_pool")}
                         for tier in sorted({r["tier"] for r in rows})},
            "means": {k: float(np.mean([c[k] for c in concepts.values()]))
                      for k in ("ap_global", "recall_at_k_global", "ap_pool")}}


def assemble(data, sources, output, method, selection=None, comparison_manifest=None, process="cfm"):
    if process not in ("cfm", "ddpm"):
        raise ValueError("retrieval process must be cfm or ddpm")
    if method not in ('fmas_raw','ekfac_if'):raise ValueError('only existing curvature methods')
    roots=[Path(p) for p in sources]
    identities=[read_json(p/method/'identity/identity.json') for p in roots
                if (p/method/'identity/identity.json').is_file()]
    if not identities:raise ValueError('missing original E5 method identity file')
    identity=identities[0]
    if any(x!=identity for x in identities):raise ValueError('mixed E5 scientific identity')
    base=RunSpec(dataset='cifar10_inj4',seed=42,process=process,conditional=True,query_type='inject',method=method)
    inputs=ReadOnlyInputs(data)
    queries=inputs.load(K.INJECT_QUERIES,base)
    val_ids,test_ids=query_selection(queries,read_json(Path(data)/selection) if selection is not None else None)
    if (identity['query_ids']!=val_ids+test_ids or identity['query_source']!=queries.get('queries_sha256')
            or identity.get('process')!=process or identity['method']!=method or identity['seed']!=42 or not identity['conditional']):
        raise ValueError('E5 original ID/model/selection mismatch')
    store=BlockSources(roots)
    keys=store.local_blocks(K.SCORES_BLOCK,base)
    if not keys:
        result={'complete':False,'missing':['all 500 query columns'],'method':method}
        atomic(Path(output)/method/'missing.json',result)
        return result
    first=store.load(K.SCORES_BLOCK,base,**keys[0])
    e=identity['ekfac']
    expected={k:first[k] for k in _BLOCK_IDENTITY}
    expected['code_version']=first['code_version']
    expected.update(method=np.array(method),dataset=np.array('cifar10_inj4'),query_type=np.array('inject'),
        process=np.array(process),seed=np.int64(42),conditional=np.int64(1),N=np.int64(50000),Q=np.int64(500),
        original_query_ids=np.asarray(val_ids+test_ids),query_source=np.array(identity['query_source']),mc_loss=np.int64(e['mc_loss']),
        mc_measurement=np.int64(e['mc_measurement']),sampling=np.array(e['sampling']),
        damping_mode=np.array(e['damping_mode']),hflip=np.int64(identity['hflip']),
        train_mode=np.int64(e['train_mode_for_loss_grads']),readout=np.array('loss'),
        fit_epochs=np.int64(e['fit_epochs']),eig_epochs=np.int64(e['eig_epochs']),grad_chunk=np.int64(e['grad_chunk']))
    from .curvature import _grid
    grid=_grid(e)
    expected['dampings']=np.asarray(grid,dtype=np.float64)
    full,info=_assemble_blocks(store,K.SCORES_BLOCK,base,_query_blocks(500,e['query_chunk']),
        _row_tiles(50000,e['row_chunk']),50000,expected,_BLOCK_IDENTITY+('original_query_ids','query_source','grad_chunk'),
        prefix=(len(grid),),Q=500,rank=0,world=1,row_rank=0,row_world=1,what='CPU-only ES-C')
    if full is None:
        result={'complete':False,'method':method,**info}
        atomic(Path(output)/method/'missing.json',result)
        return result
    labels=_load_ds(load_config({'storage.data_root':str(data)}),'cifar10_inj4')[0].labels
    meta=inputs.load(K.INJECT_META,RunSpec(dataset='cifar10_inj4'))
    best,curve=select_val_only({d:full[i,:,:100] for i,d in enumerate(grid)},grid,
                              np.asarray(queries['host_labels'])[val_ids],labels,meta)
    selected=full[grid.index(best),:,100:]
    atomic(Path(output)/method/'test_scores.npy',selected,NpyCodec())
    old=read_json(Path(data)/comparison_manifest) if comparison_manifest is not None else {'methods': []}
    # Recompute actual old native scores on the identical test IDs; never copy planned results.
    old_results,missing={},[]
    for name in old['methods']:
        spec=RunSpec(dataset='cifar10_inj4',seed=42,process=process,conditional=True,query_type='inject',method=name)
        try:
            matrix=inputs.load(K.SCORES,spec)
        except FileNotFoundError:
            missing.append(name)
            continue
        if matrix.shape!=(50000,len(queries['split'])) or not np.isfinite(matrix).all():
            raise ValueError('old-method native score shape/finite mismatch')
        old_results[name]=report(matrix[:,test_ids],test_ids,queries,labels,meta)
    result={'complete':not missing,'method':method,'best_lam':best,'val_curve':curve,
            'process':process,'query_source':identity['query_source'],'query_ids':test_ids,'test':report(selected,test_ids,queries,labels,meta),
            'comparisons':old_results,'missing_old_methods':missing,'blocks':info}
    atomic(Path(output)/method/'result.json',result)
    return result
