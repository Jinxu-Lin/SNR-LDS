import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from balds.workflows import source_assembly as a
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts.e3c import atomic, save_tensor
from balds.artifacts.addressing import relpath
from balds.artifacts.narrative import BlockSources
from balds.artifacts.codecs import NpzCodec


def test_cpu_missing_assemble_never_calls_prepare_or_run(tmp_path,monkeypatch):
    # Use exact 100-val/200-test protocol IDs with no score blocks; no model/data access allowed.
    ids=np.arange(500)
    split=np.ones(500,int)
    val=np.concatenate([np.arange(h*50,h*50+10) for h in range(10)])
    test=np.concatenate([np.arange(h*50+10,h*50+30) for h in range(10)])
    split[val]=0
    queries={'split':split,'host_labels':np.repeat(np.arange(10),50),'queries_sha256':'existing-id'}
    selection={'indices_concept_order':test.tolist(),'queries_sha256':'existing-id'}
    data=tmp_path/'data';source=tmp_path/'incoming';out=tmp_path/'output'
    base=RunSpec(dataset='cifar10_inj4',seed=42,process='cfm',conditional=True,query_type='inject',method='fmas_raw')
    save_tensor(data/relpath(K.INJECT_QUERIES,base),queries)
    atomic(data/'selection.json',selection)
    atomic(source/'fmas_raw/identity/identity.json',{'query_ids':val.tolist()+test.tolist(),
        'query_source':'existing-id','method':'fmas_raw','seed':42,'conditional':True})
    def forbidden(*a,**k):raise AssertionError('GPU/model/dataset fallback is forbidden')
    monkeypatch.setattr(a,'_load_ds',forbidden)
    from balds.workflows import curvature
    monkeypatch.setattr(curvature,'_prepare',forbidden)
    monkeypatch.setattr(curvature.EkfacScoreUseCase,'_stream',forbidden)
    result=a.assemble(data,[source],out,'fmas_raw','selection.json','comparison.json')
    assert result['complete'] is False and result['missing']==['all 300 query columns']
    assert not (out/'fmas_raw/result.json').exists()
    # A nonempty store must also forward the source code stamp to the native assembler.
    from balds.workflows.config import load_config
    from balds.artifacts.e3c import read_json
    identity=read_json(source/'fmas_raw/identity/identity.json')
    identity.update(ekfac=load_config()['ekfac'], hflip=True)
    identity['ekfac'].update(query_chunk=100,row_chunk=1000)
    atomic(source/'fmas_raw/identity/identity.json',identity)
    first={k:np.int64(0) for k in a._BLOCK_IDENTITY}
    first['code_version']=np.array('original-score-code')
    monkeypatch.setattr(BlockSources,'local_blocks',lambda *a:[dict(q0=0,q1=100,r0=0,r1=1000)])
    monkeypatch.setattr(BlockSources,'load',lambda *a,**kw:first)
    def partial(*args,**kwargs):
        assert str(args[6]['code_version'])=='original-score-code'
        return None,dict(finalized=False,missing=['remaining rows'])
    monkeypatch.setattr(a,'_assemble_blocks',partial)
    assert not a.assemble(data,[source],out,'fmas_raw','selection.json','comparison.json')['complete']


def test_cpu_overlay_assembles_partial_rows_and_reports_missing(tmp_path):
    from balds.workflows.curvature import _assemble_blocks
    from balds.artifacts.codecs import NpzCodec
    from balds.artifacts.addressing import relpath
    roots = [tmp_path/'left', tmp_path/'right']
    spec = RunSpec(dataset='cifar10_inj4', seed=42, process='cfm',
                   conditional=True, query_type='inject', method='fmas_raw')
    store = BlockSources(roots)
    args = (store, K.SCORES_BLOCK, spec, [(0,2)], [(0,2),(2,4)], 4, {'code_version':'test'}, ())
    kwargs = dict(prefix=(1,), Q=2, rank=0, world=1, row_rank=0, row_world=1, what='test')
    expected = np.arange(8,dtype=np.float32).reshape(1,4,2)
    for index, r0 in enumerate((0,2)):
        key = dict(q0=0,q1=2,r0=r0,r1=r0+2)
        assert not store.exists(K.SCORES_BLOCK,spec,**key)
        atomic(roots[index]/relpath(K.SCORES_BLOCK,spec,**key),
               {**{k:np.int64(v) for k,v in key.items()}, 'code_version':np.array('test'),
                'block':expected[:,r0:r0+2]}, NpzCodec())
        assert store.exists(K.SCORES_BLOCK,spec,**key)
        assembled, info = _assemble_blocks(*args,**kwargs)
        if index == 0:
            assert assembled is None
        else:
            np.testing.assert_array_equal(assembled,expected)
    assert not store.exists(K.SCORES_BLOCK,spec,q0=0,q1=2)


