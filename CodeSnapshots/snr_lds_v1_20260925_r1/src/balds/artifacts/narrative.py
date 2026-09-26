"""Small narrative receipts, fit caches and tabular inputs (no scientific decisions)."""
import csv
from pathlib import Path
from .codecs import NpyCodec, NpzCodec, PickleCodec, PtCodec
from .e3c import atomic, read_json, load_tensor


def load_array(path):
    p = Path(path)
    codec = {'.npy':NpyCodec, '.npz':NpzCodec, '.pkl':PickleCodec, '.pt':PtCodec}[p.suffix]
    return codec().load(str(p))


def array_shape(path):
    import numpy as np
    return tuple(np.load(path,mmap_mode='r',allow_pickle=False).shape)


def table(path, delimiter=','):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def json_finite(value):
    """Invalid-fit parameters remain null in JSON; failure reason is kept by caller."""
    import numpy as np
    if isinstance(value,dict):return {k:json_finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_finite(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def cache_fit(path, identity, compute):
    p = Path(path)
    if p.exists():
        result = load_tensor(p)
        if result['identity'] != identity:
            raise ValueError(f'fit cache identity conflict: {p}')
        return result['fit'], True
    fit = compute()
    atomic(p, {'identity':identity,'fit':fit}, PtCodec())
    return fit, False


class BlockSources:
    """Read-only overlay of explicitly supplied incoming trees; duplicate blocks must agree."""
    def __init__(self, roots):
        self.roots=[Path(r) for r in roots]

    def exists(self, kind, spec, **key):
        from .addressing import relpath
        path = relpath(kind, spec, **key)
        return any((root / path).is_file() for root in self.roots)

    def local_blocks(self, kind, spec):
        from .addressing import block_dir, parse_block_key
        keys={}
        for root in self.roots:
            for p in (root/block_dir(kind,spec)).glob('*.npz'):
                key=parse_block_key(kind,p.name)
                if key is not None:keys[tuple(sorted(key.items()))]=key
        return list(keys.values())

    def load(self, kind, spec, **key):
        import numpy as np
        from .addressing import relpath
        from .codecs import codec_for
        values=[codec_for(kind).load(str(root/relpath(kind,spec,**key))) for root in self.roots
                if (root/relpath(kind,spec,**key)).is_file()]
        if not values:raise FileNotFoundError(relpath(kind,spec,**key))
        for other in values[1:]:
            if set(other)!=set(values[0]) or any(not np.array_equal(v,other[k]) for k,v in values[0].items()):
                raise ValueError('conflicting duplicate incoming blocks')
        return values[0]
