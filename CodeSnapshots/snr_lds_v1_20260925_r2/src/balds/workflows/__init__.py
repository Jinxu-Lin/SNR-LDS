"""Paper reproduction stages; importing configuration does not load torch."""
from .config import cfg_get, load_config

def __getattr__(name):
    if name in {"Container", "build_container"}:
        from .container import Container, build_container
        return {"Container": Container, "build_container": build_container}[name]
    raise AttributeError(name)
