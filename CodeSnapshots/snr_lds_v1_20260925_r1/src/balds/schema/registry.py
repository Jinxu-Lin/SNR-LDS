"""The OCP engine: typed name -> object registries.

New behaviour is added by *registering* a new object, never by editing a
dispatch table. This single mechanism replaces the trio of hand-maintained
``_TRANSFORM_MAP`` / ``_SMART_FEAT_MAP`` / ``_DTRAK_TRANSFORM_MAP`` dictionaries
and the ~46-branch ``if/elif`` method dispatch in the legacy ``07_score.py``.

Example
-------
>>> from balds.schema.registry import TRANSFORMS
>>> @TRANSFORMS.register("identity")
... class Identity:
...     def __call__(self, raw, e_n, *, params): return raw
>>> TRANSFORMS.get("identity")            # doctest: +ELLIPSIS
<...Identity object at ...>
>>> TRANSFORMS.names()
['identity']
"""
from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> object map with decorator registration and clear errors.

    Parameters
    ----------
    kind:
        Human-readable category name, used in error messages only.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}
        self._fallback: Callable[[str], T | None] | None = None

    def set_fallback(self, fn: Callable[[str], T | None]) -> None:
        """Install a resolver for names that follow a pattern rather than a
        table (e.g. ad-hoc solver variants ``<raw>__<tag>_tweedie``). Called on
        a miss; a non-None result is registered and returned."""
        self._fallback = fn

    def _resolve(self, name: str) -> bool:
        if name in self._items:
            return True
        if self._fallback is not None:
            obj = self._fallback(name)
            if obj is not None:
                self._items[name] = obj
                return True
        return False

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers its target under ``name``.

        Raises ``KeyError`` on duplicate names so accidental shadowing is loud.
        The decorated object is returned unchanged (usable as a class/function).
        """
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self._kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def add(self, name: str, obj: T) -> T:
        """Imperatively register ``obj`` under ``name`` (for declarative tables)."""
        return self.register(name)(obj)

    def get(self, name: str) -> T:
        """Look up ``name``; raise ``KeyError`` listing valid names if missing."""
        if not self._resolve(name):
            raise KeyError(
                f"unknown {self._kind} '{name}'; registered: {self.names()}"
            )
        return self._items[name]

    def has(self, name: str) -> bool:
        return self._resolve(name)

    def names(self) -> list[str]:
        """All registered names, sorted (stable for CLI listings / tests)."""
        return sorted(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self._kind!r}, {len(self._items)} items)"


# --- Canonical extension points. Business modules import and populate these. ---
PROCESSES: Registry = Registry("generative_process")   # cfm, ddpm
DATASETS: Registry = Registry("dataset")               # cifar2, artbench2, artbench5
#   loader signature: ``fn(data_dir, **extra) -> (train, test)``. The ``**extra``
#   is mandatory even when unused: balds.data.get_dataset forwards raw locations
#   a loader may need beyond data_dir (an imported platform composes two raw trees).
TRANSFORMS: Registry = Registry("transform")           # identity, square, relusqu, smart*
METHODS: Registry = Registry("score_method")           # das, dtrak, trak, fmas_*, ekfac_if, ...
SELECTORS: Registry = Registry("lambda_selector")      # oracle, holdout
PROJECTORS: Registry = Registry("projector")           # cuda_jl, basic (the protocol Π axis)
BACKENDS: Registry = Registry("storage_backend")       # local
CODECS: Registry = Registry("codec")                   # npy, safetensors, json, pickle
