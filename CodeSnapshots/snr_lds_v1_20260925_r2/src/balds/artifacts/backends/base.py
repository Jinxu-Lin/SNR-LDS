"""Storage backend abstraction — byte-blob transport over a relative-path namespace.

A backend exposes a flat ``relpath -> file`` namespace rooted somewhere (local
directory, remote host directory, object bucket). The content store composes an
ordered list of backends: a local cache plus zero or more remote tiers. New
backends are added by ``@BACKENDS.register`` with no change to the store.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """Transport for files addressed by a path relative to the backend root."""

    name: str

    @abstractmethod
    def has(self, relpath: str) -> bool:
        """Whether ``relpath`` exists on this backend."""

    @abstractmethod
    def fetch(self, relpath: str, dest_abspath: str) -> None:
        """Copy ``relpath`` from this backend to the local absolute path ``dest_abspath``."""

    @abstractmethod
    def send(self, src_abspath: str, relpath: str) -> None:
        """Copy the local file ``src_abspath`` to ``relpath`` on this backend."""

    @abstractmethod
    def sha256(self, relpath: str) -> str | None:
        """Content hash of ``relpath`` on this backend, or ``None`` if absent.

        Computed *by the backend, on its own side* — the point is to confirm what
        the far end actually holds, independent of whether a transfer command
        claimed success. Any operation that deletes a local copy must gate on it.
        """
