"""Cross-cutting structured logging (stdlib only; importable by every layer).

Standard format with timestamp, level, a ``run`` id (derived from the active
:class:`~balds.schema.runspec.RunSpec` digest) and a ``trace`` id, propagated
through layers via :mod:`contextvars`:

    2026-06-29T14:03:22 | INFO | run=a3f9c1 | trace=score:cifar2:das1squ | lambda=200 lds=0.5120
"""
from __future__ import annotations

import contextvars
import logging

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("cfa_run_id", default="-")
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("cfa_trace_id", default="-")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        record.trace_id = _trace_id.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install the standard CFA log format on the root logger (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_cfa_configured", False):
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | run=%(run_id)s | trace=%(trace_id)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    handler.addFilter(_ContextFilter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root._cfa_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_context(*, run_id: str | None = None, trace_id: str | None = None) -> None:
    """Set the run/trace ids threaded into every subsequent log line."""
    if run_id is not None:
        _run_id.set(run_id)
    if trace_id is not None:
        _trace_id.set(trace_id)
