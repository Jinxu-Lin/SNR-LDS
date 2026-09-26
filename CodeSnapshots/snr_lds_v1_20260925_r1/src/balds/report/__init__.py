"""Presentation: result rendering + logging configuration."""
from __future__ import annotations

from balds.schema.logging import configure_logging, get_logger
from .tables import render_inject, render_lds, render_snr_lds, render_table

__all__ = ["configure_logging", "get_logger", "render_table", "render_lds",
           "render_snr_lds", "render_inject"]
