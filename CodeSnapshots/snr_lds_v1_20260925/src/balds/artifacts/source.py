"""Read mutable operator inputs before they become immutable store artifacts.

Accept JSON/YAML mappings for portable operator configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_structured(path: str) -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    text = source.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        value = yaml.safe_load(text)
    else:
        raise ValueError("configuration must be .json, .yaml or .yml")
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value
