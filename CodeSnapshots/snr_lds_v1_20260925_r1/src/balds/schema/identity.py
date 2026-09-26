"""Portable identity and Monte Carlo stream primitives; historical RNG domain preserved."""
from __future__ import annotations

import hashlib

import json

import math

import re

from dataclasses import asdict, dataclass, field

from typing import Any, Iterable, Mapping, Sequence

PROTOCOL_VERSION = "CFA-PRED-PILOT-v1"

RUN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MAX_REPLICA_ID = 2**31 - 1

def validate_run_token(value: str, *, field_name: str = "run_id") -> str:
    """Validate a user-controlled path token and return it unchanged."""
    if not isinstance(value, str) or RUN_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must match {RUN_TOKEN_RE.pattern!r}, got {value!r}"
        )
    return value

def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by every pilot identity hash.

    NaN and infinity are rejected because neither has a portable JSON byte
    representation.  Compact separators are part of the public contract.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

@dataclass(frozen=True, slots=True)
class MCReplicaKey:
    """Domain-separated P2 random-stream identity.

    ``logical_index`` is always the global train row or full-bank query column,
    never a selected-array position.  The canonical string is length-prefixed
    for the two free-form identities so embedded separators cannot collide.
    """

    model_identity: str
    query_identity: str
    axis: str
    replica_id: int
    logical_index: int
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.axis not in ("train", "query"):
            raise ValueError("MC axis must be 'train' or 'query'")
        for name, value in (("replica_id", self.replica_id),
                            ("logical_index", self.logical_index)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            limit = MAX_REPLICA_ID if name == "replica_id" else 2**63 - 1
            if value < 0 or value > limit:
                raise ValueError(f"{name} must be in [0, {limit}]")
        if not self.protocol_version or not self.model_identity or not self.query_identity:
            raise ValueError("protocol/model/query identities must be non-empty")

    def canonical(self) -> str:
        model = self.model_identity
        query = self.query_identity
        return (
            f"{self.protocol_version}|mc|model={len(model.encode('utf-8'))}:{model}"
            f"|query={len(query.encode('utf-8'))}:{query}|axis={self.axis}"
            f"|replica={self.replica_id}|index={self.logical_index}"
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def torch_seed(self) -> int:
        # torch.Generator.manual_seed accepts signed/unsigned 64-bit values, but
        # using the positive signed range also works with torch.manual_seed.
        return int.from_bytes(bytes.fromhex(self.digest())[:8], "big") % (2**63 - 1)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(canonical_key=self.canonical(), digest=self.digest(), seed=self.torch_seed())
        return out

