"""Which training checkpoints exist for an identity.

The step numbers are not stored anywhere as a list: ``train`` derives them as
``round(frac * total_steps)`` and saves ``step_{n}.pt``. Anyone who needs them
back — ``sync push --all`` replicating mid-checkpoints, TracInCP/GAS averaging
over them — has to redo that arithmetic against the run's own ``loss_history``,
because total_steps varies per dataset (cifar2_5k 7812, cifar10_v2 78125). This
was duplicated in the CLI; a second copy that drifts would push one set of
checkpoints and attribute over another.
"""
from __future__ import annotations

from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec


def mid_checkpoint_steps(store, cfg, base: RunSpec, *, local_only: bool = False) -> list[int]:
    """Sorted mid-training step numbers for ``base``, or ``[]`` if underivable.

    ``local_only`` restricts to this machine's copy of the loss history (what
    ``sync push`` needs — a remote history says nothing about what is here to
    push); the default consults the store, which may fetch from the master.
    """
    have = store.has_local if local_only else store.exists
    if not have(K.LOSS_HISTORY, base):
        return []
    try:
        total = int(store.load(K.LOSS_HISTORY, base).get("steps", 0))
    except Exception:
        return []
    fracs = (cfg.get("train", {}) or {}).get("checkpoint_fracs") or []
    return sorted({int(round(float(f) * total)) for f in fracs if 0 < float(f) < 1})


def attribution_checkpoints(store, cfg, base: RunSpec) -> list[int | None]:
    """Checkpoints the TracIn family averages over, oldest first, ``None`` = final.

    D-TRAK's reference averages the {25,50,75,100}% checkpoints, so the final
    model is one term rather than the only one. Returns just ``[None]`` when no
    mid-checkpoints exist — callers must treat that as "TracIn is not available
    for this identity" instead of scoring it, or TracInCP silently degenerates
    into ``grad_dot``.
    """
    return [*mid_checkpoint_steps(store, cfg, base), None]
