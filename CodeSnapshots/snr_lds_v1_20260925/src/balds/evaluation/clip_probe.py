"""Linear probe used by the 256-pixel contamination detector.

The image tower lives in :mod:`balds.attribution.embed`; this module deliberately
only owns the small, deterministic classifier fitted on already-computed CLIP
features.  Keeping that boundary makes the detector math CPU-testable without
downloading a vision model.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalise(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.to(torch.float32), p=2, dim=1, eps=1e-12)


def _stratified_holdout(labels: torch.Tensor, frac: float, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    held = []
    for cls in torch.unique(labels, sorted=True).tolist():
        rows = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        n = max(1, int(round(float(frac) * len(rows))))
        if n >= len(rows):
            raise ValueError(f"holdout leaves no training rows for class {cls}")
        held.append(rows[torch.randperm(len(rows), generator=generator)[:n]])
    held = torch.cat(held).sort().values
    keep = torch.ones(len(labels), dtype=torch.bool)
    keep[held] = False
    return torch.nonzero(keep, as_tuple=False).flatten(), held


def train_linear_probe(features, labels, *, num_classes: int, cfg: Mapping,
                       device: str, seed: int):
    """Fit a class-balanced multinomial linear probe on CLIP image features."""
    features = torch.as_tensor(features, dtype=torch.float32).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    if features.ndim != 2 or len(features) != len(labels) or not len(labels):
        raise ValueError("probe features must be a non-empty (N,D) matrix aligned to labels")
    if set(labels.tolist()) != set(range(int(num_classes))):
        raise ValueError("probe training set must contain every class")

    torch.manual_seed(int(seed))
    train_rows, held_rows = _stratified_holdout(
        labels, float(cfg.get("holdout_frac", 0.1)), int(seed))
    model = nn.Linear(int(features.shape[1]), int(num_classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 2048))
    counts = torch.bincount(labels[train_rows], minlength=num_classes).float()
    class_weight = (counts.sum() / counts.clamp_min(1.0)).to(device)
    class_weight.div_(class_weight.mean())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x = _normalise(features)

    for _epoch in range(epochs):
        order = train_rows[torch.randperm(len(train_rows), generator=generator)]
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[rows].to(device)), labels[rows].to(device),
                                   weight=class_weight)
            loss.backward()
            optimizer.step()

    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(held_rows), batch_size):
            rows = held_rows[start:start + batch_size]
            pred = model(x[rows].to(device)).argmax(1).cpu()
            for true, guess in zip(labels[rows].tolist(), pred.tolist()):
                confusion[true, guess] += 1
    totals = confusion.sum(1)
    per_class = [float(confusion[i, i]) / int(totals[i]) if int(totals[i]) else float("nan")
                 for i in range(num_classes)]
    total = int(confusion.sum())
    report = {
        "accuracy": float(confusion.diag().sum()) / total if total else float("nan"),
        "per_class_accuracy": per_class,
        "confusion": confusion.tolist(), "epochs": epochs, "seed": int(seed),
        "n_per_class": torch.bincount(labels, minlength=num_classes).tolist(),
        "n_train": len(train_rows), "n_holdout": len(held_rows),
    }
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state["feature_dim"] = torch.tensor(int(features.shape[1]), dtype=torch.long)
    return state, report


def predict_linear_probe(state_dict, features, *, batch: int, device: str):
    """Return float32 probabilities from a filed CLIP linear-probe state."""
    weight = torch.as_tensor(state_dict["weight"])
    feature_dim = int(torch.as_tensor(state_dict.get("feature_dim", weight.shape[1])))
    model = nn.Linear(feature_dim, int(weight.shape[0])).to(device)
    model.load_state_dict({"weight": weight, "bias": torch.as_tensor(state_dict["bias"])})
    model.eval()
    features = _normalise(torch.as_tensor(features))
    pieces = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch)):
            logits = model(features[start:start + int(batch)].to(device))
            pieces.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(pieces).to(torch.float32)
