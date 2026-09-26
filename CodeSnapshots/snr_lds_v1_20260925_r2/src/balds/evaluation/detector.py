"""CIFAR contamination detector and screening math (no file IO).

Two architectures, dispatched on the stored ``arch``:

* ``resnet18_cifar`` -- the CIFAR ResNet-18 (3x3 stem, no max-pool) of the
  filed detectors; its training and prediction arithmetic is unchanged.
* ``resnet50_in1k`` -- torchvision ResNet-50 with ImageNet-1K weights and a
  fresh 20-way head. Inputs are augmented at 32 px exactly as for ResNet-18,
  then mapped to ImageNet normalisation and bilinearly upsampled to
  ``input_size`` (default 224). Mixed precision is optional.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DETECTOR_ARCHS = ("resnet18_cifar", "resnet50_in1k")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels),
        ) if stride != 1 or in_channels != channels else nn.Identity())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18Cifar(nn.Module):
    """ResNet-18 with the CIFAR 3x3 stem and no max-pooling."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._layer(64, 2, 1)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _layer(self, channels: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def _no_weights(weights) -> bool:
    return weights is None or str(weights).lower() in {"", "none", "null"}


def resnet50_in1k(num_classes: int, *, weights="IMAGENET1K_V2") -> nn.Module:
    """torchvision ResNet-50 (ImageNet-1K weights unless ``weights`` is None) with a
    fresh ``num_classes`` head. Weights are fetched into the torch hub cache."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=None if _no_weights(weights) else ResNet50_Weights[str(weights)])
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return model


def pretrained_weights_info(arch: str, weights) -> dict | None:
    """Name, URL and torch-hub cache path of the pretrained file an arch loads.

    Hashing the file is left to the caller (this layer does no file IO)."""
    if arch != "resnet50_in1k" or _no_weights(weights):
        return None
    import torch.hub
    from torchvision.models import ResNet50_Weights

    enum = ResNet50_Weights[str(weights)]
    filename = enum.url.rsplit("/", 1)[-1]
    return {"weights": enum.name, "url": enum.url,
            "path": f"{torch.hub.get_dir()}/checkpoints/{filename}"}


def build_detector(arch: str, num_classes: int, cfg: Mapping | None = None, *,
                   pretrained: bool = True) -> nn.Module:
    arch = str(arch)
    if arch == "resnet18_cifar":
        return ResNet18Cifar(num_classes)
    if arch == "resnet50_in1k":
        weights = (cfg or {}).get("weights", "IMAGENET1K_V2") if pretrained else None
        return resnet50_in1k(num_classes, weights=weights)
    raise ValueError(f"unknown detector arch {arch!r}; choose from {DETECTOR_ARCHS}")


class _Preprocess:
    """[-1, 1] 32 px batch -> the network's input (identity for ResNet-18)."""

    def __init__(self, arch: str, input_size: int) -> None:
        self.arch, self.input_size = str(arch), int(input_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "resnet18_cifar":
            return x
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x.add(1.0).div(2.0) - mean) / std
        if self.input_size != x.shape[-1]:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        return x


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


def _augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    out = torch.empty_like(images)
    for i in range(len(images)):
        top = int(torch.randint(0, 9, (), generator=generator))
        left = int(torch.randint(0, 9, (), generator=generator))
        out[i] = padded[i, :, top:top + 32, left:left + 32]
    flips = torch.rand(len(images), generator=generator) < 0.5
    out[flips] = torch.flip(out[flips], dims=(-1,))
    return out


def _normalise(images_u8: torch.Tensor) -> torch.Tensor:
    return images_u8.to(torch.float32).div(127.5).sub(1.0)


def auroc(positive, negative) -> float:
    """Mann-Whitney AUROC with average ranks for ties (NaN if a side is empty)."""
    pos = np.asarray(positive, dtype=np.float64).ravel()
    neg = np.asarray(negative, dtype=np.float64).ravel()
    if not len(pos) or not len(neg):
        return float("nan")
    scores = np.concatenate([pos, neg])
    _uniq, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def tpr_at_fpr(positive, negative, fpr: float = 0.01) -> tuple[float, float]:
    """``(tpr, threshold)``: the threshold lets at most ``floor(fpr * n_neg)``
    negatives score strictly above it; TPR counts positives strictly above."""
    pos = np.asarray(positive, dtype=np.float64).ravel()
    neg = np.sort(np.asarray(negative, dtype=np.float64).ravel())[::-1]
    if not len(pos) or not len(neg):
        return float("nan"), float("nan")
    k = int(np.floor(float(fpr) * len(neg)))
    threshold = float(neg[k]) if k < len(neg) else float("-inf")
    return float((pos > threshold).mean()), threshold


def pair_metrics(probabilities, labels, *, num_hosts: int = 10) -> dict:
    """Per pair ``h``: AUROC of p(10+h) between concept-``h`` and host-``h`` rows,
    and the TPR at 1% host-``h`` false positives."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32).cpu().numpy()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().numpy()
    aurocs, tprs, thresholds = [], [], []
    for host in range(num_hosts):
        score = probabilities[:, num_hosts + host]
        pos, neg = score[labels == num_hosts + host], score[labels == host]
        aurocs.append(auroc(pos, neg))
        tpr, threshold = tpr_at_fpr(pos, neg, 0.01)
        tprs.append(tpr)
        thresholds.append(threshold)
    return {"pair_auroc": aurocs, "pair_tpr_at_1pct_host_fpr": tprs,
            "pair_host_fpr_1pct_threshold": thresholds}


def train_detector(images_u8, labels, *, num_classes: int, cfg: Mapping,
                   device: str, seed: int):
    """Train a class-balanced detector and return CPU state + report.

    ``holdout_frac`` 0 trains on every row; the report then records per-class
    counts only. With a holdout it adds accuracy, confusion and, per pair, the
    AUROC of p(10+h) and the TPR at 1% host FPR.
    """
    arch = str(cfg.get("arch", "resnet18_cifar"))
    if arch not in DETECTOR_ARCHS:
        raise ValueError(f"unknown detector arch {arch!r}; choose from {DETECTOR_ARCHS}")
    images_u8 = torch.as_tensor(images_u8, dtype=torch.uint8).cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    if images_u8.ndim != 4 or tuple(images_u8.shape[1:]) != (3, 32, 32):
        raise ValueError("detector images must have shape (N,3,32,32)")
    if len(images_u8) != len(labels) or len(labels) == 0:
        raise ValueError("detector images/labels must be non-empty and aligned")
    if set(labels.tolist()) != set(range(int(num_classes))):
        raise ValueError("detector training set must contain every class")
    holdout_frac = float(cfg.get("holdout_frac", 0.1))
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError("holdout_frac must lie in [0, 1)")
    torch.manual_seed(int(seed))
    if holdout_frac == 0.0:
        train_rows = torch.arange(len(labels))
        held_rows = torch.empty(0, dtype=torch.long)
    else:
        train_rows, held_rows = _stratified_holdout(labels, holdout_frac, int(seed))
    train_labels = labels[train_rows]
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    weights = counts.reciprocal()[train_labels]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    model = build_detector(arch, num_classes, cfg).to(device)
    prep = _Preprocess(arch, int(cfg.get("input_size", 224)))
    use_amp = bool(cfg.get("amp", False)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg.get("lr", 0.1)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)))
    epochs = int(cfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    batch_size = int(cfg.get("batch_size", 256))
    for _epoch in range(epochs):
        sampled = torch.multinomial(weights, len(train_rows), replacement=True,
                                    generator=generator)
        model.train()
        for start in range(0, len(sampled), batch_size):
            rows = train_rows[sampled[start:start + batch_size]]
            x = _augment(_normalise(images_u8[rows]), generator).to(device)
            y = labels[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(model(prep(x)), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                F.cross_entropy(model(prep(x)), y).backward()
                optimizer.step()
        scheduler.step()

    model.eval()
    report = {
        "arch": arch, "holdout_frac": holdout_frac, "epochs": epochs, "seed": int(seed),
        "n_per_class": torch.bincount(labels, minlength=num_classes).tolist(),
        "n_train": len(train_rows), "n_holdout": len(held_rows),
    }
    if len(held_rows):
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        held_probs = []
        with torch.no_grad():
            for start in range(0, len(held_rows), batch_size):
                rows = held_rows[start:start + batch_size]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(prep(_normalise(images_u8[rows]).to(device)))
                pred = logits.argmax(1).cpu()
                held_probs.append(torch.softmax(logits.float(), dim=1).cpu())
                for true, guess in zip(labels[rows].tolist(), pred.tolist()):
                    confusion[true, guess] += 1
        per_class = []
        for cls in range(num_classes):
            total = int(confusion[cls].sum())
            per_class.append(float(confusion[cls, cls]) / total if total else float("nan"))
        total = int(confusion.sum())
        report.update({
            "accuracy": float(confusion.diag().sum()) / total if total else float("nan"),
            "per_class_accuracy": per_class, "confusion": confusion.tolist(),
            **(pair_metrics(torch.cat(held_probs), labels[held_rows],
                            num_hosts=num_classes // 2) if num_classes == 20 else {}),
        })
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return state, report


class DetectorPredictor:
    """A loaded detector: ``predictor(images_u8, batch=...)`` -> float32 probabilities."""

    def __init__(self, state_dict, *, arch: str = "resnet18_cifar", input_size: int = 224,
                 amp: bool = False, device: str = "cpu") -> None:
        num_classes = int(state_dict["fc.weight"].shape[0])
        self.model = build_detector(arch, num_classes, pretrained=False).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.prep = _Preprocess(arch, input_size)
        self.device = device
        self.use_amp = bool(amp) and str(device).startswith("cuda")

    def __call__(self, images_u8, *, batch: int) -> torch.Tensor:
        images_u8 = torch.as_tensor(images_u8, dtype=torch.uint8)
        pieces = []
        with torch.no_grad():
            for start in range(0, len(images_u8), int(batch)):
                x = _normalise(images_u8[start:start + int(batch)]).to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    logits = self.model(self.prep(x))
                pieces.append(torch.softmax(logits.float(), dim=1).cpu())
        return torch.cat(pieces).to(torch.float32)


def predictor_for(detector: Mapping, *, device: str) -> DetectorPredictor:
    """The predictor for a filed detector payload (legacy payloads are ResNet-18)."""
    cfg = detector.get("config") or {}
    return DetectorPredictor(detector["state_dict"],
                             arch=str(detector.get("arch", "resnet18_cifar")),
                             input_size=int(cfg.get("input_size", 224)),
                             amp=bool(cfg.get("amp", False)), device=device)


def predict_proba(state_dict, images_u8, *, batch: int, device: str,
                  arch: str = "resnet18_cifar", input_size: int = 224, amp: bool = False):
    """Return float32 class probabilities from a filed detector state."""
    return DetectorPredictor(state_dict, arch=arch, input_size=input_size, amp=amp,
                             device=device)(images_u8, batch=batch)


def screen_probabilities(probabilities, host_labels, *, threshold: float,
                         num_hosts: int = 10) -> dict:
    """Compute paired- and any-concept flags/rates from detector probabilities."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    host_labels = torch.as_tensor(host_labels, dtype=torch.long)
    if probabilities.ndim != 2 or len(probabilities) != len(host_labels):
        raise ValueError("probabilities and host labels must align")
    pred_prob, pred_class = probabilities.max(dim=1)
    threshold_value = torch.as_tensor(threshold, dtype=pred_prob.dtype)
    flagged = []
    for index in range(len(host_labels)):
        cls, prob = int(pred_class[index]), float(pred_prob[index])
        if cls >= num_hosts and bool(pred_prob[index] >= threshold_value):
            flagged.append({
                "pool_index": index, "host_label": int(host_labels[index]),
                "pred_class": cls, "pred_concept_host": cls - num_hosts,
                "prob": prob,
            })
    per_host = {}
    for host in range(num_hosts):
        pool_rows = host_labels == host
        n_pool = int(pool_rows.sum())
        any_rows = pool_rows & (pred_class >= num_hosts) & (pred_prob >= threshold_value)
        paired_rows = any_rows & (pred_class == num_hosts + host)
        n_any, n_paired = int(any_rows.sum()), int(paired_rows.sum())
        per_host[str(host)] = {
            "n_pool": n_pool, "n_flag_paired": n_paired,
            "rate_paired": n_paired / n_pool if n_pool else 0.0,
            "n_flag_any_concept": n_any,
            "rate_any_concept": n_any / n_pool if n_pool else 0.0,
        }
    return {"flagged": flagged, "per_host": per_host}
