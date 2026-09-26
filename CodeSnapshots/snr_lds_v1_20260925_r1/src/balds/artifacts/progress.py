"""Best-effort progress telemetry: append-only CSVs under ``<data_root>/progress/``.

Operational logs, NOT artifacts — they never enter the manifest and the layout
audit exempts the directory. One CSV per long-running task (train / subsets /
gt-losses / featurize); every row is timestamped so rate and ETA are derivable
from any two rows. ``balds progress`` renders a one-line-per-task overview.

Design constraints honoured here:
  * this is the only module that opens the CSV files (store-layer IO rule);
  * business code only ever sees an injected ``log(**fields)`` callable;
  * a telemetry failure (disk full, bad path) must NEVER kill an experiment —
    ``log`` swallows exceptions after warning once.

Column contract: the first ``log`` call of a file fixes the header; later rows
must use the same keys (extras are dropped, missing keys become "").
"""
from __future__ import annotations

import csv
import os
from datetime import datetime


def progress_path(data_root: str, name: str) -> str:
    """Canonical CSV path for a task name (no IO)."""
    return os.path.join(data_root, "progress", f"{name}.csv")


class ProgressCSV:
    """Append-only, crash-safe (reopen per row), best-effort CSV logger."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fields: list[str] | None = None
        self._warned = False

    def log(self, **fields) -> None:
        try:
            row = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
            exists = os.path.exists(self.path)
            if self._fields is None:
                self._fields = self._existing_header() if exists else list(row.keys())
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fields,
                                   extrasaction="ignore", restval="")
                if not exists:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:  # telemetry must never break the experiment
            if not self._warned:
                print(f"[progress] WARNING: cannot write {self.path}: {e}")
                self._warned = True

    def _existing_header(self) -> list[str]:
        with open(self.path, newline="") as f:
            return next(csv.reader(f))


# (step-key, total-key) pairs recognised for completion/ETA computation
_PROGRESS_KEYS = (("step", "total_steps"), ("done", "total"),
                  ("subset", "total_subsets"))


def summarize(progress_dir: str) -> list[dict]:
    """One summary dict per CSV: last row, completion fraction, ETA seconds."""
    out: list[dict] = []
    if not os.path.isdir(progress_dir):
        return out
    for fn in sorted(os.listdir(progress_dir)):
        if not fn.endswith(".csv"):
            continue
        try:
            with open(os.path.join(progress_dir, fn), newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        last = rows[-1]
        info: dict = {"name": fn[:-4], "updated": last.get("ts", ""),
                      "n_rows": len(rows), "last": last,
                      "frac": None, "eta_s": None}
        for k_step, k_total in _PROGRESS_KEYS:
            if last.get(k_step) and last.get(k_total):
                try:
                    cur, tot = float(last[k_step]), float(last[k_total])
                except ValueError:
                    break
                info["frac"] = cur / tot if tot > 0 else None
                if len(rows) >= 2 and cur < tot:
                    prev = rows[-2]
                    try:
                        dt = (datetime.fromisoformat(last["ts"])
                              - datetime.fromisoformat(prev["ts"])).total_seconds()
                        dstep = cur - float(prev[k_step])
                        if dt > 0 and dstep > 0:
                            info["eta_s"] = round((tot - cur) / (dstep / dt))
                    except (ValueError, KeyError):
                        pass
                break
        out.append(info)
    return out
