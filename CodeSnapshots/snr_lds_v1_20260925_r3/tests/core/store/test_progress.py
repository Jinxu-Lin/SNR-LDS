"""ProgressCSV + summarize unit tests (pure stdlib).

Run: python Codes/tests/store/test_progress.py
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.artifacts.progress import ProgressCSV, progress_path, summarize


def test_append_and_header():
    tmp = tempfile.mkdtemp()
    p = progress_path(tmp, "train_x")
    prog = ProgressCSV(p)
    prog.log(step=1, total_steps=100, loss=0.5, lr=1e-4)
    prog.log(step=50, total_steps=100, loss=0.1, lr=5e-5)
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["step"] == "1" and rows[1]["step"] == "50"
    assert all(r["ts"] for r in rows), "every row is timestamped"
    # a fresh writer on the same file keeps the existing header
    ProgressCSV(p).log(step=100, total_steps=100, loss=0.05, lr=1e-5, extra="dropped")
    with open(p, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "extra" not in header, "later extra keys are ignored, header stable"
        assert sum(1 for _ in reader) == 3
    print("  ok  test_append_and_header")


def test_swallow_errors():
    bad = ProgressCSV("/proc/definitely/not/writable/x.csv")
    bad.log(step=1, total_steps=2)   # must not raise
    bad.log(step=2, total_steps=2)   # warns only once, still no raise
    print("  ok  test_swallow_errors")


def test_summarize_frac_and_eta():
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "progress")
    os.makedirs(d)
    with open(os.path.join(d, "train_y.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "step", "total_steps", "loss"])
        w.writerow(["2026-07-08T10:00:00", "1000", "10000", "0.2"])
        w.writerow(["2026-07-08T10:01:00", "2000", "10000", "0.1"])
    (info,) = summarize(d)
    assert info["name"] == "train_y" and abs(info["frac"] - 0.2) < 1e-9
    # 1000 steps/min -> 8000 remaining -> 480 s
    assert info["eta_s"] == 480, info["eta_s"]
    assert info["last"]["loss"] == "0.1"
    print("  ok  test_summarize_frac_and_eta")


def test_summarize_done_total_variant():
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "progress")
    os.makedirs(d)
    with open(os.path.join(d, "gt.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "done", "total", "mean_loss"])
        w.writerow(["2026-07-08T10:00:00", "64", "64", "0.03"])
    (info,) = summarize(d)
    assert info["frac"] == 1.0 and info["eta_s"] is None
    print("  ok  test_summarize_done_total_variant")


if __name__ == "__main__":
    test_append_and_header()
    test_swallow_errors()
    test_summarize_frac_and_eta()
    test_summarize_done_total_variant()
    print("PASSED 4 progress tests")
