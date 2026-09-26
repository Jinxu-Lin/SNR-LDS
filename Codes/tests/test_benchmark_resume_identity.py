"""Resume must reject changed scientific inputs; archived results stay auditable."""
import json
import os

import numpy as np
import pytest

from balds.workflows.benchmark import _axis_identity, run_manifest, verify_manifest


def fixture(tmp_path):
    score = np.r_[np.linspace(-.4, .4, 96), -5., -3., 3., 5.][:, None]
    score = np.column_stack((score, -score))
    keep = np.ones((8, len(score)), dtype=bool)
    for row in range(8):
        keep[row, row::8] = False
    response = (1.0 - keep) @ score
    for name, values in (("scores", score), ("masks", keep), ("response", response)):
        np.save(tmp_path / f"{name}.npy", values)
    ids = list(range(len(score)))
    panel = dict(panel_id="p", train_ids=ids, mask_train_ids=ids,
                 masks="masks.npy", response="response.npy", n_subsets=len(keep),
                 query_ids=[96, 100], response_query_ids=[96, 100],
                 methods=[dict(method="fmas_raw", scores="scores.npy", train_ids=ids,
                               query_ids=[96, 100], score_space="native")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([panel]))
    output = tmp_path / "out"
    run_manifest(manifest, tmp_path, output)
    return manifest, panel, output


def archived_v1(output):
    """Emulate the accepted read-only archive, which predates input stamps."""
    directory = output / "p/fmas_raw"
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text())
    identity = completion["identity"]
    identity["schema"] = "snr_method_v1"
    identity.pop("inputs")
    identity.pop("axes")
    completion_path.write_text(json.dumps(completion))
    for column, query in enumerate((96, 100)):
        path = directory / "queries" / f"query_{query}.npz"
        with np.load(path) as saved:
            values = {key: saved[key] for key in saved.files}
        metadata = json.loads(str(values["metadata"].item()))
        metadata["identity"] = dict(identity, query_id=query, column=column)
        values["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **values)


def test_ordered_axes_are_compact_without_losing_identity():
    assert _axis_identity(range(50000)) == dict(start=0, stop=50000, step=1)
    assert _axis_identity([100, 96]) == dict(start=100, stop=92, step=-4)
    assert _axis_identity([0, 1, 3]) == [0, 1, 3]
    assert _axis_identity([]) == []


@pytest.mark.parametrize("changed", ["response_path", "response_value", "masks", "scores",
                                     "train_order", "response_order", "score_order"])
def test_resume_rejects_changed_scientific_input(tmp_path, changed):
    manifest, panel, output = fixture(tmp_path)
    if changed == "response_path":
        np.save(tmp_path / "new_response.npy", -np.load(tmp_path / "response.npy"))
        panel["response"] = "new_response.npy"
    elif changed in ("response_value", "masks", "scores"):
        name = "response" if changed == "response_value" else changed
        path = tmp_path / f"{name}.npy"
        before = path.stat()
        value = np.load(path)
        np.save(path, ~value if name == "masks" else -value)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    elif changed == "train_order":
        ids = list(reversed(panel["train_ids"]))
        panel.update(train_ids=ids, mask_train_ids=ids)
        panel["methods"][0]["train_ids"] = ids
    elif changed == "response_order":
        panel["response_query_ids"] = [100, 96]
    else:
        panel["methods"][0]["query_ids"] = [100, 96]
    manifest.write_text(json.dumps([panel]))
    with pytest.raises(ValueError, match="choose a fresh output root"):
        run_manifest(manifest, tmp_path, output)
    with pytest.raises(ValueError, match="identity differs"):
        verify_manifest(manifest, tmp_path, output)


def test_archived_v1_verifies_but_cannot_resume_and_detects_changed_responses(tmp_path):
    manifest, panel, output = fixture(tmp_path)
    archived_v1(output)
    assert verify_manifest(manifest, tmp_path, output)["verified_queries"] == 2
    with pytest.raises(ValueError, match="choose a fresh output root"):
        run_manifest(manifest, tmp_path, output)
    np.save(tmp_path / "new_response.npy", -np.load(tmp_path / "response.npy"))
    panel["response"] = "new_response.npy"
    manifest.write_text(json.dumps([panel]))
    with pytest.raises(ValueError, match="response-based full_lds differs"):
        verify_manifest(manifest, tmp_path, output)


@pytest.mark.parametrize("location", ["published", "checkpoint"])
@pytest.mark.parametrize("field", ["full_lds", "snr_lds", "secondary_zeta"])
def test_verification_replays_every_reported_correlation(tmp_path, location, field):
    manifest, _, output = fixture(tmp_path)
    archived_v1(output)
    directory = output / "p/fmas_raw"
    if location == "published":
        path = directory / "per_query.json"
        payload = json.loads(path.read_text())
        row = payload[0]
    else:
        path = directory / "queries/query_96.npz"
        with np.load(path) as saved:
            values = {key: saved[key] for key in saved.files}
        payload = json.loads(str(values["metadata"].item()))
        row = payload["row"]
    if field == "secondary_zeta":
        row["by_zeta"]["1.0"]["snr_lds"] += .2
    else:
        row[field] += .2
    if location == "published":
        path.write_text(json.dumps(payload))
    else:
        values["metadata"] = np.asarray(json.dumps(payload))
        np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="response-based"):
        verify_manifest(manifest, tmp_path, output)


def test_input_symlink_cannot_escape_data_root(tmp_path):
    manifest, panel, output = fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.npy"
    try:
        np.save(outside, np.load(tmp_path / "scores.npy"))
        (tmp_path / "escaped.npy").symlink_to(outside)
        panel["methods"][0]["scores"] = "escaped.npy"
        manifest.write_text(json.dumps([panel]))
        with pytest.raises(ValueError, match="resolve below the data root"):
            run_manifest(manifest, tmp_path, output)
    finally:
        outside.unlink(missing_ok=True)
