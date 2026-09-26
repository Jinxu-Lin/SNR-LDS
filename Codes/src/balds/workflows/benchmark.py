"""Portable array IO and paper evaluation, independent of training services."""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np

from balds.evaluation.background import (
    RULE, SNR_RULE, SNR_ZETAS, benchmark_method_selection,
    evaluate_snr, evaluate_snr_das,
)


def load_array(path):
    """Read numerical inputs; pickle/PT files must be trusted research artifacts."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != 1:
                raise ValueError("multi-array NPZ needs an explicit single-array export")
            return data[data.files[0]]
    if path.suffix in (".pkl", ".pickle"):
        with path.open("rb") as stream:
            return np.asarray(pickle.load(stream))
    if path.suffix in (".pt", ".pth"):
        import torch
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, torch.Tensor):
            return value.numpy()
        return np.asarray(value)
    raise ValueError(f"unsupported array format: {path.suffix}")


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _stored_fit(fit):
    """Persist only the versioned O(N)+101-point SNR cache, never N-point KDE."""
    return {key: value for key, value in fit.items() if key not in ("x", "selected")}


def save_evaluation(result, output, *, provenance=None, save_fits=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", dict(rule=result["rule"], **result["summary"],
                                             provenance=provenance or {}))
    write_json(output / "per_query.json", result["per_query"])
    array_names = ["selected", "full_prediction"]
    array_names += ["selected_by_zeta", "zetas", "snr_prediction",
                    "snr_prediction_by_zeta"]
    temporary = output / f"predictions.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: result[key] for key in array_names})
    temporary.replace(output / "predictions.npz")
    if save_fits:
        for row, fit in zip(result["per_query"], result["fits"]):
            write_json(output / "fits" / f"query_{row['query_id']}.json", _stored_fit(fit))


def evaluate_files(scores, masks, responses, output, *, fit_scores=None,
                   score_space="native", n_subsets=None, query_ids=None,
                   device="cpu", chunk=256, save_fits=False,
                   rule=SNR_RULE, zetas=SNR_ZETAS, primary_zeta=3.0):
    if fit_scores is not None and score_space != "native":
        raise ValueError("use either explicit --fit-scores or --score-space das-presquare")
    s, k, y = load_array(scores), load_array(masks), load_array(responses)
    t = load_array(fit_scores) if fit_scores is not None else None
    ids = list(range(s.shape[1])) if query_ids is None else list(query_ids)
    if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
        raise ValueError("query IDs must be distinct nonnegative column indices")
    s, y = s[:, ids], y[:, ids]
    if t is not None:
        t = t[:, ids]
    if n_subsets is not None:
        if n_subsets < 2 or min(len(k), len(y)) < n_subsets:
            raise ValueError("not enough subset rows for --n-subsets")
        k, y = k[:n_subsets], y[:n_subsets]
    if score_space not in ("native", "das-presquare"):
        raise ValueError("unknown score space")
    if rule != SNR_RULE:
        raise ValueError(f"unsupported evaluation rule: {rule}")
    options = dict(device=device, chunk=chunk, zetas=zetas, primary_zeta=primary_zeta)
    result = (evaluate_snr_das(s, k, y, **options) if score_space == "das-presquare"
              else evaluate_snr(s, k, y, fit_scores=t, **options))
    for row, query in zip(result["per_query"], ids):
        row["query_id"] = query
    save_evaluation(result, output, provenance=dict(score_space=score_space,
                    fit_space="signed_presquare" if score_space == "das-presquare" else
                              "explicit_fit_scores" if t is not None else "native",
                    aggregation_space="native_squared" if score_space == "das-presquare" else "native",
                    query_ids=ids, n_subsets=len(k), rule=rule,
                    zetas=list(map(float, zetas)) if rule == SNR_RULE else None,
                    primary_zeta=float(primary_zeta) if rule == SNR_RULE else None),
                    save_fits=save_fits)
    return result["summary"]


def _relative_input(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("batch manifest inputs must be data-root-relative; rebase historical manifests first")
    if not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError("batch manifest inputs must resolve below the data root")
    return root / path


def _axis_identity(values):
    """Preserve order without repeating a 50,000-element range per checkpoint."""
    ids = list(map(int, values))
    if not ids:
        return []
    step = ids[1] - ids[0] if len(ids) > 1 else 1
    if step and all(value == ids[0] + i * step for i, value in enumerate(ids)):
        return dict(start=ids[0], stop=ids[-1] + step, step=step)
    return ids


def _method_identity(root, panel, item, available, n_subsets, zetas, primary_zeta):
    """Local resume contract; metadata stamps detect replaced numerical inputs.

    This is not a content hash: deliberately preserving both size and mtime
    after changing a file is outside this lightweight resume contract.
    """
    inputs = {}
    for key, value in (("scores", item["scores"]), ("masks", panel["masks"]),
                       ("response", panel["response"])):
        path = _relative_input(root, value)
        stat = path.stat()
        inputs[key] = dict(path=str(path.relative_to(root)), size=stat.st_size,
                           mtime_ns=stat.st_mtime_ns)
    train_ids = panel["train_ids"]
    return dict(
        schema="snr_method_v2", rule=SNR_RULE, panel_id=str(panel["panel_id"]),
        method=item["method"], score_space=item["score_space"],
        source=inputs["scores"]["path"], inputs=inputs,
        axes=dict(train=_axis_identity(train_ids),
                  mask_train=_axis_identity(panel.get("mask_train_ids", train_ids)),
                  score_train=_axis_identity(item["train_ids"]),
                  score_query=_axis_identity(item["query_ids"]),
                  response_query=_axis_identity(panel["response_query_ids"])),
        query_ids=list(map(int, available)), n_train=len(train_ids),
        n_subsets=n_subsets, zetas=list(map(float, zetas)),
        primary_zeta=float(primary_zeta), declared_identity=item.get("identity", {}))


def _save_query_checkpoint(path, result, identity):
    """Publish one self-contained query result with a single atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(json_value(dict(
        schema="snr_query_checkpoint_v1", identity=identity,
        row=result["per_query"][0], fit=_stored_fit(result["fits"][0]))),
        sort_keys=True, allow_nan=False)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, metadata=np.asarray(metadata), zetas=result["zetas"],
            selected_by_zeta=result["selected_by_zeta"][:, :, 0],
            full_prediction=result["full_prediction"][:, 0],
            snr_prediction_by_zeta=result["snr_prediction_by_zeta"][:, :, 0])
    temporary.replace(path)


def _load_query_checkpoint(path, identity, n_train, n_subsets, zetas):
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata.get("schema") != "snr_query_checkpoint_v1" or metadata.get("identity") != identity:
                raise ValueError("query checkpoint identity differs; choose a fresh output root")
            if (saved["selected_by_zeta"].shape != (len(zetas), n_train)
                    or saved["full_prediction"].shape != (n_subsets,)
                    or saved["snr_prediction_by_zeta"].shape != (len(zetas), n_subsets)
                    or not np.array_equal(saved["zetas"], np.asarray(zetas, dtype=float))):
                raise ValueError("query checkpoint shape or zeta contract differs")
            return (metadata["row"], metadata["fit"],
                    saved["selected_by_zeta"].copy(), saved["full_prediction"].copy(),
                    saved["snr_prediction_by_zeta"].copy())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid query checkpoint: {path}") from error


def _run_snr_method(matrix, masks, response, query_ids, score_space, directory,
                    identity, *, zetas, primary_zeta, device, chunk, save_fits):
    """Evaluate/resume at atomic query boundaries and publish method summaries."""
    zs = tuple(map(float, zetas))
    primary_index = zs.index(float(primary_zeta))
    rows, fits, selected, full, snr = [], [], [], [], []
    for column, query_id in enumerate(query_ids):
        query_identity = dict(identity, query_id=int(query_id), column=int(column))
        checkpoint = directory / "queries" / f"query_{query_id}.npz"
        if checkpoint.is_file():
            row, fit, chosen, full_prediction, snr_prediction = _load_query_checkpoint(
                checkpoint, query_identity, matrix.shape[0], len(masks), zs)
        else:
            options = dict(device=device, chunk=chunk, zetas=zs,
                           primary_zeta=primary_zeta)
            result = (evaluate_snr_das(matrix[:, column:column + 1], masks,
                                       response[:, column:column + 1], **options)
                      if score_space == "das-presquare" else
                      evaluate_snr(matrix[:, column:column + 1], masks,
                                   response[:, column:column + 1], **options))
            result["per_query"][0]["query_id"] = int(query_id)
            _save_query_checkpoint(checkpoint, result, query_identity)
            row, fit = result["per_query"][0], result["fits"][0]
            chosen = result["selected_by_zeta"][:, :, 0]
            full_prediction = result["full_prediction"][:, 0]
            snr_prediction = result["snr_prediction_by_zeta"][:, :, 0]
        rows.append(row); fits.append(fit); selected.append(chosen)
        full.append(full_prediction); snr.append(snr_prediction)
    selected = np.stack(selected, axis=2)
    full = np.stack(full, axis=1)
    snr = np.stack(snr, axis=2)
    valid = [row for row in rows if row["snr_lds"] is not None]
    summary = dict(
        n_total=len(rows), n_valid=len(valid), n_failed=len(rows) - len(valid),
        n_fit_failed=sum(row["fit_status"] == "fit_failed" for row in rows),
        n_all_zero=sum(row["fit_status"] == "all_zero_empty" for row in rows),
        n_empty=sum(row["n_selected"] == 0 for row in valid),
        n_constant_prediction=sum(row["constant_prediction"] for row in valid),
        n_constant_response=sum(row["constant_response"] for row in valid),
        full_lds=float(np.mean([row["full_lds"] for row in valid])) if valid else None,
        snr_lds=float(np.mean([row["snr_lds"] for row in valid])) if valid else None,
        primary_zeta=float(primary_zeta), zetas=list(zs))
    result = dict(rule=SNR_RULE, per_query=rows, summary=summary,
                  selected=selected[primary_index], selected_by_zeta=selected,
                  zetas=np.asarray(zs), full_prediction=full,
                  snr_prediction=snr[primary_index], snr_prediction_by_zeta=snr,
                  fits=fits)
    save_evaluation(result, directory, provenance=identity, save_fits=save_fits)
    write_json(directory / "completion.json", dict(
        schema="snr_method_completion_v1", complete=True, identity=identity,
        n_queries=len(rows)))
    return result


def run_manifest(manifest, data_root, output, *, device="cpu", chunk=256,
                 save_fits=False, rule=SNR_RULE, zetas=SNR_ZETAS,
                 primary_zeta=3.0):
    """Evaluate panels with explicit training/query IDs; missing cells stay missing.

    Input shape follows the historical panel manifest, but paths are relative
    and each method declares ``score_space``: native or das-presquare. The old
    ambiguous ``score_representation`` field cannot change the new protocol.
    """
    if rule != SNR_RULE:
        raise ValueError("batch requires the final SNR-LDS rule")
    root, out = Path(data_root), Path(output)
    panels = json.loads(Path(manifest).read_text())
    if isinstance(panels, dict):
        panels = panels["panels"]
    all_summaries = []
    for panel in panels:
        pid = str(panel["panel_id"])
        if Path(pid).name != pid or pid in (".", ".."):
            raise ValueError("invalid panel ID")
        shared = [_relative_input(root, panel[k]) for k in ("masks", "response")]
        if any(not p.is_file() for p in shared):
            state = dict(panel_id=pid, complete=False, status="missing_input",
                         missing=[str(p.relative_to(root)) for p in shared if not p.is_file()])
            write_json(out / pid / "panel.json", state)
            all_summaries.append(state)
            continue
        masks, response = (load_array(p) for p in shared)
        n = int(panel.get("n_subsets", len(masks)))
        if min(len(masks), len(response)) < n:
            raise ValueError("manifest requests more subset rows than available")
        train_ids = panel["train_ids"]
        if panel.get("mask_train_ids", train_ids) != train_ids:
            raise ValueError("mask training-row identity differs")
        ids = panel["query_ids"]
        response_ids = panel["response_query_ids"]
        if any(len(set(v)) != len(v) for v in (train_ids, ids, response_ids)):
            raise ValueError("duplicate training/query IDs")
        rcols = [response_ids.index(q) for q in ids]
        rows, summaries = [], []
        for item in panel["methods"]:
            method = item["method"]
            if Path(method).name != method or method in (".", ".."):
                raise ValueError("invalid method ID")
            path = _relative_input(root, item["scores"])
            if not path.is_file():
                rows.extend(dict(method=method, query_id=q, status="missing_input",
                                 reason=str(path.relative_to(root)), full_lds=None, snr_lds=None) for q in ids)
                summaries.append(dict(method=method, status="missing_input", n_valid=0, n_total=len(ids)))
                continue
            if item["train_ids"] != train_ids or len(set(item["query_ids"])) != len(item["query_ids"]):
                raise ValueError("score identity differs from panel identity")
            available = [q for q in ids if q in item["query_ids"]]
            rows.extend(dict(method=method, query_id=q, status="missing_query",
                             reason="not in score columns", full_lds=None, snr_lds=None)
                        for q in ids if q not in available)
            if not available:
                summaries.append(dict(method=method, status="missing_query", n_valid=0, n_total=len(ids)))
                continue
            matrix = load_array(path)
            if matrix.shape != (len(train_ids), len(item["query_ids"])):
                raise ValueError("score matrix shape does not match declared IDs")
            matrix = matrix[:, [item["query_ids"].index(q) for q in available]]
            y = response[:n, [response_ids.index(q) for q in available]]
            space = item.get("score_space")
            if space not in ("native", "das-presquare"):
                raise ValueError("each method must declare score_space: native or das-presquare")
            if method in ("das_T100", "das_native_sq") and space != "das-presquare":
                raise ValueError("paper DAS requires signed pre-square input, never V3 linear aggregation")
            method_identity = _method_identity(root, panel, item, available, n,
                                                zetas, primary_zeta)
            method_dir = out / pid / method
            completion = method_dir / "completion.json"
            if completion.is_file():
                state = json.loads(completion.read_text())
                if (state.get("schema") != "snr_method_completion_v1"
                        or not state.get("complete") or state.get("identity") != method_identity):
                    raise ValueError(f"completed method identity differs: {pid}/{method}; "
                                     "choose a fresh output root")
                summary_payload = json.loads((method_dir / "summary.json").read_text())
                result = dict(summary={key: value for key, value in summary_payload.items()
                                      if key not in ("rule", "provenance")},
                              per_query=json.loads((method_dir / "per_query.json").read_text()))
            else:
                result = _run_snr_method(
                    matrix, masks[:n], y, available, space, method_dir,
                    method_identity, zetas=zetas, primary_zeta=primary_zeta,
                    device=device, chunk=chunk, save_fits=save_fits)
            result["summary"].update(n_total=len(ids), n_evaluated=len(available),
                                     n_missing_query=len(ids)-len(available))
            for row, q in zip(result["per_query"], available):
                row.update(method=method, query_id=q)
            rows.extend(result["per_query"])
            summaries.append(dict(method=method, **result["summary"]))
        state = dict(panel_id=pid, rule=rule, complete=not any(
            r["status"].startswith("missing") for r in rows), rows=rows, summary=summaries,
                     not_applicable=panel.get("not_applicable", []))
        write_json(out / pid / "panel.json", state)
        all_summaries.append(dict(panel_id=pid, complete=state["complete"], summary=summaries))
    write_json(out / "panels.json", all_summaries)
    return all_summaries


def deletion(evaluations, utilities, output):
    evaluation_path = Path(evaluations)
    if evaluation_path.is_dir():
        rows = []
        for model_seed in (42, 123, 456):
            panel = evaluation_path / f"cifar2_5k_s{model_seed}_gen" / "panel.json"
            payload = json.loads(panel.read_text())
            rows.extend(dict(row, seed=model_seed) for row in payload["rows"]
                        if row.get("method") in
                        ("fmas_raw", "dtrak_T100", "das_native_sq", "ekfac_if"))
    else:
        payload = json.loads(evaluation_path.read_text())
        rows = None
    if rows is not None:
        pass
    elif isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
    elif isinstance(payload, dict):
        rows = []
        for method, directory in payload.items():
            directory = Path(directory)
            if not directory.is_absolute():
                directory = evaluation_path.parent / directory
            records = json.loads((directory / "per_query.json").read_text())
            rows.extend(dict(row, method=method) for row in records)
    else:
        rows = payload
    with Path(utilities).open() as stream:
        measured = list(csv.DictReader(stream, delimiter="\t"))
    result = benchmark_method_selection(rows, measured)
    result["rule"] = SNR_RULE
    write_json(Path(output) / "edel_selection.json", result)
    return result


def manifest_status(manifest, output):
    panels = json.loads(Path(manifest).read_text())
    panels = panels.get("panels", panels) if isinstance(panels, dict) else panels
    root = Path(output)
    states = []
    for panel in panels:
        panel_state_path = root / panel["panel_id"] / "panel.json"
        panel_state = json.loads(panel_state_path.read_text()) if panel_state_path.is_file() else {}
        for method in panel["methods"]:
            directory = root / panel["panel_id"] / method["method"]
            recorded = next((item for item in panel_state.get("summary", [])
                             if item.get("method") == method["method"]), {})
            state = ("complete" if (directory / "completion.json").is_file() else
                     recorded.get("status", "pending"))
            states.append(dict(
                panel_id=panel["panel_id"], method=method["method"],
                query_checkpoints=len(list((directory / "queries").glob("query_*.npz"))),
                expected_queries=len(set(panel["query_ids"]) & set(method["query_ids"])),
                state=state, complete=state == "complete"))
    return dict(rule=SNR_RULE, cells=states,
                complete=sum(state["complete"] for state in states),
                missing=sum(state["state"].startswith("missing") for state in states),
                pending=sum(state["state"] == "pending" for state in states), total=len(states))


def _verify_query_metrics(row, fit, selected, full, snr, response, zetas, primary_zeta):
    """Check published statistics against current responses, without fitting."""
    from balds.evaluation.lds import compute_lds

    def equal(actual, expected, field):
        if expected is None:
            if actual is not None:
                raise AssertionError(f"{field} must be null for a failed fit")
        elif actual is None:
            raise AssertionError(f"missing {field}")
        else:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                                       err_msg=f"response-based {field} differs")

    equal(row["full_lds"], float(compute_lds(response[:, None], full[:, None])[0][0]),
          "full_lds")
    valid = fit["status"] != "fit_failed"
    for zi, zeta in enumerate(zetas):
        expected = (float(compute_lds(response[:, None], snr[zi, :, None])[0][0])
                    if valid else None)
        entry = row["by_zeta"][str(float(zeta))]
        equal(entry["snr_lds"], expected, f"snr_lds at zeta={zeta}")
        equal(entry["n_selected"], int(selected[zi].sum()) if valid else None,
              f"n_selected at zeta={zeta}")
        if float(zeta) == float(primary_zeta):
            equal(row["snr_lds"], expected, "primary snr_lds")
            equal(row["n_selected"], entry["n_selected"], "primary n_selected")
    if row["fit_status"] != fit["status"]:
        raise AssertionError("published fit status differs")
    if row["constant_response"] != bool(np.ptp(response) == 0):
        raise AssertionError("published constant response status differs")


def verify_manifest(manifest, data_root, output, *, zetas=SNR_ZETAS,
                    primary_zeta=3.0):
    """Replay selection, predictions and response-based LDS without fitting KDE.

    Archived v1 metadata remains readable. New v2 results additionally enforce
    the input/axis resume contract, while both versions are checked against
    the current aligned response values and published per-query statistics.
    """
    from balds.evaluation.lds import predicted_influence
    root, out = Path(data_root), Path(output)
    panels = json.loads(Path(manifest).read_text())
    panels = panels.get("panels", panels) if isinstance(panels, dict) else panels
    errors, verified, missing = [], 0, []
    tmp = [str(path) for path in out.rglob("*.tmp")]
    if tmp:
        errors.append(f"temporary files remain: {tmp[:5]}")
    zs = tuple(map(float, zetas))
    if float(primary_zeta) not in zs:
        raise ValueError("primary_zeta must be present in zetas")
    for panel in panels:
        shared = [_relative_input(root, panel[key]) for key in ("masks", "response")]
        if any(not path.is_file() for path in shared):
            missing.append(dict(panel_id=panel["panel_id"], reason="missing shared input"))
            continue
        masks, response = (np.asarray(load_array(path)) for path in shared)
        n_subsets = int(panel.get("n_subsets", len(masks)))
        if (masks.ndim != 2 or masks.shape[1] != len(panel["train_ids"])
                or response.ndim != 2 or response.shape[1] != len(panel["response_query_ids"])
                or n_subsets < 2 or min(len(masks), len(response)) < n_subsets
                or not np.isin(masks, [0, 1]).all() or not np.isfinite(response).all()):
            raise ValueError("invalid mask/response identity, shape or numerical values")
        if (panel.get("mask_train_ids", panel["train_ids"]) != panel["train_ids"]
                or any(len(set(panel[key])) != len(panel[key])
                       for key in ("train_ids", "query_ids", "response_query_ids"))):
            raise ValueError("invalid training/query identity")
        masks = masks[:n_subsets]
        for item in panel["methods"]:
            path = _relative_input(root, item["scores"])
            if not path.is_file():
                missing.append(dict(panel_id=panel["panel_id"], method=item["method"],
                                    reason="missing score input"))
                continue
            available = [query for query in panel["query_ids"] if query in item["query_ids"]]
            if (item["train_ids"] != panel["train_ids"]
                    or len(set(item["query_ids"])) != len(item["query_ids"])):
                raise ValueError("score identity differs from panel identity")
            if not available:
                missing.append(dict(panel_id=panel["panel_id"], method=item["method"],
                                    reason="no available query columns"))
                continue
            # Match evaluate_snr[_das]: convert before squaring/normalizing.
            # Squaring float32 first changes the source algebra, not merely
            # the final summation tolerance.
            matrix = np.asarray(load_array(path), dtype=np.float64)
            if (matrix.shape != (len(panel["train_ids"]), len(item["query_ids"]))
                    or not np.isfinite(matrix).all()):
                raise ValueError("score matrix shape or numerical values differ")
            columns = [item["query_ids"].index(query) for query in available]
            matrix = matrix[:, columns]
            aligned_response = response[:n_subsets, [panel["response_query_ids"].index(query)
                                                     for query in available]]
            if item["score_space"] not in ("native", "das-presquare"):
                raise ValueError("unknown score space")
            aggregate = matrix * matrix if item["score_space"] == "das-presquare" else matrix
            directory = out / panel["panel_id"] / item["method"]
            if not (directory / "completion.json").is_file():
                errors.append(f"incomplete available cell: {panel['panel_id']}/{item['method']}")
                continue
            completion = json.loads((directory / "completion.json").read_text())
            identity = completion.get("identity", {})
            if (completion.get("schema") != "snr_method_completion_v1"
                    or not completion.get("complete")
                    or identity.get("schema") not in ("snr_method_v1", "snr_method_v2")
                    or identity.get("rule") != SNR_RULE
                    or identity.get("query_ids") != available
                    or identity.get("n_train") != len(matrix)
                    or identity.get("n_subsets") != len(masks)
                    or identity.get("score_space") != item["score_space"]
                    or identity.get("source") != str(path.relative_to(root))
                    or identity.get("zetas") != list(zs)
                    or identity.get("primary_zeta") != float(primary_zeta)):
                errors.append(f"completion identity differs: {panel['panel_id']}/{item['method']}")
                continue
            if identity["schema"] == "snr_method_v2" and identity != _method_identity(
                    root, panel, item, available, n_subsets, zs, primary_zeta):
                errors.append(f"input/axis identity differs: {panel['panel_id']}/{item['method']}")
                continue
            try:
                published = json.loads((directory / "per_query.json").read_text())
                published_by_id = {row["query_id"]: row for row in published}
                if len(published_by_id) != len(published) or set(published_by_id) != set(available):
                    raise ValueError("published per-query coverage differs")
            except (OSError, KeyError, ValueError) as error:
                errors.append(f"{panel['panel_id']}/{item['method']}: {error}")
                continue
            for column, query in enumerate(available):
                try:
                    row, fit, selected, full, snr = _load_query_checkpoint(
                        directory / "queries" / f"query_{query}.npz",
                        dict(identity, query_id=int(query), column=column), len(matrix), len(masks), zs)
                    # Replay the producer's row-wise dot products as well as
                    # its dtype. A matrix-vector BLAS reduction has a different
                    # rounding order at large score scales. Keep strict atol.
                    expected_full = predicted_influence(
                        aggregate[:, column:column + 1], masks, range(len(masks)))[:, 0]
                    np.testing.assert_allclose(full, expected_full, rtol=0, atol=1e-12)
                    if fit["status"] != "fit_failed":
                        fitting = matrix[:, column]
                        x = (np.zeros_like(fitting) if fit["rms"] == 0
                             else fitting / float(fit["rms"]))
                        sigma_x = fit["params"].get("sigma_x")
                        expected = np.zeros((len(zs), len(x)), dtype=bool) if sigma_x is None else np.stack(
                            [np.abs(x) / sigma_x > zeta for zeta in zs])
                        np.testing.assert_array_equal(selected, expected)
                        expected_prediction = np.stack(
                            [predicted_influence(
                                np.where(chosen, aggregate[:, column], 0.0)[:, None],
                                masks, range(len(masks)))[:, 0]
                             for chosen in expected])
                        np.testing.assert_allclose(snr, expected_prediction, rtol=0, atol=1e-12)
                    if row["query_id"] != query:
                        raise AssertionError("query ID differs")
                    for record in (row, published_by_id[query]):
                        _verify_query_metrics(record, fit, selected, full, snr,
                                              aligned_response[:, column], zs, primary_zeta)
                    verified += 1
                except (ValueError, AssertionError, OSError, KeyError) as error:
                    errors.append(f"{panel['panel_id']}/{item['method']}/q{query}: {error}")
    result = dict(rule=SNR_RULE, verdict="pass" if not errors else "fail",
                  verified_queries=verified, missing=missing, errors=errors)
    if errors:
        raise ValueError("SNR verification failed: " + errors[0])
    return result
