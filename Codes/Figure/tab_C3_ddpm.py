"""Final Table 10: four methods, 100 matched DDPM queries, fixed-score transforms.

Only the accepted first64 subsets and fixed dampings are used. Projected
D-TRAK/L1 scores are reconstructed from retained features at lambda10; no
new feature sampling, training, query selection or damping search is performed.
"""
import json
import pickle
import numpy as np
import torch
from _common import DATA, lds_per_query, bootstrap_ci, md_table, write_json, write_table
from _paths import METADATA

TRANSFORMS = (("identity", lambda a: a), ("fold", lambda a: a**2),
              ("signed", lambda a: np.sign(a)*a**2))


def main():
    from balds.workflows import build_container
    from balds.workflows.common import _load_ds
    from balds.data.base import balanced_query_indices
    from balds.schema.registry import METHODS
    from balds.schema.estimator import FeatureSet
    container = build_container({"storage": {"data_root": str(DATA), "manifest_db": str(METADATA / "artifacts.sqlite")}}, device="cpu")
    _, test = _load_ds(container.cfg, "cifar2_das")
    with (DATA / "subsets/cifar2_das_masks.pkl").open("rb") as handle:
        masks = pickle.load(handle)[:64]
    records, rows = {}, []
    for track in ("gen", "val"):
        ds = "cifar2_das" + ("_val" if track == "val" else "")
        cols = list(range(100)) if track == "gen" else balanced_query_indices(test.labels, 100)
        gt = np.load(DATA / f"results/gt_matrix_ddpm_{ds}_seed_42.npy")[:64, cols]
        for method, label, damping in (("dtrak_T100", "D-TRAK", 10.), ("l1norm_T100", "DAS", 10.),
                                      ("ekfac_if", "EK-FAC IF", 1e-12), ("fmas_raw", "FMAS", 1e-7 if track == "gen" else 1e-8)):
            if method in ("dtrak_T100", "l1norm_T100"):
                feature = DATA / f"featurize/{method}/cifar2_das/ddpm/seed_42"
                train = torch.load(feature / "train_features.pt", map_location="cpu", weights_only=False)
                query = torch.load(feature / f"query_features_{track}.pt", map_location="cpu", weights_only=False)
                query = torch.as_tensor(query)
                feats = FeatureSet(grads=train, error=None, feat_method=method)
                scores = METHODS.get(method).score(feats, query, damping, device="cpu")[:, cols]
            else:
                source = DATA / f"scores/{method}/{ds}/ddpm/seed_42"
                candidates = [p for p in source.glob("scores_lambda_*.npy") if float(p.stem.removeprefix("scores_lambda_")) == damping]
                if len(candidates) != 1:
                    raise FileNotFoundError(f"Expected exactly one {method} {track} damping={damping} score matrix")
                scores = np.load(candidates[0])
            assert scores.shape == (5000, 100), (method, track, scores.shape)
            score64 = scores.astype(np.float64)
            per_query = {name: lds_per_query(fn(score64), gt, masks)[0] for name, fn in TRANSFORMS}
            delta = bootstrap_ci(per_query["fold"] - per_query["identity"], n_bootstrap=1000, seed=42)
            records[f"{method}|{track}"] = {"damping": damping, "query_ids": cols,
                "per_query": {key: value.tolist() for key, value in per_query.items()}, "delta_ci": list(delta)}
            rows.append([label, track] + [f"{100*per_query[k].mean():.2f}" for k, _ in TRANSFORMS]
                        + [f"{100*delta[0]:+.2f} [{100*delta[1]:.2f}, {100*delta[2]:.2f}]"])
            print(method, track, rows[-1][2:])
    write_json("tab_C3_ddpm_matched100", {"subsets": 64, "bootstrap": 1000, "bootstrap_seed": 42, "rows": records})
    write_table("Table10", md_table(["Method", "Track", "Original", "Folded", "Signed", "Delta [95% CI]"], rows),
                "Final Table 10, LDS ×100, fixed dampings, 100 matched queries")

if __name__ == "__main__":
    main()
