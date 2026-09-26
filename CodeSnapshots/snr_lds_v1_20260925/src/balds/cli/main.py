"""``balds`` command-line entry point (Presentation layer).

Builds a :class:`~balds.workflows.container.Container` and dispatches to use-cases, then
renders results.
"""
from __future__ import annotations

import argparse
import json

from balds import report
from balds.workflows import build_container
from balds.schema.registry import METHODS
from balds.schema.runspec import RunSpec


def _add_identity(p, *, method=True, query_choices=("gen", "val", "inject")):
    if method:
        p.add_argument("--method", required=True)
    p.add_argument("--dataset", default="cifar2_5k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-type", dest="query_type", default="gen",
                   choices=list(query_choices))
    p.add_argument("--process", default="cfm", choices=["cfm", "ddpm"])
    p.add_argument("--uncond", action="store_true",
                   help="unconditional model identity ([b] 2026-08-09): RunSpec.conditional=False; "
                        "checkpoint/generation paths use the _uncond segment (e.g. cfm_uncond)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="balds-run", description="Paper model, attribution and response workflows")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", help="artifact root (or BALDS_DATA_ROOT)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="dotted config override, e.g. datasets.raw_dirs.cifar=/path; "
                        "a value starting with [ is a JSON list, e.g. "
                        "ekfac.blockshrink_grid=[1e-6,1e-5,1e-4]")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="compute attribution scores (lambda sweep + select)")
    _add_identity(ps)
    ps.add_argument("--proj-dim", dest="proj_dim", type=int, default=None,
                    help="HP-p only: read the features of this projection dimension and write the "
                         "scores under the matching segment")
    ps.add_argument("--proj-seed", dest="proj_seed", type=int, default=None)
    ps.add_argument("--rescore", action="store_true")

    pe = sub.add_parser("evaluate", help="LDS + bootstrap CI of saved scores")
    _add_identity(pe)
    pe.add_argument("--proj-dim", dest="proj_dim", type=int, default=None,
                    help="HP-p only: evaluate the scores stored under this projection segment")
    pe.add_argument("--proj-seed", dest="proj_seed", type=int, default=None)
    pe.add_argument("--config-tag", dest="config_tag", default=None,
                    help="direct-baseline scientific configuration identity")
    pe.add_argument("--kappas", default=None,
                    help="comma-separated support fractions for LDS@kappa (e.g. 0.05,0.1,1)")

    ppw = sub.add_parser("parameter-weighting",
                         help="E4 parameter-weighted D-TRAK: prepare group contributions, fit, score")
    ppw.add_argument("action", choices=["prepare", "fit", "score"])
    _add_identity(ppw, method=False, query_choices=("gen", "val"))
    ppw.add_argument("--config-tag", required=True)
    ppw.add_argument("--force", action="store_true",
                     help="prepare/fit: replace existing precompute; score: rescore")

    pabu = sub.add_parser("abu", help="E4 AbU+: prepare baseline/Fisher or score finite updates")
    pabu.add_argument("action", choices=["prepare", "score"])
    _add_identity(pabu, method=False, query_choices=("gen", "val"))
    pabu.add_argument("--config-tag", required=True)
    pabu.add_argument("--force", action="store_true",
                      help="prepare: replace existing precompute; score: rescore")

    pnda = sub.add_parser("nda", help="E4 nonparametric patch attribution")
    pnda.add_argument("action", choices=["score"])
    _add_identity(pnda, method=False, query_choices=("gen", "val"))
    pnda.add_argument("--config-tag", required=True)
    pnda.add_argument("--force", action="store_true", help="recompute an existing score matrix")

    pt = sub.add_parser("train", help="stage 02: train the full model")
    _add_identity(pt, method=False)
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--force", action="store_true")

    pg = sub.add_parser("generate", help="stage 03: generate Q query images")
    _add_identity(pg, method=False)
    pg.add_argument("--Q", type=int, default=None)
    pg.add_argument("--ode-steps", dest="ode_steps", type=int, default=None,
                    help="sampling steps (default: platform sampler default; 100 on pixel CFM)")
    pg.add_argument("--pool", type=int, default=None,
                    help="build an INJECT screening pool instead of the LDS query artifact")
    pg.add_argument("--tag", default=None,
                    help="pool identity tag; default <sampler><steps>_n<N>")
    pg.add_argument("--sampler", choices=["ddim", "ancestral", "euler"], default=None)
    pg.add_argument("--steps", type=int, default=None,
                    help="pool sampler steps (DDIM 50 / Euler 100 by default)")
    pg.add_argument("--eta", type=float, default=0.0,
                    help="DDIM stochasticity; the INJECT protocol uses 0")
    pg.add_argument("--gen-seed", dest="gen_seed", type=int, default=None,
                    help="pool default is inject.pool.gen_seed; LDS default remains base + seed")
    pg.add_argument("--artifact-name", dest="artifact_name", default=None,
                    help="store an independent named query generation (for example "
                         "weight_learning_samples) without replacing samples.pt")
    pg.add_argument("--batch", type=int, default=None,
                    help="pool or ordinary pixel-CFM forward batch; CFM draws the full initial noise "
                         "once before slicing; default ordinary generation keeps its original full batch")
    pg.add_argument("--store-latents", action="store_true",
                    help="latent pools only: retain fp16 sampled latents beside images_u8")
    pg.add_argument("--cf", default=None,
                    help="INJECT counterfactual drop_tag; requires --pool")
    pg.add_argument("--trajectory", action="store_true",
                    help="Journey-TRAK ([e]) prerequisite: instead of generating, REPLAY this "
                         "identity's existing generation and store the ODE's intermediate "
                         "states. Verifies the replay lands on the stored samples, so the "
                         "journey provably belongs to the queries being attributed.")
    pg.add_argument("--journey-points", dest="journey_points", type=int, default=None,
                    help="how many points along the trajectory to keep (default lds.journey_points)")
    pg.add_argument("--force", action="store_true")

    pf = sub.add_parser("featurize", help="stage 06: project per-sample gradients -> feature tensors")
    _add_identity(pf, method=False)
    from balds.workflows.common import EMBED_FEATS as _EMB, FEAT_CHOICES as _FR
    pf.add_argument("--feat", default="das", choices=_FR,
                    help="readout + timestep count; the unsuffixed names are T=10. "
                         f"{'/'.join(_EMB)} are non-gradient embeddings ([k]) and "
                         "ignore --proj-dim/--split repeat (no model, no noise)")
    pf.add_argument("--split", default="train", choices=["train", "query", "both", "repeat"],
                    help="train = train gradients + e_n; query = --query-type side; both; "
                         "repeat = R re-featurizations of M train samples for the shrinkage "
                         "layer's sigma-hat (needs the same --proj-dim as the features it calibrates)")
    pf.add_argument("--max-samples", dest="max_samples", type=int, default=None,
                    help="cap #samples (debug/smoke only; leave unset for full featurization)")
    pf.add_argument("--proj-dim", dest="proj_dim", type=int, default=None,
                    help="HP-p only: featurize at this projection dimension and store it under a "
                         "p{dim}s{seed} segment so the rungs never overwrite each other. "
                         "Unset = the configured default with no segment (production path).")
    pf.add_argument("--proj-seed", dest="proj_seed", type=int, default=None,
                    help="HP-p only: projection seed; same segmenting rule as --proj-dim")
    pf.add_argument("--ckpt-step", dest="ckpt_step", type=int, default=None,
                    help="TracInCP/GAS only: featurize a MID-TRAINING checkpoint "
                         "(step_{N}.pt) instead of the final model, stored under a "
                         "step_{N} segment. Unset = the final model (production path). "
                         "`balds-run checkpoints` lists the steps for an identity.")
    pf.add_argument("--no-error-weight", action="store_true",
                    help="train/both only: skip e_n, unused by registered INJECT methods")
    pf.add_argument("--force", action="store_true")

    pck = sub.add_parser("checkpoints",
                         help="list the training checkpoints TracInCP/GAS would average over")
    _add_identity(pck, method=False)

    pl = sub.add_parser(
        "latents",
        help="latent platform stage 01: VAE-encode the dataset (both "
             "horizontal orientations for the train split) + precompute per-style "
             "prompt embeddings. One-off per dataset; needs the raw 256px image "
             "folder (datasets.raw_dirs.artbench) and the platform weights. Every "
             "later stage reads these caches, never the pixels.")
    pl.add_argument("--dataset", default="artbench2_256")
    pl.add_argument("--force", action="store_true")

    pinj = sub.add_parser(
        "inject", help="contamination-injection benchmark construction and evaluation")
    pinj.add_argument("action", choices=[
        "build", "detector-train", "screen", "review-export",
        "queries", "evaluate", "compare", "mine"])
    pinj.add_argument("--dataset", default="cifar10_inj4")
    pinj.add_argument("--process", choices=["cfm", "ddpm"], default="ddpm")
    pinj.add_argument("--seed", type=int, default=42)
    pinj.add_argument("--tag", default=None)
    pinj.add_argument("--detector-dataset", default=None,
                      help="filed detector identity (default: --dataset)")
    pinj.add_argument("--threshold", type=float, default=None)
    pinj.add_argument("--detector-tag", default=None,
                      help="detector-train/screen/mine: tagged detector "
                           "results/inject/<ds>/detector_<tag>.pt (default: the legacy detector.pt)")
    pinj.add_argument("--detector-config", default="detector",
                      help="detector-train: config block inject.<name> holding the recipe "
                           "(arch, data_mode, holdout_frac, ...); non-legacy recipes need --detector-tag")
    pinj.add_argument("--mine-tag", default=None,
                      help="mine/review-export/queries: mined candidate set "
                           "generations/<ds>/<model>/seed_<n>/mine_<tag>.pt")
    pinj.add_argument("--target", type=int, default=None,
                      help="mine: kept images per class (default inject.mine.target)")
    pinj.add_argument("--chunk", type=int, default=None,
                      help="mine: generated images per chunk (default inject.mine.chunk)")
    pinj.add_argument("--keep-threshold", type=float, default=None,
                      help="mine: keep when p(10+class) >= this (default inject.mine.keep_threshold)")
    pinj.add_argument("--max-per-class", type=int, default=None,
                      help="mine: generated-image cap per class (default inject.mine.max_per_class)")
    pinj.add_argument("--include-pool", default=None,
                      help="mine: first screen this filed pool tag of the same model")
    pinj.add_argument("--classes", default=None,
                      help="mine: comma-separated host names or labels, in mining order (default all)")
    pinj.add_argument("--batch", type=int, default=None,
                      help="mine: sampler/detector batch (default inject.mine.batch_size)")
    pinj.add_argument("--per-concept", type=int, default=None)
    pinj.add_argument("--random", dest="random_n", type=int, default=None)
    pinj.add_argument("--review", default=None)
    pinj.add_argument("--version", default=None)
    pinj.add_argument("--reviewer", default=None)
    pinj.add_argument("--method", default=None)
    pinj.add_argument("--methods", default=None,
                      help="compare only: exactly two comma-separated method names A,B")
    pinj.add_argument("--split", choices=["test", "val", "all"], default="test")
    pinj.add_argument("--metric", default="precision_at_k_pool")
    pinj.add_argument("--k", type=int, default=None)
    pinj.add_argument("--cf", default=None,
                      help="screen only: counterfactual pool drop_tag")
    pinj.add_argument("--uncond", action="store_true")
    pinj.add_argument("--force", action="store_true")

    pek = sub.add_parser(
        "ekfac",
        help="curvature path (the paper's FMAS + the EK-FAC IF baseline): fit = two MC "
             "passes over the train set -> curvature package (per dataset/seed/process, "
             "query-independent); score = stream per-sample gradients once, evaluate the "
             "damping grid, select via score.lambda_selector, save scores under the method "
             "name (then `balds-run evaluate --method <name>` as usual); repeats = re-score M "
             "train samples R times for sigma-hat. Knobs live under `ekfac.` "
             "(--set ekfac.fit_epochs=... etc.); sampling/damping_mode are BOUND to the "
             "method name and a conflicting --set is refused.")
    pek.add_argument("action", choices=["fit", "score", "repeats"])
    _add_identity(pek, method=False)
    pek.add_argument("--method", default="ekfac_if",
                     help="fmas/fmas_raw: stratified-antithetic MC and layer damping; "
                          "ekfac_if: iid MC and global damping")
    pek.add_argument("--force", action="store_true",
                     help="fit/repeats: recompute even if the artifact exists")
    pek.add_argument("--rescore", action="store_true",
                     help="score: recompute even if scores exist (raw method = full re-stream, "
                          "ignoring and overwriting this machine's block partials; refused with "
                          "--world > 1 or --row-world > 1)")
    pek.add_argument("--rank", type=int, default=0,
                     help="score/repeats: this shard's index on the QUERY axis; process only "
                          "query blocks whose index %% world == rank. Blocks are the "
                          "ekfac.query_chunk tiling of [0, Q), each persisted as it finishes, "
                          "so a killed run resumes per block and N cards/machines can split one "
                          "cell. Whichever run finds every block complete assembles and files "
                          "scores_lambda/scores (bit-identical to the single-process run).")
    pek.add_argument("--world", type=int, default=1,
                     help="score/repeats: number of shards splitting the query blocks "
                          "(1 = single process). Needs ekfac.query_chunk <= Q/world to have "
                          "more than one block to hand out. NOTE each query block re-streams "
                          "the whole train side, so this axis multiplies total work by `world`; "
                          "prefer --row-world when you just want more cards.")
    pek.add_argument("--row-rank", dest="row_rank", type=int, default=0,
                     help="score/repeats: this shard's index on the TRAIN-ROW axis, one level "
                          "below --rank. Every query block this process handles is restricted "
                          "to the row tiles whose index %% row_world == row_rank; tiles are the "
                          "ekfac.row_chunk tiling of [0, N) (of [0, M) for repeats). Row i's "
                          "gradient is keyed by its GLOBAL index, so a row tile is bit-identical "
                          "however it was split — and unlike --rank this axis re-runs only the "
                          "block's query pass, not the train stream.")
    pek.add_argument("--row-world", dest="row_world", type=int, default=1,
                     help="score/repeats: number of shards splitting each query block's train "
                          "rows (1 = the whole row range in one partial, the pre-2026-09-09 "
                          "shape). Needs ekfac.row_chunk <= N/row_world. Combines with "
                          "--rank/--world: 8 cards = --world 4 x --row-world 2.")
    pek.add_argument("--clean-partials", dest="clean_partials", action="store_true",
                     help="score/repeats: after the whole matrix is assembled and filed, delete "
                          "THIS machine's block partials (they are kept by default — cheap, and "
                          "they make a re-file free)")

    pcf = sub.add_parser(
        "counterfactual",
        help="counterfactual chain: build = top-k removal sets per method (fmas / "
             "dtrak_T100 per query, plus n-random query-independent sets); retrain = one "
             "model per set (rank/world shardable, idempotent; LoRA on the SD3.5 platform, "
             "full UNet on CIFAR); regenerate = replay the stored generation (same "
             "seed/sampler/steps) under each counterfactual model; verify = replay with "
             "the undeleted model and demand bit-identity with samples.pt; analyze = "
             "query-loss change + pixel L2 + CLIP + AUC-vs-random + box-plot + "
             "qualitative grid (values comparable across methods only)")
    pcf.add_argument("action", choices=["build", "retrain", "regenerate", "verify", "analyze"])
    pcf.add_argument("--dataset", default="artbench2_256")
    pcf.add_argument("--seed", type=int, default=42)
    pcf.add_argument("--process", default="cfm", choices=["cfm", "ddpm"])
    pcf.add_argument("--uncond", action="store_true",
                     help="unconditional model identity (cifar2_5k); not accepted on the SD3.5 platform")
    pcf.add_argument("--arm", default=None,
                     help="build/analyze: comma-separated methods (default fmas,dtrak_T100,random); "
                          "retrain/regenerate: exactly one arm required")
    pcf.add_argument("--k", type=int, default=1000,
                     help="removed samples per set; every counterfactual artifact is filed under "
                          "a k-specific arm name so k=300 and k=1000 never mix")
    pq = pcf.add_mutually_exclusive_group()
    pq.add_argument("--n-queries", dest="n_queries", type=int, default=None,
                    help="build: the FIRST n query images get a set (method arms) / are "
                         "regenerated per random set (default 50). Note the generated "
                         "queries are laid out class by class, so a prefix is not a "
                         "sample of the query set -- see --balanced")
    pq.add_argument("--query-indices", dest="query_indices", default=None,
                    help="build: explicit comma-separated query indices (e.g. 0,50,1,51) "
                         "instead of a prefix; recorded in the arm's meta and used by "
                         "retrain/regenerate/analyze")
    pq.add_argument("--balanced", dest="balanced", type=int, default=None,
                    help="build: K queries, K/num_classes per class, taken from the "
                         "generation's labels and stored in the meta as EXPLICIT indices "
                         "(ArtBench-2: --balanced 10 = 5 per style)")
    pcf.add_argument("--n-random", dest="n_random", type=int, default=5,
                     help="build: number of independent random removal sets")
    pcf.add_argument("--fmas-rho", "--fmas-fixed-rho", dest="fmas_rho", type=float,
                     default=None,
                     help="build only, FMAS only: use the raw score at this exact rho from "
                          "fmas_raw/scores_lambda_<rho>.npy; no repeat-score transform")
    pcf.add_argument("--rank", type=int, default=0)
    pcf.add_argument("--world", type=int, default=1)
    pcf.add_argument("--force", action="store_true")
    pcf.add_argument("--no-strict", dest="strict", action="store_false",
                     help="verify: report instead of raising on a failed replay check")
    pcf.add_argument("--l2-tol", dest="l2_tol", type=float, default=None,
                     help="verify: per-query pixel-L2 replay floor allowed for a `tolerant` "
                          "verdict (default counterfactual.replay_l2_tol=0.05); `strict` = "
                          "bit-identical. `regenerate --arm reference` files the undeleted "
                          "model's same-machine replay so analyze needs no floor at all")
    pcf.add_argument("--no-loss", dest="no_loss", action="store_true",
                     help="analyze: skip the query-loss change (needs every counterfactual "
                          "checkpoint loaded once more; on by default)")

    psub = sub.add_parser("subsets", help="stage 04: masks / train subsets / GT")
    psub.add_argument("action", choices=["masks", "train", "losses", "gt"])
    _add_identity(psub, method=False)
    psub.add_argument("--e-seeds", dest="e_seeds", default="0", help="comma-separated GT noise seeds")
    psub.add_argument("--rank", type=int, default=0,
                      help="train/losses: this shard's index; process only subsets m %% world == rank")
    psub.add_argument("--world", type=int, default=1,
                      help="train/losses: number of shards splitting the 64 subsets across GPUs (1 = single process)")
    psub.add_argument("--force", action="store_true",
                      help="losses: recompute even if gt_losses already exist for this (query_type, e_seed)")
    psub.add_argument("--replica", type=int, default=0,
                      help="train/losses/gt: noise-floor duplicate run id ([m]); 0 = main run. "
                           "Same masks, shifted retrain seed, artifacts under a _r{R} suffix")
    psub.add_argument("--chain", type=int, default=None,
                      help="losses (§6.2-14 decoupling): subset-CHAIN seed when it differs "
                           "from --seed — score chain R's subset models on this identity's "
                           "queries (gen-track cross combination for gt_avg). Omitted or "
                           "equal to --seed = the legacy diagonal. val track refuses it "
                           "(chain R's own val losses already are that computation)")
    psub.add_argument("--chains", default=None,
                      help="gt (§6.2-14 gt_avg): comma-separated subset-chain seeds, e.g. "
                           "42,123,456 — average the per-subset losses over these retrain "
                           "chains x --e-seeds and write the result to the standard "
                           "gt_matrix path (THE adjudicated main-table GT switch). Strict: "
                           "every (chain, e_seed) cell must exist")

    pdas = sub.add_parser(
        "import-das",
        help="assimilate the DAS release archive (§6.2-33 ⑦): file its projected "
             "features, 128 subset masks, replica-averaged ground truth and generated "
             "queries under the cifar2_das dataset key, so `balds-run score/evaluate` run on "
             "DAS's own diffusion numbers unchanged. Idempotent; imports nothing that "
             "already exists unless --force.")
    pdas.add_argument("--root", default=None,
                      help="archive root (holding CIFAR2/ and _code_DAS/) or its "
                           "CIFAR2/saved/5000-0.5 sub-tree; default = "
                           "datasets.raw_dirs.das_archive")
    pdas.add_argument("--feats", default=None,
                      help="comma-separated readouts to import (default das,dtrak,l1norm). "
                           "`trak` is opt-in: its dumps carry an undocumented per-timestep "
                           "row axis, see balds.workflows.das_import.TRAK_CAVEAT")
    pdas.add_argument("--model", action="store_true",
                      help="also file the archive's trained DDPM (ddpm/ddpm_42) as the "
                           "cifar2_das checkpoint, so `balds-run ekfac fit/score --dataset "
                           "cifar2_das --process ddpm` can run; refuses a scheduler that "
                           "is not the registered ddpm process")
    pdas.add_argument("--force", action="store_true")

    sub.add_parser("methods", help="list paper attribution methods")
    sub.add_parser("progress", help="summarize local stage progress")
    return p


def coerce_override(key: str, value: str):
    """One ``--set KEY=VALUE`` value as a Python value.

    A value starting with ``[`` is a JSON list literal, so ``1e-6`` (no dot) is a
    number, not the string ``yaml.safe_load`` would make of it. Numbers in a list
    that mixes ints and floats all become floats -- ``[1.0e-6,1e-5,1]`` is a float
    grid -- while an all-int list stays ints. A malformed literal raises naming the
    key; it is never passed on as a string. Every other value keeps the original
    int, then float, then string coercion.
    """
    if value.lstrip().startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--set {key}: malformed list literal {value!r} ({exc.msg}); "
                             f"write e.g. {key}=[1e-6,1e-5] (quote strings)") from None
        if not isinstance(parsed, list):
            raise ValueError(f"--set {key}: {value!r} starts with '[' but is not a list")
        numbers = [v for v in parsed if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if any(isinstance(v, float) for v in numbers):
            parsed = [float(v) if v in numbers and not isinstance(v, bool) else v
                      for v in parsed]
        return parsed
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_overrides(pairs) -> dict:
    """``["a.b=1", ...]`` from repeated ``--set`` flags -> ``{"a.b": 1, ...}``."""
    out = {}
    for kv in pairs:
        key, value = kv.split("=", 1)
        out[key] = coerce_override(key, value)
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "methods":
        from balds.workflows.config import load_config, resolve_paper_method
        print("\n".join(METHODS.names()))
        print(f"fmas -> {resolve_paper_method(load_config({}), 'fmas')}  (paper bilinear score)")
        return 0

    if args.cmd == "progress":
        from balds.workflows.config import load_config
        from balds.workflows.usecases import progress_overview
        cfg = load_config({"storage.data_root": args.data_root} if args.data_root else {})
        infos = progress_overview(cfg)
        if not infos:
            print(f"(no progress files under {cfg['storage']['data_root']}/progress/)")
            return 0
        for i in infos:
            pct = f"{i['frac'] * 100:5.1f}%" if i["frac"] is not None else "    ?"
            eta = (f"ETA {i['eta_s'] // 3600}h{(i['eta_s'] % 3600) // 60:02d}m"
                   if i["eta_s"] else ("done " if i["frac"] == 1.0 else ""))
            metrics = " ".join(f"{k}={v}" for k, v in i["last"].items()
                               if k in ("loss", "lr", "epoch", "mean_loss", "final_loss",
                                        "phase") and v != "")
            print(f"{i['name']:<48s} {pct}  {eta:<10s} @{i['updated']}  {metrics}")
        return 0

    overrides = parse_overrides(args.overrides)
    if args.data_root is not None:
        overrides["storage.data_root"] = args.data_root
    container = build_container(overrides, device=args.device)
    # where this run writes artifacts and reads raw data, before any stage touches
    # either (§6.2-53: a detached checkout looked for raw data inside itself)
    from balds.workflows.config import describe_paths
    from balds.schema.logging import get_logger
    get_logger("balds.cli").info(describe_paths(container.cfg))

    if args.cmd == "inject":
        uc = container.inject_usecase()
        if args.action == "build":
            out = uc.build(args.dataset, force=args.force)
        elif args.action == "detector-train":
            out = uc.detector_train(args.dataset, force=args.force,
                                    detector_tag=args.detector_tag,
                                    detector_config=args.detector_config)
        elif args.action == "mine":
            if not args.mine_tag:
                raise SystemExit("inject mine requires --mine-tag")
            out = uc.mine(args.dataset, args.seed, process=args.process,
                          mine_tag=args.mine_tag, detector_tag=args.detector_tag,
                          detector_dataset=args.detector_dataset, target=args.target,
                          chunk=args.chunk, keep_threshold=args.keep_threshold,
                          max_per_class=args.max_per_class, include_pool=args.include_pool,
                          classes=args.classes, batch=args.batch,
                          conditional=not args.uncond)
        elif args.action == "evaluate":
            if not args.method:
                raise SystemExit("inject evaluate requires --method")
            out = uc.evaluate(args.method, args.dataset, args.seed,
                              process=args.process, split=args.split, k=args.k,
                              conditional=not args.uncond, force=args.force)
            print(report.render_inject(out))
            return 0
        elif args.action == "compare":
            methods = ([value.strip() for value in args.methods.split(",") if value.strip()]
                       if args.methods else [])
            out = uc.compare(methods, args.dataset, args.seed, process=args.process,
                             split=args.split, metric=args.metric,
                             conditional=not args.uncond)
        else:
            mined = args.action in ("review-export", "queries") and args.mine_tag
            if not args.tag and not mined:
                raise SystemExit(f"inject {args.action} requires --tag"
                                 + (" or --mine-tag" if args.action != "screen" else ""))
            if args.tag and mined:
                raise SystemExit(f"inject {args.action} takes --tag or --mine-tag, not both")
            common = dict(process=args.process, tag=args.tag,
                          conditional=not args.uncond)
            if args.action == "screen":
                out = uc.screen(args.dataset, args.seed,
                                detector_dataset=args.detector_dataset,
                                detector_tag=args.detector_tag,
                                threshold=args.threshold, cf=args.cf, **common)
            elif args.action == "review-export":
                out = uc.review_export(args.dataset, args.seed,
                                       per_concept=args.per_concept,
                                       random_n=args.random_n, mine_tag=args.mine_tag,
                                       **common)
            else:
                if not args.review or not args.version:
                    raise SystemExit("inject queries requires --review and --version")
                out = uc.queries(args.dataset, args.seed, review=args.review,
                                 version=args.version, reviewer=args.reviewer,
                                 force=args.force, mine_tag=args.mine_tag, **common)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "score":
        out = container.score_usecase().run(
            args.method, args.dataset, args.seed, query_type=args.query_type,
            process=args.process, conditional=not args.uncond,
            proj_dim=args.proj_dim, proj_seed=args.proj_seed, rescore=args.rescore)
        print(json.dumps(out, indent=2))
    elif args.cmd == "evaluate":
        if args.query_type == "inject":
            raise SystemExit("inject has no LDS ground truth; use `balds-run inject evaluate`")
        kappas = [float(value) for value in args.kappas.split(",")] if args.kappas else None
        out = container.evaluate_usecase().run(
            args.method, args.dataset, args.seed, query_type=args.query_type,
            process=args.process, conditional=not args.uncond,
            proj_dim=args.proj_dim, proj_seed=args.proj_seed, config_tag=args.config_tag,
            kappas=kappas)
        print(report.render_lds(out, method=args.method, dataset=args.dataset,
                                query_type=args.query_type))
    elif args.cmd == "parameter-weighting":
        uc = container.parameter_weighting_usecase()
        common = dict(dataset=args.dataset, seed=args.seed, config_tag=args.config_tag,
                      process=args.process, conditional=not args.uncond)
        if args.action == "prepare":
            out = uc.prepare(**common, force=args.force)
        elif args.action == "fit":
            out = uc.fit(**common, force=args.force)
        else:
            out = uc.score(**common, query_type=args.query_type, rescore=args.force)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "abu":
        uc = container.abu_usecase()
        common = dict(dataset=args.dataset, seed=args.seed, config_tag=args.config_tag,
                      process=args.process, conditional=not args.uncond)
        if args.action == "prepare":
            out = uc.prepare(**common, force=args.force)
        else:
            out = uc.score(**common, query_type=args.query_type, rescore=args.force)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "nda":
        out = container.nda_usecase().score(
            args.dataset, args.seed, config_tag=args.config_tag,
            query_type=args.query_type, process=args.process,
            conditional=not args.uncond, rescore=args.force)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "train":
        out = container.train_usecase().run(args.dataset, args.seed, process=args.process,
                                            steps=args.steps, conditional=not args.uncond,
                                            force=args.force)
        print(json.dumps(out, indent=2))
    elif args.cmd == "generate":
        uc = container.generate_usecase()
        if args.trajectory:
            if args.pool is not None:
                raise SystemExit("--trajectory and --pool are mutually exclusive")
            out = uc.trajectory(args.dataset, args.seed, process=args.process,
                                n_points=args.journey_points,
                                conditional=not args.uncond, force=args.force)
        else:
            out = uc.run(args.dataset, args.seed, process=args.process,
                         Q=args.Q, ode_steps=args.ode_steps,
                         gen_seed=args.gen_seed, conditional=not args.uncond,
                         force=args.force, pool=args.pool, tag=args.tag,
                         sampler=args.sampler, steps=args.steps, eta=args.eta,
                         batch=args.batch, store_latents=args.store_latents,
                         cf=args.cf, artifact_name=args.artifact_name)
        print(json.dumps(out, indent=2))
    elif args.cmd == "featurize":
        out = container.featurize_usecase().run(
            args.dataset, args.seed, process=args.process, feat=args.feat, split=args.split,
            query_type=args.query_type, max_samples=args.max_samples,
            conditional=not args.uncond, proj_dim=args.proj_dim, proj_seed=args.proj_seed,
            ckpt_step=args.ckpt_step, no_error_weight=args.no_error_weight,
            force=args.force)
        print(json.dumps(out, indent=2))
    elif args.cmd == "latents":
        out = container.encode_usecase().run(args.dataset, force=args.force)
        print(json.dumps(out, indent=2))
    elif args.cmd == "ekfac":
        if args.action == "fit":
            out = container.ekfac_fit_usecase().run(
                args.dataset, args.seed, process=args.process,
                conditional=not args.uncond, force=args.force)
        elif args.action == "repeats":
            out = container.ekfac_repeats_usecase().run(
                args.dataset, args.seed, query_type=args.query_type,
                process=args.process, conditional=not args.uncond,
                force=args.force, method=args.method, rank=args.rank, world=args.world,
                row_rank=args.row_rank, row_world=args.row_world,
                clean_partials=args.clean_partials)
        else:
            out = container.ekfac_score_usecase().run(
                args.dataset, args.seed, query_type=args.query_type,
                process=args.process, conditional=not args.uncond,
                rescore=args.rescore, method=args.method, rank=args.rank, world=args.world,
                row_rank=args.row_rank, row_world=args.row_world,
                clean_partials=args.clean_partials)
        print(json.dumps(out, indent=2))
    elif args.cmd == "counterfactual":
        uc = container.counterfactual_usecase()
        arms = [arm.strip() for arm in args.arm.split(",") if arm.strip()] if args.arm else None
        cond = not args.uncond
        if args.fmas_rho is not None and args.action != "build":
            raise SystemExit("--fmas-rho is only valid for `counterfactual build`")
        if args.action == "build":
            qidx = ([int(v) for v in args.query_indices.split(",") if v != ""]
                    if args.query_indices else None)
            out = uc.build(args.dataset, args.seed, process=args.process, conditional=cond,
                           arms=arms, k=args.k, n_queries=args.n_queries,
                           query_indices=qidx, balanced=args.balanced,
                           n_random=args.n_random, fmas_rho=args.fmas_rho,
                           force=args.force)
        elif args.action in ("retrain", "regenerate"):
            if not arms or len(arms) != 1:
                raise SystemExit("exactly one --arm is required for retrain/regenerate")
            fn = uc.retrain if args.action == "retrain" else uc.regenerate
            kw = {"force": args.force} if args.action == "regenerate" else {}
            out = fn(args.dataset, args.seed, arm=args.arm, k=args.k, process=args.process,
                     conditional=cond, rank=args.rank, world=args.world, **kw)
        elif args.action == "verify":
            out = uc.verify(args.dataset, args.seed, process=args.process, conditional=cond,
                            strict=args.strict, l2_tol=args.l2_tol)
        else:
            out = uc.analyze(args.dataset, args.seed, k=args.k, process=args.process,
                             conditional=cond, arms=arms, with_loss=not args.no_loss)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "checkpoints":
        from balds.workflows.checkpoints import attribution_checkpoints
        base = RunSpec(dataset=args.dataset, seed=args.seed, process=args.process,
                       conditional=not args.uncond)
        steps = attribution_checkpoints(container.store, container.cfg, base)
        print(json.dumps({"dataset": args.dataset, "seed": args.seed,
                          "steps": [s for s in steps if s is not None],
                          "tracin_terms": len(steps),
                          "tracin_available": len(steps) > 1}, indent=2))
    elif args.cmd == "subsets":
        uc = container.subsets_usecase()
        if args.action == "masks":
            out = uc.generate_masks(args.dataset, args.seed, conditional=not args.uncond)
        elif args.action == "train":
            out = uc.train_subsets(args.dataset, args.seed, process=args.process,
                                   rank=args.rank, world=args.world,
                                   replica=args.replica, conditional=not args.uncond)
        elif args.action == "losses":
            e_seeds = [int(e) for e in args.e_seeds.split(",")]
            out = {"per_eseed": [uc.compute_losses(args.dataset, args.seed, process=args.process,
                                                   query_type=args.query_type, e_seed=e,
                                                   replica=args.replica, chain=args.chain,
                                                   rank=args.rank, world=args.world,
                                                   conditional=not args.uncond,
                                                   force=args.force) for e in e_seeds]}
        else:  # gt
            chains = ([int(c) for c in args.chains.split(",") if c != ""]
                      if args.chains else None)
            out = uc.compute_gt(args.dataset, args.seed, process=args.process,
                                query_type=args.query_type, replica=args.replica,
                                conditional=not args.uncond, chains=chains,
                                e_seeds=[int(e) for e in args.e_seeds.split(",")])
        print(json.dumps(out, indent=2))
    elif args.cmd == "import-das":
        feats = [f.strip() for f in args.feats.split(",") if f.strip()] if args.feats else None
        out = container.das_import_usecase().run(args.root, force=args.force, feats=feats,
                                                 model=args.model)
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
