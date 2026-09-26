"""Addressing golden tests — zero-migration for CFM + process segment for DDPM.

Guards audit 2026-07-07 gap #2: featurize/scores paths must stay
byte-identical for the legacy CFM case, while non-CFM processes get their own
path segment so matched-DDPM artifacts can never overwrite CFM ones.
Pure stdlib; run with any python: ``python Codes/tests/store/test_addressing.py``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts.addressing import block_dir, parse_block_key, relpath


def test_cfm_paths_unchanged():
    """Legacy CFM paths are the zero-migration contract — golden strings."""
    spec = RunSpec(dataset="cifar2", seed=42, method="das", query_type="gen")
    assert relpath(K.TRAIN_FEATURES, spec, feat="das") == \
        "featurize/das/cifar2/seed_42/train_features.pt"
    assert relpath(K.ERROR_WEIGHT, spec, feat="das") == \
        "featurize/das/cifar2/seed_42/error_train.npy"
    assert relpath(K.QUERY_FEATURES, spec, feat="das") == \
        "featurize/das/cifar2/seed_42/query_features_gen.pt"
    assert relpath(K.SCORES, spec) == "scores/das/cifar2/seed_42/scores.npy"
    assert relpath(K.SCORES_LAMBDA, spec, lam=0.5) == \
        "scores/das/cifar2/seed_42/scores_lambda_5.00e-01.npy"

    val = RunSpec(dataset="cifar2", seed=42, method="das", query_type="val")
    assert relpath(K.SCORES, val) == "scores/das/cifar2_val/seed_42/scores.npy"


def test_ddpm_paths_get_process_segment():
    """Non-CFM attribution artifacts live under their own process segment."""
    cfm = RunSpec(dataset="cifar2", seed=42, method="das", query_type="gen")
    ddpm = RunSpec(dataset="cifar2", seed=42, method="das", query_type="gen",
                   process="ddpm")
    assert relpath(K.TRAIN_FEATURES, ddpm, feat="das") == \
        "featurize/das/cifar2/ddpm/seed_42/train_features.pt"
    assert relpath(K.SCORES, ddpm) == "scores/das/cifar2/ddpm/seed_42/scores.npy"
    for kind, key in ((K.TRAIN_FEATURES, {"feat": "das"}), (K.SCORES, {})):
        assert relpath(kind, cfm, **key) != relpath(kind, ddpm, **key)


def test_gt_and_checkpoint_already_distinguish_process():
    """Pre-existing behaviour (model_key) — pin it so it never regresses."""
    fm = RunSpec(dataset="cifar2", seed=42)
    dd = RunSpec(dataset="cifar2", seed=42, process="ddpm")
    assert relpath(K.GT_MATRIX, fm) == "results/gt_matrix_fm_cifar2_seed_42.npy"
    assert relpath(K.GT_MATRIX, dd) == "results/gt_matrix_ddpm_cifar2_seed_42.npy"
    assert relpath(K.CHECKPOINT, fm) != relpath(K.CHECKPOINT, dd)


def test_replica_and_uncond_paths():
    """[m]/[b] 2026-08-09: replica suffix + uncond segment; absent/0 = legacy."""
    s5 = RunSpec(dataset="cifar2_5k", process="cfm", seed=42, conditional=False)
    # uncond checkpoint segment
    assert relpath(K.CHECKPOINT, s5) == "checkpoints/cifar2_5k/cfm_uncond/seed_42/final.pt"
    assert relpath(K.CHECKPOINT, s5, step=1953) == "checkpoints/cifar2_5k/cfm_uncond/seed_42/step_1953.pt"
    # subset + GT: replica absent -> plain; replica=1 -> _r1; replica=0 -> plain
    assert relpath(K.SUBSET_CHECKPOINT, s5, m=3) == "checkpoints/subsets/fm/cifar2_5k/seed_42/subset_3/final.pt"
    assert relpath(K.SUBSET_CHECKPOINT, s5, m=3, replica=1) == "checkpoints/subsets/fm/cifar2_5k/seed_42_r1/subset_3/final.pt"
    assert relpath(K.SUBSET_CHECKPOINT, s5, m=3, replica=0) == "checkpoints/subsets/fm/cifar2_5k/seed_42/subset_3/final.pt"
    assert relpath(K.GT_LOSSES, s5, eseed=2, replica=1) == "results/gt_losses_fm_cifar2_5k_seed_42_r1_eseed_2.npy"
    assert relpath(K.GT_MATRIX, s5, replica=1) == "results/gt_matrix_fm_cifar2_5k_seed_42_r1.npy"
    assert relpath(K.GT_LOSSROW, s5, eseed=0, m=7, replica=1) == "results/gt_lossrows_fm_cifar2_5k_seed_42_r1_eseed_0/subset_7.npy"
    # legacy cifar2 strings byte-identical when replica untouched
    s0 = RunSpec(dataset="cifar2", process="cfm", seed=42)
    assert relpath(K.GT_LOSSES, s0, eseed=0) == "results/gt_losses_fm_cifar2_seed_42_eseed_0.npy"
    print("PASS test_replica_and_uncond_paths")


def test_latent_platform_paths():
    """[o] SD3.5 platform: latents/prompt embeds are dataset-level; everything
    else (checkpoints/GT/scores) rides the existing rules under the new keys."""
    ab = RunSpec(dataset="artbench10")
    assert relpath(K.LATENTS, ab, split="train") == "latents/artbench10/train.pt"
    assert relpath(K.LATENTS, ab, split="train", flip=True) == "latents/artbench10/train_flip.pt"
    assert relpath(K.LATENTS, ab, split="test") == "latents/artbench10/test.pt"
    assert relpath(K.PROMPT_EMBEDS, ab) == "latents/artbench10/prompt_embeds.pt"
    ck = RunSpec(dataset="artbench10", process="cfm", seed=42, conditional=True)
    assert relpath(K.CHECKPOINT, ck) == "checkpoints/artbench10/cfm_cond/seed_42/final.pt"
    assert relpath(K.SUBSET_CHECKPOINT, ck, m=5) == \
        "checkpoints/subsets/fm/artbench10/seed_42/subset_5/final.pt"
    assert relpath(K.GT_MATRIX, ck.with_(query_type="val")) == \
        "results/gt_matrix_fm_artbench10_val_seed_42.npy"


def test_gt_chain_paths():
    """§6.2-14 decoupling: cross-chain GT losses get a _chain_ segment; the
    diagonal (no chain key) stays byte-identical to the legacy names."""
    s = RunSpec(dataset="cifar2_5k", seed=42, conditional=False, query_type="gen")
    assert relpath(K.GT_LOSSES, s, eseed=0, chain=123) == \
        "results/gt_losses_fm_cifar2_5k_seed_42_chain_123_eseed_0.npy"
    assert relpath(K.GT_LOSSROW, s, eseed=2, m=7, chain=456) == \
        "results/gt_lossrows_fm_cifar2_5k_seed_42_chain_456_eseed_2/subset_7.npy"
    assert relpath(K.GT_LOSSES, s, eseed=0) == \
        "results/gt_losses_fm_cifar2_5k_seed_42_eseed_0.npy"
    # gt_matrix has no chain variant on purpose: gt_avg IS the main-table switch
    assert relpath(K.GT_MATRIX, s) == "results/gt_matrix_fm_cifar2_5k_seed_42.npy"


def test_counterfactual_paths():
    """AN-2 chain: arm + qi keyed; checkpoints under their own tree."""
    s = RunSpec(dataset="artbench2_256", seed=42, conditional=True)
    assert relpath(K.CF_MASK, s, arm="fmas_tweedie_T100", qi=7) == \
        "counterfactual/artbench2_256/fmas_tweedie_T100/seed_42/q7.npy"
    assert relpath(K.CF_META, s, arm="random") == "counterfactual/artbench2_256/random/seed_42/meta.json"
    assert relpath(K.CF_CHECKPOINT, s, arm="random", qi=0) == \
        "checkpoints/counterfactual/fm/artbench2_256/random/seed_42/q0/final.pt"
    assert relpath(K.CF_GENERATION, s, arm="fmas_T100", qi=3) == \
        "counterfactual/artbench2_256/fmas_T100/seed_42/q3/regen.pt"
    assert relpath(K.CF_ANALYSIS, s, name="summary") == \
        "counterfactual/artbench2_256/analysis/seed_42/summary.json"
    assert relpath(K.CF_FIGURE, s, name="boxplot") == \
        "counterfactual/artbench2_256/analysis/seed_42/boxplot.png"
    sc = RunSpec(dataset="cifar2_5k", seed=42, conditional=False, method="fmas_tweedie_T100")
    assert relpath(K.SCORES_META, sc) == "scores/fmas_tweedie_T100/cifar2_5k/seed_42/meta.json"
    # subset checkpoints untouched
    assert relpath(K.SUBSET_CHECKPOINT, s, m=5) == \
        "checkpoints/subsets/fm/artbench2_256/seed_42/subset_5/final.pt"


def test_ekfac_factor_paths():
    """[n] EK-FAC curvature package: query-independent, process-segmented."""
    cfm = RunSpec(dataset="cifar2_5k", seed=42, conditional=False)
    assert relpath(K.EKFAC_FACTORS, cfm) == "curvature/ekfac/cifar2_5k/seed_42/factors.pt"
    ddpm = RunSpec(dataset="cifar2", seed=123, process="ddpm")
    assert relpath(K.EKFAC_FACTORS, ddpm) == "curvature/ekfac/cifar2/ddpm/seed_123/factors.pt"
    # query_type must NOT leak into the path (one package serves both tracks)
    assert relpath(K.EKFAC_FACTORS, cfm.with_(query_type="val")) == \
        relpath(K.EKFAC_FACTORS, cfm)
    # scores ride the existing method-segment rule under the ekfac_if name
    sc = RunSpec(dataset="cifar2_5k", seed=42, conditional=False,
                 method="ekfac_if", query_type="val")
    assert relpath(K.SCORES, sc) == "scores/ekfac_if/cifar2_5k_val/seed_42/scores.npy"


def test_query_block_partial_paths():
    """EKFAC-SHARD 2026-09-09: per-query-block partials live in a blocks/
    sub-directory of the cell, so every whole-matrix name is untouched."""
    sc = RunSpec(dataset="artbench2_256", seed=42, conditional=True,
                 method="fmas_raw", query_type="val")
    assert relpath(K.SCORES_BLOCK, sc, q0=68, q1=100) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks/scores_q68_100.npz"
    assert relpath(K.REPEAT_SCORES_BLOCK, sc, q0=0, q1=34) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks/repeat_scores_q0_34.npz"
    # the cell's own artifacts are byte-identical to before the blocks existed
    assert relpath(K.SCORES, sc) == "scores/fmas_raw/artbench2_256_val/seed_42/scores.npy"
    assert relpath(K.SCORES_LAMBDA, sc, lam=1e-2) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/scores_lambda_1.00e-02.npy"
    assert relpath(K.REPEAT_SCORES, sc) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/repeat_scores.npz"
    # ddpm/proj segments ride the same rule as the rest of the scores family
    dd = RunSpec(dataset="cifar2", seed=123, process="ddpm", method="ekfac_if")
    assert relpath(K.SCORES_BLOCK, dd, q0=0, q1=50) == \
        "scores/ekfac_if/cifar2/ddpm/seed_123/blocks/scores_q0_50.npz"
    # dir + filename parse are exact inverses, and never confuse the two stems
    assert block_dir(K.SCORES_BLOCK, sc) == "scores/fmas_raw/artbench2_256_val/seed_42/blocks"
    assert parse_block_key(K.SCORES_BLOCK, "scores_q68_100.npz") == {"q0": 68, "q1": 100}
    assert parse_block_key(K.SCORES_BLOCK, "repeat_scores_q0_34.npz") is None
    assert parse_block_key(K.REPEAT_SCORES_BLOCK, "repeat_scores_q0_34.npz") == {"q0": 0, "q1": 34}
    assert parse_block_key(K.SCORES_BLOCK, "scores.npy") is None


def test_row_sharded_block_partial_paths():
    """EKFAC-ROWSHARD 2026-09-09: the train-row range is a SECOND optional level
    under the query range. Absent = the whole row range = the EKFAC-SHARD
    filename, byte-identical (the partials of a val shard already running when
    this landed must keep addressing)."""
    sc = RunSpec(dataset="artbench2_256", seed=42, conditional=True,
                 method="fmas_raw", query_type="val")
    assert relpath(K.SCORES_BLOCK, sc, q0=0, q1=25, r0=0, r1=2500) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks/scores_q0_25_r0_2500.npz"
    assert relpath(K.SCORES_BLOCK, sc, q0=0, q1=25, r0=2500, r1=5000) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks/scores_q0_25_r2500_5000.npz"
    assert relpath(K.REPEAT_SCORES_BLOCK, sc, q0=0, q1=34, r0=128, r1=256) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks/repeat_scores_q0_34_r128_256.npz"
    # GOLDEN (zero migration): no r0/r1 key -> the pre-row-shard filename
    assert relpath(K.SCORES_BLOCK, sc, q0=68, q1=100) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks/scores_q68_100.npz"
    # blocks/ is one directory for both levels
    assert block_dir(K.SCORES_BLOCK, sc, q0=0, q1=25, r0=0, r1=2500) == \
        "scores/fmas_raw/artbench2_256_val/seed_42/blocks"
    # parse is the exact inverse: the key it returns is the key that rebuilds it
    for key in ({"q0": 68, "q1": 100},
                {"q0": 0, "q1": 25, "r0": 0, "r1": 2500},
                {"q0": 0, "q1": 25, "r0": 2500, "r1": 5000}):
        name = relpath(K.SCORES_BLOCK, sc, **key).rsplit("/", 1)[1]
        assert parse_block_key(K.SCORES_BLOCK, name) == key
    assert parse_block_key(K.SCORES_BLOCK, "repeat_scores_q0_34_r0_128.npz") is None
    assert parse_block_key(K.REPEAT_SCORES_BLOCK, "repeat_scores_q0_34_r0_128.npz") == \
        {"q0": 0, "q1": 34, "r0": 0, "r1": 128}
    # a half-written row suffix is not a block file at all
    assert parse_block_key(K.SCORES_BLOCK, "scores_q0_25_r0.npz") is None


def test_e4_baseline_config_paths_do_not_collide():
    spec = RunSpec(dataset="cifar2_5k", seed=42, conditional=False,
                   method="nda", query_type="gen")
    assert relpath(K.SCORES, spec, config="single_t100") == \
        "scores/nda/cifar2_5k/seed_42/cfg_single_t100/scores.npy"
    assert relpath(K.SCORES, spec, config="single_t200") != \
        relpath(K.SCORES, spec, config="single_t100")
    weighted = spec.with_(method="dtrak_param_weighted_T100")
    assert relpath(K.PARAM_CONTRIBUTIONS, weighted, config="e4v1") == \
        "baselines/dtrak_param_weighted_T100/cifar2_5k/seed_42/cfg_e4v1/learning_contributions.pt"
    assert relpath(K.PARAM_WEIGHTS, weighted, config="e4v1").endswith("/weights.pt")
    abu = spec.with_(method="abu_plus")
    assert relpath(K.ABU_PREP, abu, config="diag_v1").endswith("/cfg_diag_v1/prepare.pt")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} addressing tests passed")
