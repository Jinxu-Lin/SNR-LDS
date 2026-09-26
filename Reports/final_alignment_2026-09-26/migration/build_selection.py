"""Build the final-paper selection from accepted manifests, without transferring data."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(source):
    old = json.loads((HERE.parents[1] / 'migration/selection.json').read_text())
    prior = {f['id']: f for f in old['families']}
    families = []
    common = dict(status='selected', profile='analysis', role='input',
                  source_addressing='Codes/cfa/store/addressing.py',
                  destination_addressing='Codes/src/balds/artifacts/addressing.py',
                  paper=['Paper/Sections/05_Experiment.tex', 'Paper/Sections/XY03_MechanismExperiment.tex',
                         'Paper/Sections/XY05_EvaluationExperiments.tex'])

    def add(name, include, reason, exclude=(), **kwargs):
        families.append({**common, 'id': name, 'include': sorted(set(include)),
                         'exclude': list(exclude), 'reason': reason, **kwargs})

    def reuse(name, **updates):
        f = deepcopy(prior[name])
        f.update(paper=common['paper'], **updates)
        families.append(f)
        return f

    manifest = json.loads((source / '_Data/results/snr_lds_20260925/a3/inputs/benchmark_manifest.json').read_text())
    score_paths, response_paths = set(), set()
    for panel in manifest:
        response_paths.update('_Data/' + panel[k] for k in ['masks', 'response'])
        for method in panel['methods']:
            if method.get('scores'):
                path = '_Data/' + method['scores']
                score_paths.update([path, str(Path(path).parent / 'meta.json')])
    add('final_snr_benchmark_score_inputs', score_paths,
        'Exact 16-panel accepted A3 score inputs; original IDs and score spaces are retained. Missing methods remain explicit gaps.')
    add('final_snr_benchmark_masks_responses', response_paths,
        'Exact accepted keep masks and measured responses used by Full/SNR; preserve all stored columns and select by explicit IDs.')
    add('final_snr_outputs', [
        '_Data/results/snr_lds_20260925/a3/inputs/*',
        '_Data/results/snr_lds_20260925/a3/panels/*',
        '_Data/results/snr_lds_20260925/a4/tables/*',
        '_Data/results/snr_lds_20260925/a4/edel/*',
        '_Data/results/snr_lds_20260925/a4/figures/*',
        '_Data/results/snr_lds_20260925/a4/verify_a.json'],
        'Accepted final SNR fitting/selection/predictions and A4 summaries, effects and figures; excludes superseded A3 summaries and diagnostics.', role='result')
    add('legacy_manifest_input_provenance', [
        '_Data/results/ba_c2_c10_das_presquare_20260922/inputs/bench_manifest.json',
        '_Data/results/c10_cpu_closeout_20260923/inputs/bench_manifest.json',
        '_Data/results/ddpm_noncore_val_repair_20260924/cpu/ba_manifest.json',
        '_Data/results/ddpm_noncore_val_repair_20260924/shared_gpu/ba_manifest.json',
        '_Data/results/ab2_if_closeout_20260922/inputs/ab2_if_manifest.json'],
        'Input/identity evidence used to construct the accepted SNR manifest; no historical BA predictions or fits are selected.', role='metadata')
    add('ab2_if_oracle_lambda_grid', [
        '_Data/scores/ekfac_if/artbench2_256/seed_42/scores_lambda_*.npy',
        '_Data/scores/ekfac_if/artbench2_256_val/seed_42/scores_lambda_*.npy'],
        'Ten native IF damping candidates supporting final trackwise Full-LDS oracle lambda=1e-7/1e-6 selection.')
    add('core_ddpm_matched_query_damping_inputs', [
        f'_Data/scores/{method}/cifar2_das{track}/ddpm/seed_42/scores_lambda_*.npy'
        for method in ['fmas_raw', 'ekfac_if'] for track in ['', '_val']],
        'Existing grids for the retained matched-100 DDPM squaring comparison; the 1000-query extension is retired.')
    add('ddpm_matched100_projected_and_loss_inputs', [
        '_Data/featurize/l1norm_T100/cifar2_das/*',
        '_Data/featurize/l1norm_T100/cifar2_das_val/*',
        *[f'_Data/results/gt_losses_ddpm_cifar2_das{track}_seed_42_eseed_{seed}.npy'
          for track in ['', '_val'] for seed in [0, 1, 2]]],
        'Retained Table10 recomputes projected damping from original DTRAK/L1-readout features and verifies GT from the three measured-loss replicas. L1 readout is a mechanism input, not an extra main-benchmark method.')

    methods = ['pixel_dot', 'pixel_cos', 'clip_dot', 'clip_cos', 'grad_dot_T100', 'grad_cos_T100',
               'trak_T100', 'dtrak_T100', 'das_T100', 'relative_if_T100', 'renorm_if_T100']
    add('reviewed500_standard_retrieval', [
        f'_Data/scores/{method}/cifar10_inj4_inject/{segment}/seed_42/{filename}'.replace('//', '/')
        for method in methods for segment in ['', 'ddpm']
        for filename in ['scores.npy', 'meta.json', 'inject_result.json']],
        'Final 11 methods on reviewed500 CFM/DDPM queries; native 500-column score arrays, exact query identity and 400-test metrics.')
    add('cfm_reviewed500_curvature_final', [
        '_Data/results/cfm_reviewed500_20260923/verified_summary.json',
        '_Data/results/cfm_test400_20260923/analysis/*',
        '_Data/results/cfm_test400_20260923/lane*/fmas_raw/identity/identity.json',
        '_Data/results/cfm_test400_20260923/lane*/ekfac_if/identity/identity.json',
        '_Data/results/cfm_test400_20260923/lane*/fmas_raw/result.json',
        '_Data/results/cfm_test400_20260923/lane*/ekfac_if/result.json',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/fmas_raw/identity/identity.json',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/ekfac_if/identity/identity.json',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/fmas_raw/result.json',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/ekfac_if/result.json',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/fmas_raw/val_scores.npy',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/ekfac_if/val_scores.npy',
        '_Data/results/cfm_val100_20260923/incoming/xc3-*/scores/*/cifar10_inj4_inject/seed_42/scores_lambda_*.npy'],
        'Final CFM v2 cb0f6c...: val100 selects FMAS1 and IF1e-9; held-out400 final arrays/metrics. Avoid duplicate test scoring blocks.', role='result')
    add('ddpm_reviewed500_curvature_final', [
        '_Data/results/runpod_ddpm_source_20260923/identity/identity.json',
        '_Data/results/runpod_ddpm_source_20260923/plan.json',
        '_Data/results/runpod_ddpm_source_20260923/complete.json',
        '_Data/results/runpod_ddpm_source_20260923/verify.remote.json',
        '_Data/results/runpod_ddpm_source_20260923/score/*/val.json',
        '_Data/results/runpod_ddpm_source_20260923/score/*/test.json',
        '_Data/results/runpod_ddpm_source_20260923/score/val/scores/*/cifar10_inj4_inject/ddpm/seed_42/scores*.npy',
        '_Data/results/runpod_ddpm_source_20260923/score/val/scores/*/cifar10_inj4_inject/ddpm/seed_42/meta.json',
        '_Data/results/runpod_ddpm_source_20260923/score/test/scores/*/cifar10_inj4_inject/ddpm/seed_42/scores.npy',
        '_Data/results/runpod_ddpm_source_20260923/score/test/scores/*/cifar10_inj4_inject/ddpm/seed_42/meta.json'],
        'Reviewed DDPM v2 30e901... validation grids and selected test400 scores; skip duplicate fit shards and test lambda grids.', role='result')

    reuse('queries')
    add('final_source_identity_review', [
        '_Data/subsets/cifar10_inj4_inject_meta.json',
        '_Data/results/inject/cifar10_inj4/detector_all_r50_final.pt',
        *[f'_Data/results/inject/cifar10_inj4/{name}/{leaf}'
          for name in ['c10mine_cfm_v2_png', 'c10mine_v2_png']
          for leaf in ['*.csv', '*.json', 'candidates/*']]],
        'Current reviewed500 CFM and DDPM query provenance, official selected images and injection identity; excludes old v1 selection/backup pools.')
    reuse('models_current')
    add('ddpm_checkpoint_family_steps', [
        '_Data/results/ddpm_gaps_20260923/retrain_j2/step_2000.pt',
        '_Data/results/ddpm_gaps_20260923/retrain_j2/step_4000.pt',
        '_Data/results/ddpm_gaps_20260923/retrain_j2/step_6000.pt',
        '_Data/results/ddpm_gaps_20260923/retrain_j2/step_8000.pt',
        '_Data/results/ddpm_gaps_20260923/retrain_j2/recipe.json',
        '_Data/results/ddpm_gaps_20260923/retrain_j2/result.json'],
        'Four accepted checkpoint-family producer inputs for DDPM TracInCP/GAS. Existing steps do not establish missing final score/SNR results.',
        profile='regeneration')
    reuse('gradient_features')
    reuse('curvature')
    reuse('ab2_latents')
    reuse('ab2_raw_images')
    reuse('cifar_dataset_cache')
    reuse('das_archive_identity')
    reuse('pretrained_sd35', profile='optional_pretrained', status='selected',
          reason='External pretrained SD3.5 weights; downloading/configuring an existing local cache is supported. Separate optional 45.5GiB copy, not a default duplicate.')
    reuse('subset_models', profile='optional_retraining',
          reason='Regenerable subset checkpoint banks. Accepted responses are selected by default; copying these 62.6GiB banks is optional for loss remeasurement.')
    reuse('deletion_models', profile='optional_retraining',
          reason='Regenerable intervention checkpoint bank. All accepted masks, effects and visual outputs remain selected; these 53.3GiB weights are optional.')

    reuse('deletion_measurements', exclude=[], include=prior['deletion_measurements']['include'] + [
        '_Data/counterfactual/cifar2_5k/random_k300/*',
        '_Data/counterfactual/cifar2_5k/random_k1000/*',
        '_Data/counterfactual/cifar2_5k/reference/*'])
    reuse('edel_effects')
    add('deletion_image_effects', [
        '_Data/results/fmas_das_visual_20260922/*',
        '_Data/results/fmas_das_random_visual_20260922/*'],
        'Final Figure3(b) all50-query paired image effects and shared random-deletion controls.',
        exclude=['*/supervised/*', '*.log', '*.lock'], role='result')
    reuse('r16_legacy_scores')
    reuse('r16_new_scores')
    reuse('r16_final_analysis')
    reuse('r16_pilot_manifest')
    add('r16_runtime_package_indices', [
        '_Data/results/narrative_wave_20260921/packages/pilot_manifest.json',
        '_Data/reports/narrative_gpu_wave_2026-09-21/VERIFY_M2A_R16_V1.json'],
        'Read by current VarRatio/independent-pilot analysis; retain index addresses and provenance IDs.')
    add('matched_das_squaring_controls', [
        '_Data/results/lds_nextwave_20260920/j2/a1/transforms/*'],
        'Retained Tables8/9 matched-DAS readouts and paired-query bootstrap evidence; no signed-support range sweep output.', role='result')
    reuse('paper_figure_tables', include=[
        f'Codes/Figure/out/json/{name}.json' for name in [
            'fig_score_magnitude', 'fig2_head_profile', 'fig1_power_c2', 'tab_3_1_square',
            'tab_4_2_square_gain', 'tab_4_2_crossed', 'tab_C3_ddpm']],
        reason='Existing numerical sources for retained figures/tables; final renderers select current core methods/powers0..4 and DDPMmatched100.')
    add('final_acceptance_reports', [
        '_Data/reports/experiment_closeout_2026-09-24/ACCEPTANCE.md',
        '_Data/reports/experiment_closeout_2026-09-24/VERIFY.json',
        '_Data/reports/cfm_reviewed500_2026-09-23/MIGRATION.md',
        '_Data/reports/cfm_test400_2026-09-23/DELIVERY.md',
        '_Data/reports/ab2_if_joint_2026-09-21/ACCEPTANCE*.md',
        '_Data/reports/narrative_gpu_wave_2026-09-21/ACCEPTANCE_M2A_R16_V1.md'],
        'Immutable source acceptance/protocol evidence, stored in Reports separately from computational artifacts.',
        role='provenance', source_prefix='_Data/reports/',
        destination_prefix='Reports/final_alignment_2026-09-26/source_evidence/')
    pending = [
        {'id': 'ddpm_tracin_gas_final', 'status': 'unresolved_source',
         'reason': 'Paper Sep26 reports four author-supplied means; accepted A3 has missing inputs and no final per-query SNR records found in a1. Do not replace with three-checkpoint diagnostics.'},
        {'id': 'ddpm_journey_generation', 'status': 'missing_input',
         'reason': 'New trajectory/query features are not aligned to old GT; final compatible benchmark score/response remains absent.'},
    ]
    excluded = [
        'Old BA fits/predictions and positive/soft BA pilots: superseded by final zero-mean amplitude-SNR evaluator.',
        'CFM common200 and old CFMv1 full400 retrieval: final reviewed500 query source is different.',
        'Retired signed-support range sweeps, Table3 standaloneband, Table11 DDPM1000, extra-readout/p6/8-only score families.',
        'PW-DTRAK, AbU+, NDA experiment outputs: not in final paper benchmark.',
        'AB2 IF eigencoordinate cache35.6GiB and scoring shards: final matrices+lambda grids selected.',
        'RunPod curvature fitting shards21.5GiB and duplicate test-lambda matrices: canonical factors/final matrices selected.',
        'Full raw DAS archive and unrelated ArtBench10/inj8 data, oldruns, logs, lockfiles and raw backup query candidates.',
    ]
    return {'schema_version': 2, 'scientific_status': {
        'paper': 'Final flat Paper include graph dated2026-09-26 is authoritative.',
        'evaluator': 'SNR-LDS zero-mean Gaussian; amplitude abs(t)/sigma; strict zeta>3 default, thresholds1/2/3/4.',
        'DAS': 'Fit/select signed pre-square t; aggregate native t^2 exactly once.',
        'retrieval': 'Reviewed500 CFM+DDPM,100val/400test,13methods per process.',
        'source_is_read_only': True, 'transfer_performed': False},
        'families': families, 'pending': pending, 'excluded': excluded, 'runtime_json_inputs': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'selection.json').write_text(json.dumps(build(args.source), ensure_ascii=False, indent=2) + '\n')
