# 迁移盘点数字

盘点 UTC：`2026-09-22T08:07:01.234190+00:00`。这是 live source 的文件元数据快照；最终迁移前重新生成 inventory。

## 选择层

| Profile | 文件数 | 精确字节数 | GiB |
|---|---:|---:|---:|
| `analysis` | 3,300 | 7,907,849,860 | 7.365 |
| `pending` | 630 | 38,934,264,705 | 36.260 |
| `regeneration` | 13,108 | 172,102,426,600 | 160.283 |
| `external` | 138 | 48,861,594,940 | 45.506 |
| `provenance` | 81 | 213,989,096 | 0.199 |
| `optional_fit_cache` | 1,200 | 1,912,770,016 | 1.781 |

默认两层合计 **16,408 文件、180,010,276,460 字节（167.648 GiB）**。pending/external 是观察到的候选资产，不进入复制集。provenance 中的旧 V3 摘要不是当前 DAS 协议结果。

新增 DAS 最小身份闭包共 4 文件、19,218 字节（两份有序 split 索引、模型配置、scheduler 配置）；不复制完整 archive。

## 家族（规则可重叠，上表按文件去重）

| 家族 | 状态 | 文件数 | 字节数 | GiB |
|---|---|---:|---:|---:|
| `ab2_if_pending` | `pending` | 617 | 38,240,985,634 | 35.615 |
| `native_score_matrices` | `selected` | 579 | 4,158,493,854 | 3.873 |
| `source_retrieval_scores` | `selected` | 66 | 2,203,313,731 | 2.052 |
| `subset_masks` | `selected` | 4 | 4,328,335 | 0.004 |
| `responses` | `selected` | 100 | 12,710,400 | 0.012 |
| `queries` | `selected` | 18 | 197,453,134 | 0.184 |
| `models_current` | `selected` | 46 | 6,262,248,218 | 5.832 |
| `subset_models` | `selected` | 416 | 67,181,131,008 | 62.567 |
| `deletion_models` | `selected` | 400 | 57,231,073,137 | 53.301 |
| `deletion_measurements` | `selected` | 808 | 8,899,407 | 0.008 |
| `gradient_features` | `selected` | 224 | 21,639,578,753 | 20.153 |
| `curvature` | `selected` | 9 | 18,645,448,082 | 17.365 |
| `ab2_latents` | `selected` | 4 | 447,958,968 | 0.417 |
| `ab2_raw_images` | `selected` | 12,000 | 422,469,248 | 0.393 |
| `cifar_dataset_cache` | `selected` | 9 | 272,519,186 | 0.254 |
| `pretrained_sd35` | `external_dependency` | 138 | 48,861,594,940 | 45.506 |
| `r16_legacy_scores` | `selected` | 49 | 92,267,297 | 0.086 |
| `r16_new_scores` | `selected` | 141 | 18,549,997 | 0.017 |
| `r16_final_analysis` | `selected` | 10 | 15,040,574 | 0.014 |
| `r16_pilot_manifest` | `selected` | 1 | 100,704 | 0.000 |
| `edel_effects` | `selected` | 2 | 159,752 | 0.000 |
| `aggregation_supplements` | `selected` | 68 | 78,373,535 | 0.073 |
| `ba_pilot_appendix` | `historical_appendix` | 94 | 5,281,700 | 0.005 |
| `ba_v3_provenance` | `provenance_only` | 16 | 113,032,273 | 0.105 |
| `ba_das_reusable_fits` | `reusable_input` | 1,200 | 1,912,770,016 | 1.781 |
| `retrieval_final_cfm200` | `selected` | 9 | 83,947,528 | 0.078 |
| `retrieval_score_blocks` | `selected` | 303 | 903,083,085 | 0.841 |
| `retrieval_appendix_results` | `selected` | 15 | 3,940,715 | 0.004 |
| `injection_identity` | `selected` | 1,008 | 97,693,768 | 0.091 |
| `paper_figure_tables` | `selected` | 9 | 191,590 | 0.000 |
| `weighted_dtrak_pending` | `pending` | 13 | 693,279,071 | 0.646 |
| `selected_acceptance_provenance` | `provenance_only` | 64 | 636,010 | 0.001 |
| `das_archive_identity` | `selected` | 4 | 19,218 | 0.000 |
| `unselected_raw_das_archive` | `excluded` | 12,726 | 79,631,738,851 | 74.163 |
| `retired_prediction_layer` | `excluded` | 764 | 23,572,447,356 | 21.954 |
| `retired_soft_ba` | `excluded` | 18,813 | 4,308,247,924 | 4.012 |
| `superseded_native_ba` | `excluded` | 24,421 | 37,000,673,053 | 34.460 |
| `superseded_cpu_ba` | `excluded` | 26,821 | 8,757,111,167 | 8.156 |
| `retired_m2b` | `excluded` | 1,061 | 381,309,730 | 0.355 |
| `unused_artbench_foreign` | `excluded` | 6,413 | 18,631,780,768 | 17.352 |
| `m3_readout_diagnostic_scores` | `selected` | 12 | 24,001,536 | 0.022 |
| `legacy_platform_panel_identities` | `provenance_only` | 1 | 100,320,813 | 0.093 |

## 来源树全量（不是搬运选择）

| 来源类别 | 文件数 | 字节数 | GiB |
|---|---:|---:|---:|
| `Codes/Figure` | 18 | 303,671 | 0.000 |
| `_Data/_ledger` | 9 | 295,208 | 0.000 |
| `_Data/archive` | 56 | 243,276,300 | 0.227 |
| `_Data/baselines` | 7 | 680,973,195 | 0.634 |
| `_Data/checkpoints` | 1,163 | 185,595,822,089 | 172.850 |
| `_Data/counterfactual` | 854 | 18,899,163 | 0.018 |
| `_Data/curvature` | 9 | 18,645,448,082 | 17.365 |
| `_Data/featurize` | 250 | 61,428,061,265 | 57.209 |
| `_Data/generations` | 28 | 598,471,536 | 0.557 |
| `_Data/hf_cache` | 14 | 272,524,383 | 0.254 |
| `_Data/latents` | 12 | 780,707,492 | 0.727 |
| `_Data/logs` | 318 | 31,204,815 | 0.029 |
| `_Data/manifest.db` | 1 | 2,310,144 | 0.002 |
| `_Data/manifest.db.bak_20260809` | 1 | 184,320 | 0.000 |
| `_Data/models` | 138 | 48,861,594,940 | 45.506 |
| `_Data/progress` | 221 | 3,115,065 | 0.003 |
| `_Data/raw` | 79,151 | 102,198,357,941 | 95.180 |
| `_Data/reports` | 914 | 188,753,590 | 0.176 |
| `_Data/results` | 116,087 | 127,410,933,044 | 118.661 |
| `_Data/scores` | 1,080 | 7,629,355,383 | 7.105 |
| `_Data/scripts` | 140 | 470,014 | 0.000 |
| `_Data/subsets` | 7 | 4,617,386 | 0.004 |

源 registry 2,643 行，417 个引用路径本地缺失；准确名单在 inventory.registry.missing_local。未查远程、未计算 hash。
