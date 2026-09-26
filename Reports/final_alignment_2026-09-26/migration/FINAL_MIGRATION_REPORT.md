# 最终论文产物迁移结果

2026-09-26 完成默认清单的独立文件复制：**64,703 件，62,058,529,525 字节（57.80 GiB）**。复制操作未覆盖目标既有文件，CFA 没有移动、删除或写入。复制后可用空间约 94 GiB。

范围依据是当前扁平结构的 `Paper/` 实际引用图，不是旧 CFA 论文。最终稿的 8 张图、15 张表及科学定义见 [论文审计](../PAPER_SCOPE_AUDIT.md)；与旧计划的差异见 [delta.json](delta.json)。文件级对应、大小、修改时间和用途见 [files.tsv](files.tsv)。

## 已迁入的内容

| 类别 | 内容及消费关系 |
|---|---|
| SNR-LDS | `results/snr_lds_20260925/a3/inputs`、`a3/panels` 是实际拟合、选择、逐 query 预测和身份；`a4` 是已验收的汇总、删除选法、图和核验结果。不能只复制 a4。 |
| 主实验输入 | 16 面板的实际分数、原始 keep masks、实测响应、query IDs 和 score space。243 个已有单元全部覆盖，另外 5 项保留缺失状态。AB2 DAS signed-t 的 lambda=1 恢复数组随 a3 inputs 迁入。 |
| AB2 IF | 两轨完整矩阵与十档 damping 网格；生成 oracle lambda=1e-7、验证 1e-6，按各自 Full LDS 选择，不再作为部分评分任务列缺项。 |
| CFM 检索 | 当前审核 v2 的 11 份 50000×500 原生矩阵及逐 query 指标；FMAS/IF 的新 val100 网格、选参和独立 test400 最终矩阵。全部 13 方法共用同一 400 测试查询。 |
| DDPM 检索 | 当前审核 v2 的 11 份原生矩阵，以及 RunPod 根下两种曲率方法的 val100 网格、选参、test400 矩阵和指标。独立 400 列矩阵没有冒充标准 500 列地址。 |
| R16 | R0–3 和 R4–15 共 **64 份**完整评分矩阵、pilot indices、验收索引、四件 `coordinate_stats.pt` 及既有统计。支持最终 VarRatio 和 pilot-band LDS。 |
| 删除实验 | 四候选、50 queries、两预算的原始 mask、实测效用，以及共享随机删除控制、FMAS/DAS 图像效应、图像／CLIP 表征和随机视觉控制。 |
| 机制对照 | 四个核心方法的评分和 p=0…4 图源；匹配 DAS 平方对照记录；Table10 的 38 份曲率 damping 数组、D-TRAK/L1-readout 投影特征、六件原始 DDPM loss replica、原 GT 和 masks。 |
| 重建输入 | 当前完整模型与中途 checkpoint、梯度／图像特征、10 件曲率因素、AB2 latents 与原图、CIFAR cache、DDPM split 索引与标准化模型。另保留 DDPM checkpoint-family 的 step2000/4000/6000/8000。 |

核心相对地址保持不变。Figure 数值 JSON 落在 `_Data/results/paper_figures/legacy/`，原验收报告落入本轮 `source_evidence/`。R16 reader 读取的两个小型索引保留原 `_Data/reports/` 地址，以保持数据索引连接。

## 科学口径与退役范围

当前方法使用零均值 Gaussian noise scale 的 amplitude SNR，默认严格 `abs(t)/sigma > 3`。DAS 拟合和选择 signed pre-square t，聚合 t² 一次。旧 BA、soft BA、positive BA 的 fits／预测／汇总没有作为新结果迁入；仅保留生成当前 manifest 所引用的少量旧输入 manifest 作为来源证据。

CFM 最终 query 标识为 `cb0f6c7240e814c17c945abaa53e9dd0633b7bfcdb99ea5a3ab75773ea6c6cd7`；DDPM 为 `30e901c9cfbd0e21c4b2e3cc8fd455124711fbb1f7c5cc9a98be16e47c3d1963`。它们直接来自原身份记录，未计算新 hash。旧 common200 与旧 CFM v1 full400 已退出当前检索结果。

未选入退役的 signed-support 范围扫描、DDPM1000 专属扩展、额外 EK-FAC readout 评分族、p6/8 专属实验，以及 PW-DTRAK／AbU+／NDA 实验输出。当前共享输入若原本存有超过 100 列的数组，保留原轴并通过明确 IDs 选列；不为退役扩展破坏仍在用的 100-query 身份。Table10 的 `l1norm_T100` 是必要机制输入，不增加主 benchmark 方法。

最终 VarRatio 使用 `sum(sample_variances)/sum(repeat_means²)`。原重复数组、pilot 与 coordinate statistics 可复用，旧 Rep 文本不能代替新定义。作者已存的 VarRatio CSV 保留在 `Reports/figure_split_2026-09-25/`。

## 没有默认复制的大文件

| 可选内容 | 文件数 | 大小 |
|---|---:|---:|
| 子集重新训练 checkpoint | 416 | 62.57 GiB |
| 删除干预 checkpoint | 400 | 53.30 GiB |
| 外部 SD3.5 预训练缓存 | 138 | 45.51 GiB |

这些文件没有被删除或认定为退役。当前表图复核消费已迁入的原始响应、效用、mask 和图像结果；需要重新测量时，可按可选精确清单复制对应权重，或使用公开训练入口。外部预训练模型可单独取得并配置缓存。

35.6 GiB AB2 eigen-coordinate 缓存、21.5 GiB RunPod fitting shards、重复测试 lambda 数组、完整原始 DAS 归档和无关历史运行均未重复复制。最终矩阵、选参依据和必要模型／因素已选入。标准化 DDPM checkpoint 内嵌 UNet 权重及配置，不依赖旧服务器上的原始模型目录。

## 仍然缺少的本地产物

缺少 DDPM TracInCP、GAS 的 gen/val 四份最终评分与逐 query SNR 记录，以及匹配原实测响应的 Journey generation 最终结果。论文后来采用的四个作者提供均值不能代替 per-query coverage、失败计数或预测；三 checkpoint 诊断也不是四 checkpoint 正式结果，因此没有拿诊断填空。

精确缺失地址和 query mapping 见 [benchmark_dependency_check.json](benchmark_dependency_check.json)。DDPM val 使用原 IDs 0…94、96…100，不能把 q100 改名为 q95。Journey 新特征属于新生成查询，不能接到旧 GT。四件 step checkpoint 的存在也不会自动将缺格升级为完成。

## 路径、登记与核对

数据根由 `BALDS_DATA_ROOT` 或 CLI 参数配置。历史 JSON 的原绝对路径作为 provenance 原样保留，公开 reader 将支持的引用映射到当前数据根；本次没有改写历史数组或查询身份。完整运行时映射由同轮代码调整负责。

源 registry 有 4,436 条，417 条在源库已缺少本地文件。本次目标 registry 保留 **797 条**已选中、实际存在且大小符合源登记的记录。独立结果没有伪造旧登记，按精确文件清单寻址；既有 hash 字段原样保留，没有新增哈希扫描。

已完成的核对：所有复制项的源／目标大小和修改时间一致，拥有独立 inode，没有复制冲突、共享 inode 或替代数据的符号链接；303 件核心评分数组 header 可读；目标 registry 全部有对应文件；16 面板所有实际存在的 score/mask/response 已覆盖。文件内容复制通过标准独立复制完成，没有进行模型、GPU、拟合或科学实验重算。

完整回执：[COPY_RECEIPT.json](COPY_RECEIPT.json)、[REGISTRY_RECEIPT.json](REGISTRY_RECEIPT.json)、[VERIFICATION.json](VERIFICATION.json)。元数据匹配是迁移核对，不冒充论文数值的独立重算。
