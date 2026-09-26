# SNR-LDS experiment dossiers

本目录按当前论文的实验链组织，而不是按代码备份日期或计算机器组织。
实现统一使用 [Codes](../Codes/README.md)，不再保存多份完整源码副本。
每份报告包含实验目的、输入身份、固定参数、执行入口、输出和已知缺项。
命令均从仓库根目录执行；模型、特征、分数及生成图像位于外部数据根，不进入 Git。

| 实验环节 | 档案 | 当前论文对应 |
|---|---|---|
| 主模型训练与查询 | [训练索引](01_training/README.md)、[CIFAR-2 CFM 三 seed](01_training/cifar2_cfm/README.md) | 所有后续实验的模型输入 |
| 子集重训、响应、梯度特征与曲率评分 | [共用生产流程](00_pipeline/README.md) | LDS / SNR-LDS 的共同上游 |
| 分数分布 | [幅值分布](02_magnitude/README.md) | Fig. 1(a), Fig. 4 |
| 有限采样扰动 | [R16 重复测量](03_sampling_noise/README.md) | Fig. 1(b), Table 3 |
| 聚合范围、幂变换与平方 | [重加权机制](04_reweighting/README.md) | Fig. 2, Figs. 5–6, Tables 4–7 |
| 方法比较与阈值敏感性 | [SNR-LDS](05_snr_lds/README.md) | Fig. 3(a), Figs. 7–8, Tables 1, 10–12 |
| 删除并重训 | [方法选择与实际干预](06_deletion/README.md) | Fig. 3(b), Tables 8–9 |
| 受控来源检索 | [注入、审核、评分与检索](07_source_retrieval/README.md) | Tables 2, 13–15 |

## 阅读与执行顺序

只复算已接受结果：准备外部产物，按各报告的数组重放或后处理命令运行。
从头训练：先安装 [依赖](../Codes/README.md)，运行训练，再按共用生产流程生成查询、
子集响应和归因分数，最后执行对应实验分析。来源检索还需要人工审核的查询包。
导入的 DDPM 参考平台不是本仓库自训练平台，必须保留其存档身份。

训练示例默认只打印命令，不启动 GPU；显式 `--execute` 才执行。
完整源代码与配置以当前 Git 版本为准；新运行应记录 Git 版本、命令、环境和输入身份，
不能把当前实现的命令重建称为过去执行时的原始日志。

## 证据与限制

- `evidence/` 保存与该实验直接相关的小型数值核验记录；原始大数组仍在外部数据根。
- 三 seed 训练的 [产物清单](01_training/cifar2_cfm/evidence/artifacts.json) 来自实际本地产物，含 SHA-256 和检查点元数据，不是新训练结果。
- [验证说明](validation/README.md) 区分既有数值重放、当前代码测试和本次目录整理。
- DDPM TracInCP/GAS 四个 SNR 表格数值为作者提供的汇总，没有对应逐 query 覆盖记录；Journey-TRAK DDPM generation 仍缺输入。详见 SNR-LDS 报告。
- 旧 head 边界 tie 规则与当前稳定排序存在小幅数值差异，DDPM float32 重建也有舍入差异，详见重加权报告。

旧 `CodeSnapshots/` 和 `Reports/` 已整体移至本地忽略目录，保留恢复能力，不再作为公开实验入口。
退役 BA 评价、common200、旧查询版本、机器调度和论文排版记录不纳入当前档案。
历史数据目录标识和历史核验 JSON 内的旧图表编号不改写；本索引及各报告使用当前编号。
