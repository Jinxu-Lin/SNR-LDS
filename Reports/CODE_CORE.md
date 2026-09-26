# 核心代码整理与验证

> 当前项目：**SNR-LDS**（安装包 `snr-lds`）。下文保留早期整理回执，旧名称和早期评价器仅描述当时状态；当前接口与方法范围见 [Codes/README.md](../Codes/README.md)。

本次只改写 BA-LDS 目标目录。CFA 及其工作树保持只读；没有运行 GPU 训练、论文实验、远程命令，也没有迁移 `_Data` 实验产物。

## 收束结果

代码发布为标准 `src/balds` Python 包，提供三个入口。整合 source-scoring 与最后显式生成分批修复之前的包代码约 17,149 行，源主包约 21,546 行；该比较不含源工作树额外工具，不能据此声称覆盖整个历史代码库的压缩比例。

| 入口 | 用途 | 依赖 |
|---|---|---|
| `balds` | 已有分数/子集响应的 Full LDS、BA-LDS、面板与删除效用分析 | 基础 NumPy/SciPy/PyYAML；读取 Torch 文件时才需要 Torch |
| `balds-run` | 训练、query、梯度/曲率特征、子集响应、基线、检索与删除重训 | `train`；SD3.5 额外使用 `latent` |
| `balds-repeat` | 固定 query ID 的 R16 独立 MC 重复与统计 | `train`；沿用原始流与参数 |

`schema → data/models/attribution/evaluation → workflows → cli` 按职责拆分；`artifacts` 负责本地相对地址、原子写入与清单。原 pipeline 实际拆分为 `common.py`、`training.py`、`queries.py`、`features.py`、`subsets.py`，并非增加一层兼容转发。曲率流式计算与行/query 分片保留在独立曲率模块，避免重新实现数值算法。

保留平台：`cifar2_5k`、`cifar10_v2`、`artbench2_256`、导入 DAS 的 `cifar2_das`、污染检索的 `cifar10_inj4`。保留当前论文方法与机制分析需要的数值变换。参数加权 D-TRAK、AbU+、NDA 的代码作为历史实现保留；2026-09-25 作者已将三者退出实验基线，仅在 related work 中介绍，不再计入待补实验。

移除前代 prediction/sign/positive-square 编排、regional 任务、SSH/sync/prune、SD1.5、旧低分辨率 ArtBench、潜空间外来风格注入路径。共享 canonical JSON、哈希、run token 和 MC 身份原语从旧实验契约中提取为 `schema/identity.py`。历史 RNG/hash namespace 原样保留，避免仅因改名改变随机流和产物身份。

## 科学语义与接口

- `fmas` 统一解析为 `fmas_raw`，即当前论文的有符号双线性分数。
- DAS 的背景拟合输入是有符号平方前 `t`；原生聚合使用 `t²`。BA 数值层由独立评估模块实现，不沿用旧推送密度或以 `t²` 拟合背景的逻辑。
- 删除实验 `--arm das_native_sq` 读取 `das_T100` 的 `t`，平方一次后做原生 top-k。固定 `--fmas-rho` 直接读指定 rho 的 raw FMAS 分数，不要求 repeat score、不做 shrinkage。
- build/analyze 接受 `--arm fmas,dtrak_T100,das_native_sq,ekfac_if,random`，一次输出完整四方法加随机参考；retrain/regenerate 每次一个 arm。默认旧两方法加随机仍保留，因此论文执行线显式给出完整列表。
- 相对科学产物路径保留，可与迁移清单一一对应。根路径使用 `--data-root`/`BALDS_DATA_ROOT`，本地 manifest/cache/raw 路径随根目录推导。
- 独立权重学习 query 的普通像素 CFM `generate --batch 50` 现已按来源脚本执行：完整初始 noise 只抽一次，再按顺序切块，保存 `generation_batch_size`。未传 `--batch` 时仍走原来的单次全批路径。
- `featurize.projection` 可显式选择投影后端；默认仍为 `auto`。`torch_chunked` 与历史 CUDA-JL 的投影基不同，必须成套重新计算 train/query 特征，不能混用。
- CLIP 尊重调用者的缓存和 offline 环境设置，不再修改全局 Hugging Face offline 状态。`datasets` 限制为 `>=4.7,<5`，避免当前 CIFAR ID/cache 接口在 5.x 中失效。

## 选择性集成来源

模块来源明细位于 `Codes/configs/core_provenance.json`。其中 `a68da05` 表示源主工作区所处版本与本次读取的工作区快照；源工作区已有未提交状态，未执行 checkout/reset 或覆盖。

| 来源 | 本次采用的内容 |
|---|---|
| 主工作区 `a68da05` 快照 | 数据、模型、产物寻址、pipeline、删除与检索主链等基础闭包 |
| `36788a7a785b16b27fa1ad10dde0b2aa8d8bb43a` | 参数加权/AbU+/NDA、投影/梯度基线、配置身份、相关寻址与曲率修复 |
| `f686bd856ae9e9a99178ba7f34414941aa49db3b` | 曲率分片、joint 计算、R16 数值/统计/IO、kernel；MC generator 的 uint64 取模修复也显式纳入 |
| 公开版适配 | 包名/目录重组、pipeline 拆分、本地配置、共享身份原语、R16 CLI、FMAS raw 与 DAS 删除原生分数语义 |

检索 evaluate 的 replacement 支持按当前源码接口补入，并由集成测试覆盖。根任务另行负责 BA、机制控制、检索分数装配、图表与面板准备工具；这些模块的来源和验证不混同于上述 core 测试。

R16 的 `e3c.py` 保留历史数值执行器；`repeatability.py` 增加公开可用的统计适配。固定 5,000 训练行、16+16 原 query 位置、16 次独立 MC、FMAS rho 0.01、D-TRAK lambda 0.05 与原始 RNG 流均来自当前 R16 配置。原主机分工限制移除，用户可运行 0–15 任意 repeat。独立 pilot 通过显式相对路径 manifest 提供；缺失 repeat 明确报告。

## 实际验证

测试通过隔离 `PYTHONPATH=Codes/src` 指向目标包，在现有 Python 3.12 环境执行；`CUDA_VISIBLE_DEVICES=''`，Hugging Face offline，CPU 线程数 2。没有安装或修改源环境。

| 检查 | 结果 |
|---|---|
| 保留 core 完整 CPU 单元/集成集 | **290 passed, 1 skipped**，55.50 秒；`core-test-final.txt` |
| 最后 CLI/投影配置变更后的定向检查 | **13 passed**，3.29 秒；`core-portability-test.txt` |

完整集覆盖核计算、曲率/阻尼、投影、MC 身份、GT/子集、原子存储、相对路径、行/query 分片、CLI 与小型模型替身。定向检查额外覆盖：阻止 Torch 导入时 CPU BA 可导入；显式 data root 优先于环境变量；FMAS 固定 rho 无 repeats 依赖；DAS 平方前负值按平方后原生排名；CLIP 不强制 offline；完整删除 arm 列表传递；显式投影后端传入 Journey 特征。

观察到的 core 测试版本：Torch **2.6.0+cu124**、torchvision **0.21.0+cu124**、NumPy 2.3.5、SciPy 1.17.1、datasets 4.7.0、transformers 5.3.0。约束参考为 `Codes/configs/constraints-cpu-tested.txt`。测试运行时禁用了 CUDA，不能将版本后缀解读为已经验证 GPU 执行。警告主要来自 CPU 上关闭 CUDA autocast 和调度器接口弃用。一个真实数据检查因目标目录没有 CIFAR-100 本地缓存而跳过；其余测试使用小型确定性替身。

## 尚未由本次验证证明的事项

本次没有重新训练论文模型、跑真实大规模梯度/曲率、下载上游数据/权重或完成在途实验。没有验证所有 GPU/驱动/混合精度组合。ArtBench 和导入 DAS 的外部原始输入仍需按迁移报告准备；CLIP/预训练模型首次使用需缓存或上游访问条件。完整公开安装与 root-owned BA/图表/装配测试由总验证报告另记。

最后发现并补齐的执行线一致性：普通 CFM 的生成分批、完整初始 RNG 抽样、条件标签行序和默认全批行为经独立替身测试覆盖；定向结果见 `core-generation-test.txt`。未执行真实模型生成。

## 2026-09-26 最终论文对齐补记

此补记更新当前发布状态，保留上文9月22日及后续历史证据。最终 Paper 位于 BA-LDS/Paper，未覆盖或修改；对齐前 Codes 完整保存在 `Reports/final_alignment_2026-09-26/before/Codes`。

PW-DTRAK、AbU+、NDA 已从当前公开核心及专属命令、产物类型和测试中退役。普通 D-TRAK、l1norm_T100 和最终机制表需要的数值辅助功能保留。R16 改为论文最终 VarRatio：样本方差之和除以未中心化均值平方之和；主图十段按每个 query 的绝对均值升序划分，附录五段保留独立 pilot。旧 repeated-SNR calibration/cross-reference 编排已退役。

检索现支持 CFM/DDPM reviewed-v2 100val+400test，并以原 query 位置固定 MC 随机流。增加可移植 DDPM reference 与4checkpoint执行入口，以及独立的历史检索矩阵 CPU 回放工具。283项 owned CPU测试通过，最终19项定向测试通过；实际回放26方法单元共10400个测试 query，Tables2/18/19全部572个数值格与当前 Paper 的显示精度一致。发现并记录正文细/粗粒度 CFM 汇总仍有旧数值，未修改 Paper。

具体保留/退役范围、接受代码版本、命令、验证边界与日志见 `Reports/final_alignment_2026-09-26/CORE.md`；论文正文差异见同目录 `RETRIEVAL_REPLAY.md`。本次没有GPU训练或科学实验重跑。最终整体套件和安装验证由发布总报告另记。
