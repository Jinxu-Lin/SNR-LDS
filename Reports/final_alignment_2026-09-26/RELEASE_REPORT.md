# SNR-LDS 定稿论文对齐交付（2026-09-26）

依据为 SNR-LDS 项目的本地 `Paper/`，而非历史源仓中的旧论文。本次更新 `Codes/`、`_Data/` 及对应复现报告；没有修改定稿 Paper、历史源文件、用户 `.vscode/` 或 `CodeSnapshots/`。旧代码与被替换报告保存在本目录 `before/`。

## 当前交付

- 公开包更新为 **0.2.0**，保留 `balds`、`balds-run`、`balds-repeat` 三个入口。数据根通过环境变量或命令参数指定，无原服务器运行依赖。
- 实际独立复制 **64,703 个文件，62,058,529,525 bytes（57.80 GiB）**。主模型、必要特征/曲率、测量响应、评分结果与查询身份均已纳入。不是硬链接，不移动或删除原文件。
- 另有 115.87 GiB 可再生成的子集/删除模型检查点、45.51 GiB 外部预训练缓存列为可选产物，没有重复复制。重算现有表格不需要这两类权重。
- 当前实验索引与任务报告按最终 8 个图、15 个表重写；退役报告有明确历史入口。详见 [论文范围核对](PAPER_SCOPE_AUDIT.md)、[实验索引](../EXPERIMENT_INDEX.md)、[迁移回执](migration/FINAL_MIGRATION_REPORT.md)。

## 方法与实现

1. **SNR-LDS**：唯一正式规则为 `snr_zero_mean_gaussian_v1`。RMS 标准化，零中心 101 点 KDE，密度加权拟合 `A+C*x²`，严格双侧阈值，主阈值 3；1/2/3/4 复用一次拟合。失败、有效空集与常数情形分别记录。旧 positive-only BA、混合背景估计和逐 query 选法入口移除。续跑使用输入文件地址、size/mtime 和有序轴身份，旧归档可只读验证；验证按当前响应重算各阈值的 Full/SNR LDS。
2. **DAS**：所有平台以平方前有符号 t 拟合/筛选，按原生 t² 聚合一次。ArtBench 复用已验收的 lambda=1 平方前输入。
3. **删除实验**：三 seed 内四方法共同有效 query 上求均值，再等权跨 seed 选择一个全局方法。原 LDS 选 DAS，SNR-LDS 选 FMAS，使用固定 seed42 的 50 个 query 和原生删除集合。随机对照先平均每 query 的五个模型，再平均 50 个 query。视觉表输出 CLIP cosine similarity。
4. **检索**：两平台均为 reviewed500，val100/test400。保留原 500 位置的随机流、模型身份、全训练集排名与 concept 宏平均。独立的 archived-score replay 入口复算 13 方法，曲率从验证集网格重新选参。
5. **机制**：R16 使用 `sum(sample variance)/sum(repeat mean²)`；绝对分数 head 使用 ceil 和训练索引 tie-break；power 收束到四核心方法与论文网格。Tables 8/9 单独保留论文指定的支持方向 5% 对照。
6. **基线与执行**：退役 PW-DTRAK、AbU+、NDA 实验入口移除。DDPM 引用模型与重新训练的四检查点家族分开，TracInCP/GAS 使用 steps 2000/4000/6000/8000，不把重新计算的结果冒充已发表汇总值。

## 已完成的实际数值复核

| 范围 | 验证结果 |
|---|---|
| 迁移 | 文件 size/mtime、独立 inode 核对；303 个数组 header 可读；797 条目标索引均有实际文件 |
| benchmark 输入 | 在 SNR-LDS 内离线准备全部 16 个面板，源地址归一为当前数据根 |
| SNR 归档 | 243 个可用单元、24,300 条 query；掩码/预测读取核验通过，缺失单元另记 |
| 汇总与全局选法 | 119 组均值/seed SD/coverage、476 组阈值均值与 A4 完全相同；E-DEL JSON 完全相同。辅助 query CI 已修正重采样口径，见下文 |
| 检索 | 26 个方法/模型组合、10,400 条 test query 指标；Table 2/18/19 的 572 个数值单元在论文舍入精度内一致 |
| 主表/配对控制 | Tables 1/8/9/12/13/14/15/16 和 Figure 3 两个数值面板，与 Paper 的数值单元逐项一致 |
| R16/绘图 | 最终绘图入口实际执行；精确结果与差异见 figure 验证记录及范围核对报告 |

最终集成 CPU 测试 **326 passed**（59.40 秒，22 条已有 CPU/AMP 与调度器警告）；依赖层检查 **90 个模块通过**。在 checkout 外安装实际 0.2.0 wheel、无 PyTorch 的独立环境中，**32 项数值测试通过**。176 个 Python 文件 AST、当前文档链接及私人运行路径扫描通过。见 [最终验证记录](FINAL_VALIDATION.json)、[完整测试输出](final-tests.txt)、[wheel 环境](public-wheel-environment.json)。

没有重新训练 GPU 模型或启动新的科研实验。上述工作是已存产物的 CPU 复算、数值验证及普通测试，不代表跨硬件训练逐位一致。

## 明确保留的证据边界

- **四个 aggregate-only 结果**：DDPM TracInCP/GAS 的 val/gen 均值由作者于 2026-09-26 提供，只在最终 Table1 出现。`Codes/configs/paper_aggregate_supplement.json` 记录这四个舍入值的来源；不编造逐 query 分数、拟合或 coverage。加上 Journey-TRAK generation，目前共有 5 个未具备本地完整评分输入的 DDPM 单元。Journey-TRAK validation 为 n/a。
- **旧文字与最终表格**：附录 XY05 的 CFM fine/coarse 数值段仍使用旧结果，而 Table18 和最终 reviewed500 产物一致。代码按最终验收数据复算，未改 Paper。具体位置与数值见论文范围核对。
- **绝对 head 的并列规则**：旧 Fig2/Fig5 数据采用 `argpartition`；定稿规则指定训练索引顺序。相同数组已证实旧算法可精确复现旧点值。新代码遵循定稿稳定 tie-break，FMAS/IF 的图点有小幅变化（88 个 head/readout 单元最大差 0.07717 LDS×100）；历史 JSON 保留作溯源，没有用它覆盖新计算。
- **Table10 浮点差异**：CPU 从已存投影特征重算 lambda=10 的 D-TRAK/L1 行与纸面数值有约 0.02 LDS×100 以内差异；没有可直接读取的该固定 lambda 归档矩阵。曲率行及其区间一致，投影差异的精确数值原因未作额外硬件实验确认。具体 CPU 输出与差异见 [图表回执](FIGURE_REPRODUCTION.md)，未静默修成纸面值。

## 汇总代码修正

独立检查发现旧辅助 query CI 在不同模型之间取有效 query 交集，并对 generation 共用重采样索引。这会偏离各 seed 有效 query 均值的目标。本版改为 generation 在各模型内独立重采样，validation 共用 query ID 抽样并保留各 seed 有效集合；不改变论文所用的均值、seed SD、coverage 或全局选择。部分 query 缺失时，汇总从 panel 状态取完整请求数及缺失数。普通回归测试覆盖这些具体失败情形。原始 A4 文件保留不改。

## 使用入口

从 `Codes/` 安装后，先设置 `BALDS_DATA_ROOT`，按 [Codes README](../../Codes/README.md) 运行。通常无需重新训练：

```bash
python tools/reproduce_benchmarks.py prepare
python tools/reproduce_benchmarks.py postprocess \
  --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels"
python tools/reproduce_benchmarks.py verify \
  --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels"
python tools/replay_retrieval.py --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/paper/retrieval" --paper ../Paper
```

所有新图表默认写至 `_Data/results/paper/figures/`，新汇总写至 `results/paper/`。最终 Paper 不作为 renderer 输出目录。实验报告提供从训练到度量的完整执行线；历史源路径只承担溯源作用。
