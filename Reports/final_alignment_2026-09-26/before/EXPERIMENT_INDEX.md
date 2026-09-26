# 论文实验与复现索引

整理日期：2026-09-22。范围以 `Paper/ICLR/iclr2027_conference.tex` 的有效 include graph 为准；本文档是公开版复现说明，不是新的实验验收。当前 `_Data/` 尚未迁移，以下读取历史输入的命令需在迁移完成后执行。本次整理没有训练、打分、拟合或更改 CFA 中的实验。

**DAS 必须按作者本次明确的论文定义执行：用有符号平方前分数 t 拟合背景，用原生 t² 计算 Full 和 BA 的删除子集和。** 历史 2026-09-22 V3 实際使用 t 拟合、t 聚合；现稿沿用的相关数值须重新聚合、验收后才能宣称符合当前定义。此前 AB2/DDPM 的原生平方空间拟合也不是 t 拟合。论文原样迁移，不在整理时替换这些数字。详见 [E-BENCH](experiments/E_BENCH.md) 与 [E-DEL](experiments/E_DEL.md)。

## 阅读与执行顺序

1. [共用上游管线](experiments/00_SHARED_PIPELINE.md)：环境、数据、主模型、查询、投影特征/曲率、子集重训、测损和方法评分。
2. 选择下面对应实验；复用已有且身份匹配的上游工件。`balds-run` 负责生产原始分数，`balds` 负责数组级评价，两者角色不同。
3. [产物迁移报告](migration/MIGRATION_REPORT.md) 决定实际迁入内容；实验报告中的路径均相对本仓 `_Data/`。
4. 图表使用已经接受的逐查询结果。文件存在、代码可运行、科学结果验收和论文采用分别记录。

| ID | 研究问题 / 稿件位置 | 复现报告 | 本次范围 |
|---|---|---|---|
| M1 | Figure 1(a,e)，小幅值区域的集中度和预测信息 | [M1](experiments/M1_MAGNITUDE.md) | 已入稿；原始分数后处理 |
| M2a | Figure 1(f)，完整 R16 的重复性与分带 LDS | [M2a](experiments/M2A_REPEATABILITY.md) | 完整64矩阵已验收；旧R4是前四次输入 |
| M3 | Figure 1(b–d)，幂变换、终端平方、同名单和跨平台控制 | [M3](experiments/M3_TRANSFORMS.md) | 已入稿；DDPM共同100与存档1000分开 |
| E-DEL | 主文 `tab:exp-selection`，BA选法的实际删除效用 | [E-DEL](experiments/E_DEL.md) | 原干预有效；当前 DAS 定义下重聚合/选法待做 |
| E-BENCH | 主文 Table 2、附录跨平台表及coverage | [E-BENCH](experiments/E_BENCH.md) | 当前方法范围与重评安排见 [SNR-LDS 计划](snr_lds_2026-09-25/PLAN_SNR_LDS_V1.md)；三项退役方法不列缺格 |
| E-SOURCE | 主文共同200查询、附录CFM400/DDPM400检索 | [E-SOURCE](experiments/E_SOURCE.md) | 已验收并入稿；三种查询集合不可互换 |
| S1–S4 | 附录固定头部选择、同名单变换、聚合分解、扩展方法 | [补充分析](experiments/SUPPLEMENTARY.md) | 保留现稿既有结果，不新增补格 |

## 所有有效实验图表标签

[当前include/标签图](evidence/paper_include_graph.json)记录原始路径与行号。这里保留 LaTeX 标签，避免排版后表号改变造成错误引用。`tab:exp-c2` 是 `tab:exp-ba-cifar` 的别名；`tab:exp-platforms` 是 `tab:exp-ba-platforms` 的别名。

| 实验 | 有效标签 |
|---|---|
| M1 | `fig:mechanism` (a,e), `tab:mech-concentration`, `fig:mech-density-full`, `fig:mech-head-full`, `tab:mech-band-lds` |
| M2a | `fig:mechanism` (f), `tab:app-r16-bands` |
| M3 | `fig:mechanism` (b,c,d), `tab:mech-transforms`, `fig:mech-power-full`, `tab:mech-peaks-c2`, `tab:mech-peaks-c10`, `tab:mech-fm-square`, `tab:mech-square-heads`, `fig:mech-square-heads-validation`, `tab:mech-ddpm-matched`, `tab:mech-ddpm-grid`, `tab:mech-ddpm`, `tab:app-square-comparison`, `tab:app-square-gains`, `fig:mech-native-ranges`, `tab:app-cross-paired`, `tab:app-score-signs` |
| E-DEL | `tab:exp-selection`, `tab:app-delete-existing`, `tab:app-edel-methods`, `tab:app-ba-feasibility`, `tab:app-ba-utility`, `fig:app-ba-fits` |
| E-BENCH | `tab:exp-ba-cifar`, `tab:exp-ba-platforms`, `tab:app-c2-ba-coverage`, `tab:app-native-coverage-cifar10-v2`, `tab:app-native-coverage-cifar2-das`, `tab:app-native-coverage-artbench2-256` |
| E-SOURCE | `tab:exp-retrieval`, `tab:app-retrieval-global`, `tab:app-retrieval-concepts`, `tab:app-retrieval-cfm400`, `tab:app-retrieval-host`, `tab:app-retrieval-ddpm-v2` |
| S1 | `tab:app-selection-complete`, `tab:app-deletion-pairs` |
| S2 | `tab:app-transform-utility-complete` |
| S3 | `tab:app-aggregation-controls`, `tab:app-count-controls` |
| S4 | `tab:app-extended-c2`, `tab:app-extended-c10`, `tab:app-extended-ab2` |

附录A/B是推导及方法构造，C是误差分析，D末尾是平方误差界与精确四样本反例；它们不是额外GPU实验。附录E定义背景拟合、评价与来源检索协议。

## 状态与证据边界

- `历史 accepted` 指原实验定义的验收，不自动升级为新定义的结果。
- BA 表按每方法自身有效查询平均；Full 对照必须用同方法相同查询。没有共同非空全方法交集也可报告，但不同查询集的排名是描述性比较。
- E-DEL 仍取全部候选四法共同有效查询；不能为了扩大查询数删除一个候选。
- 合法空选择/常数预测记0；拟合失败记NA；缺输入记missing；Journey validation记n/a。
- 像素/CLIP validation输入跨模型共享，不能伪报三次独立估计。C10 TracInCP/GAS仅seed42的既有覆盖须保留。
- 报告中的重新执行命令不是历史现场命令的伪装；历史执行版本和来源见各报告的“历史证据”。底层实际参数由保存的有效配置及工件身份核对。

## 明确退出的实验

2026-09-25 作者决定：Parameter-weighted D-TRAK、AbU+、NDA 退出实验基线，仅在 related work 中介绍。三者不纳入结果表、coverage、排名或待补任务；[旧配置记录](experiments/PENDING_BASELINES.md) 仅作历史资料。

不迁入 M2b 的250/500/1000采样预算干预，旧训练侧256坐标诊断，独立R4结论，ES-L学入发生率，ES-D来源删除，E1/A2受控排名/随机模拟，D1–D4扰动/重训随机性/测损随机性/mask数量分析，解析FM/额外LOO，BA1–BA3参数扫描与模拟。**共享输入例外**：R4完整分数中的repeat0–3仍属于当前R16；早期BA正側pilot仍在现稿附录。按依赖选择文件，不按目录日期一刀切。

依据：源项目 `Experiment/experiment_v2.md:15–21,38–91,315–327,430–441`，`Experiment/experiment.qmd:23–106`，以及当前 `Paper/ICLR/Sections/`。所引用证据在 [evidence/sources.json](evidence/sources.json) 登记。
