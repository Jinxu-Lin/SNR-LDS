# Coder：SNR-LDS 最小评估改造

task_id: CODE_SNR_LDS_20260925；revision: 1；owner: Coder；acceptance: Experimenter。
状态：ready_for_coding，不是实验启动授权。依据及实验边界见同目录 `PLAN_SNR_LDS_V1.md`。

## 起点与非目标

直接修改 `/path/to/BA-LDS/Codes` 的现有公开包；它已经具备输入对齐、DAS fit/aggregation 分离、LDS、面板和统计工具。不要回 CFA 另写一套，也不要全库重构或改训练/特征/曲率/评分流程。

当前 BA-LDS 目录没有 Git 元数据。交付记录实际源版本/改动文件和测试，提供固定的代码快照供长作业使用；不要凭空写 execution commit。若研究者之后把它纳入 Git，记录真实提交即可，不为本任务新建版本平台。

## A：先交主表与删除效用

| 需求 | 已有位置 | 本次增量 |
|---|---|---|
| 拟合/筛选 | `src/balds/evaluation/background.py` | 增加零均值 SNR 拟合及严格阈值，不走逐样本密度比 |
| 分数/GT/ID 与 DAS | `src/balds/workflows/benchmark.py`、`tools/prepare_benchmarks.py` | 保留显式 fit/aggregation 和原 ID；清单接入最新已完成源 |
| 批量评估 | `src/balds/cli/ba.py` | existing evaluate/batch 增加显式 SNR rule、zeta；一次拟合支持多阈值 |
| 表格/图 | `src/balds/report/tables.py`、`tools/summarize_benchmarks.py` | 输出 snr_lds 与新状态统计，不把旧 ba_lds 改名冒充新结果 |
| 删除选法 | `four_method_selection`、benchmark deletion | 新增 benchmark-level 选法；旧逐 query 逻辑不直接拿来当新主表 |
| 执行封装 | 现有主库输入/既有 supervisor | 交付固定 J3 配置、串行 launcher、status/resume/verify 完整命令，不造调度系统 |

### A1. 数值定义（必须实现这一版）

每个 method/query 的 fitting vector v：

1. 稳定计算 RMS r；x=v/r。全零直接 valid empty，LDS=0。
2. h=.9 min(SD(x,ddof=1), IQR(x)/1.34) N^(-1/5)。分位数沿用 linear。
3. a=1.4826 median(abs(x-median(x)))；grid=linspace(-a,a,101)。不减去 median，不移动 grid 中心。
4. 复用精确 float64 Gaussian `kde_log_density(grid,x,h)`。
5. 最小化 sum f(grid)*(logf−A−C*grid²)²；设计矩阵只有 [1,grid²]。权重可用 exp(.5*(logf−max(logf)))；rank=2。
6. C<0 时 sigma_x=sqrt(-1/(2C))，sigma=r*sigma_x。没有 mu0/pi0 判断。
7. selected=abs(x)/sigma_x > zeta，默认3；严格大于，双侧，不乘 SNR、不乘 gamma。同一 fit 同时生成 zeta=1,2,3,4，不算 KDE(x,x)。
8. 用 D=1−K、原 aggregation score 计算 D@(selected*score)；LDS 沿用 Spearman average ties。普通方法保号原值；DAS 输入 signed t，fit=t、aggregate=t²，仅平方一次。

失败：非正 bandwidth/width、非有限/秩亏拟合、C>=0、无效 sigma => NA，不填0。valid empty、constant prediction、constant response => LDS0，并独立计数。所有 query 保留 Full，配对表只在该方法新规则有效集上比较 Full/SNR；不要因 Full 可以计算便把拟合失败称为有效。

旧 fit 缓存不可复用；只保存新版本的 RMS、h、a、grid/log-density、A/C、sigma、状态、selected 及预测。新 rule 标识建议 `snr_zero_mean_gaussian_v1`。不要求另建 hash/契约机制，不保存无用的逐样本 KDE 密度，避免旧 37 GiB 缓存模式。

### A2. 清单与表示接入

使用计划列出的已完成源、`prepare_benchmarks.py` 的显式 ID 逻辑及 DDPM 修复来源。对已有 100-column 的 DDPM val 分数，其列顺序对应原 IDs 0..94,96..100；GT 按这些 ID 取列。1000-column 源按自身 IDs 选择，不二次把原 ID 当局部列位置。

AB2 DAS 需要 lambda=1 的 signed t；若主库没有对应文件，复用现有 `controls.das_linear` 从原特征 CPU 恢复，产物放本次 inputs。不重提特征，不使用旧 lambda=.5，不用 sqrt(square) 猜符号。准备工具的输出位置须由调用者指定，不写回旧科学输入目录。

方法清单沿用 prepare_benchmarks.METHODS。2026-09-25 作者已将 Parameter-weighted D-TRAK、AbU+、NDA 退出实验基线，仅保留 related work 介绍；不接入其评分，也不为这三法生成主表、coverage、排名或 missing 待办。不把迁移报告旧 missing 当实时状态。源缺失/not_applicable/fit_failed 分开。DDPM TracIn/GAS 的重训模型身份用显式覆盖路径/元数据接入，不硬塞原 alias；可先完成其余方法。

### A3. 输出与 E-DEL

主表输出每个平台/seed/track/method 的逐 query Full/SNR、状态、保留数/比例、sigma、原 query ID；阈值敏感性共享拟合并完整报告四阈值。跨seed先逐seed平均再等权，seed SD 与 query CI分开。2000 paired query bootstrap，seed20260920；共享 val ID 跨模型同步抽样，model-independent 输入不当作独立模型重复。

E-DEL 使用 C2 generation 三seed、四候选 FMAS/D-TRAK/DAS/IF 的共同有效 query 集；每seed上 Full/SNR 同集，种子等权，分别选最高 benchmark mean 的候选。两个删除预算使用同一次选择。读取原 native 效用400行，输出完整50query的选择后效用、k300/1000均值/差/CI及选择来源。精确平局取并列候选效用平均，不用效用选赢家。若共同集为空，不静默减少候选，输出 unavailable 并继续其他表。旧逐query E-DEL 可保留旧入口，不列为本轮必跑条件。

拟合图固定 C2 seed42 gen q0、q1，四核心方法；显示原分布、零均值拟合及 ±3sigma，DAS注明 fit=t、aggregate=t²；另输出每方法 sigma/coverage 分布。不按结果挑图。

### A4. 普通代码验证与交付

- 用已知 A,C 的网格 log-density 检验加权回归、sigma；非对称样本也必须保持均值零。
- abs(x)=3sigma 必须不入选；正负同幅值相同选择；整体正缩放保持选择/LDS。
- 手算小矩阵核对 D=1−K、保号聚合；DAS选负 t 后贡献 t²、不变成 t 或 t⁴。
- 零向量/合法空/常数=0；无效拟合=NA；四阈值仅拟合一次；SNR 路径不调用 N-point KDE。
- 非连续 query IDs 对齐；benchmark 全局选法与逐query选法构造不同答案的普通测试；已知原效用不参与排名。
- 保留原接口必要回归；运行现有 layer 检查和受影响测试。不是正式冒烟实验，不需要另跑正式模型或全库审计。

交付 A 后即可运行主表，不等待 B。更新 README 的 SNR 命令、代码回执；完成配套 J3 运行书的真实环境、固定代码路径、完整已实现 launcher/状态/恢复/验收命令。没有正式实验授权给 Coder。

## B：已有数据的附录分析，A 后独立交付

复用 `tools/aggregation_controls.py`、`workflows/controls.py`、`workflows/repeatability.py` 的加载/统计，不改 R16 评分器。

1. 原 head/平方控制：在原有方法/平台/头部比例范围上，先从原分数拟合 SNR mask，一直固定；head变化取与该mask的交集，不针对变换后分数重新拟合。原值、原平方/保号平方 readout 沿用原控制的定义，不能再选有利指数。输出旧 Full 控制和新 SNR 固定集合的配对结果。
2. 已有 R16 FMAS/D-TRAK，C2 seed42、原16 gen+16 val；用独立 pilot 拟合sigma、冻结 zeta3 mask。输出 pilot SNR分布、重复标准差/sigma、retained/excluded repeatability、中心化重复残差分布、各repeat Full/retained/excluded LDS、删除子集预测差的方差。方差由完整 subset prediction differences 计算，不能假设训练坐标独立后相加。
3. 8/8 交叉参考：固定repeat0..7与8..15，两向交换；学习fold用 abs(mean)/sample_SD(ddof=1)>3，分母不是SD/sqrt8，评价fold逐个repeat评估后汇总，不把8个repeat平均再当主结果。零SD且均值非零按无限比、零均值零SD不入选，明确记数。
4. 不把所有训练坐标当独立统计样本；不确定性以query为单位。独立pilot来源及实际query IDs随结果留存。缺 IF/DAS repeats 记pending，不在本任务补算。

B 也只用 CPU。与 A 共用 SNR 拟合函数，不再实现第二套公式；最小差量测试后交付独立调用，已验收 A 不重跑。
