# SNR-LDS 改版：实验影响与执行顺序

> 2026-09-25 交付更新：Coder A/B 交付为只读r2快照，A按 `TASK_JINXU3_SNR_LDS_CPU_V2.md` 已ready。本计划下方“代码待实现”为初版历史状态。B实际R16路径、pilot格式及预测差方差需按 `CODER_FOLLOWUP_B_RUNTIME_V1.md` 修正；不阻塞A、不启动GPU。尚未启动正式重评。

日期：2026-09-25。负责人：Experimenter。状态：设计完成，代码待实现，未启动重评。

依据是本日读取的 `Paper/Sections/04_Method.tex` §4.3、`XY04_EvaluationPipelines.tex`、`XY05_EvaluationExperiments.tex`。代码以本项目 `Codes/src/balds` 为起点；历史 CFA 仓库及主库只读，不另建第二套评估实现。

## 1. 不是重新算归因分数，但也不只是改聚合阈值

| 环节 | 已有实现 | 新论文要求 | 是否重算 |
|---|---|---|---|
| 模型、特征、曲率、归因分数、子集实测响应 | 已归档 | 不变 | 否 |
| 中央拟合区间 | median ± 1.4826 MAD | 零中心 ± 1.4826 MAD | 是 |
| 对数密度拟合 | A+B x+C x²，均值自由 | A+C x²，均值固定零 | 是 |
| 筛选 | 密度比对应 gamma > .5 | abs(x)/sigma_x > 3 | 是 |
| 聚合 | 入选原值；历史 DAS 表示有差异 | 保号原值；DAS 用 t 拟合、t² 聚合 | 是 |
| LDS、coverage、排名、选法效用、相关图 | 旧规则 | 新规则 | 是，CPU |
| 来源检索 AP/Recall、实际删除模型/效用 | 已测量 | 排名/干预本身不变 | 不重算 |

旧拟合参数、旧 selected mask、gamma 和 BA-LDS 不能作为新结果复用。不能把旧自由均值拟合的 B 设零后继续使用 C；必须重新回归。RMS、原始输入和正确 ID 对齐可复用。

新规则只需在 101 个网格点算精确 KDE，不需要对所有 N 个训练分数逐点算 KDE。KDE 部分从 O(N²) 变为 O(101N)，后续还需子集聚合及统计；不能把复杂度比当成整链实测加速倍数。主重评不需要 GPU。

SNR 是本文规定的经验幅度比，不是旧混合分布后验概率，也不是功率比或 dB；阈值 3 是固定实验定义，不据 LDS 调优。

## 2. 本次实验划分

| 顺序 | 工作 | 已有输入/缺口 | 资源 |
|---|---|---|---|
| A1 | 四平台 Full/SNR 主表、逐查询结果、拟合图与 coverage | C2/C10 FM 三种子双轨；AB2/SD3.5 与 C2/DDPM 单种子双轨；使用实际已完成方法 | jinxu3 CPU |
| A2 | zeta=1,2,3,4 的敏感性、保留比例、排名与配对差 | 共用 A1 每个 query 的一次拟合，不拟合四遍 | jinxu3 CPU |
| A3 | 更新 E-DEL 的评价选法及真实效用 | 共用 A1；读原四候选、50 query、k300/1000 实测表，不重训 | jinxu3 CPU |
| B1 | 原有 head / 平方控制接入固定 SNR 选择 | 只在原方案条件上复用原分数/响应；不重新挑指数或头部范围 | jinxu3 CPU，A 后独立交付 |
| B2 | 现有 FMAS/D-TRAK R16 的 SNR 校准及 8/8 交叉诊断 | 已有完整重复和独立 pilot；不是重新生成 R16 | jinxu3 CPU，A 后独立交付 |
| 后续 GPU，非本 CPU 任务 | EK-FAC IF、DAS 完整重复；半倍/基准/双倍 MC 噪声干预 | 新附录明确待补的评分实验 | 另行代码/资源任务，不阻塞 A |

旧弱区域训练侧 MC 干预不等于新附录的完整估计器预算干预，不能直接改标签充当完成。新 GPU 项不包含新子集重训，也没有在本轮分配或启动。

## 3. 两个必须对齐的科学口径

**DAS：** 四个平台全部用原始有符号 t 拟合/选取、用 t² 一次平方聚合。旧 C2/C10 V3 使用 t/t，不能复用其中的 DAS Full 数字。非 DAS 在相同查询、响应、原生分数下，Full 逐查询值应不变；paired Full 均值可因新有效集变化而变化。

**E-DEL：** 当前附录要求以 CIFAR-2 generation、三模型种子的 benchmark mean 选出一个候选，再读取该候选在 seed42 原 50 queries 的两预算实测效用。旧 `four_method_selection` 是逐 query 选法，不是这个主实验。新选择使用四候选共同有效 benchmark queries，先每 seed 求均值再种子等权平均；Full/SNR 用相同共同集。效用评估仍覆盖完整 50 queries，不因 benchmark 某个 query 拟合失败而删去其干预。精确平局对并列方法效用等权平均，不借实测效用打破平局。此处以新版论文为准，不以旧 Reports 中尚未更新的流程为准。

## 4. 输入状态与复用优先级

源数据根：`/path/to/CFA/_Data`；新 BA-LDS `_Data` 尚未迁入科学工件，不使用空目录作为输入。

- C2/C10：`results/ba_c2_c10_das_presquare_20260922/inputs/bench_manifest.json` 是原 ID/来源参考，不直接复用其旧规则或旧路径格式。
- C10 补齐：`results/c10_cpu_closeout_20260923/inputs/bench_manifest.json`；不能继续按早期快照将 800 条一概记为缺失。
- AB2 IF：`scores/ekfac_if/artbench2_256/seed_42/` 和 `artbench2_256_val/seed_42/` 已有最终分数，不能按迁移说明中的旧状态把 U3–U7 记为待做。保留原 per-track oracle 阻尼和 development 标注，不重新按 SNR 选参。
- DDPM 九个非核心方法：`results/ddpm_noncore_val_repair_20260924/{cpu,shared_gpu}/ba_manifest.json` 已修正；执行回执在 `origin/codex/ddpm-ba-query-repair-j3-receipt-20260924`，提交 `0a19a458ef1271543f5ebb98e21fa11c5aee1d42`。val 原 ID 为 0..94,96..100，不能当作数组位置 0..99。
- DDPM TracInCP/GAS：XC3 双卡特征任务是独立在途工作。已有/后续完成的四份评分矩阵接入 SNR；未完成时列 missing_input，不为本重评再提取一次特征，也不把其新训练轨迹伪装为原模型轨迹。
- E-DEL：`results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv`，native 子表为 4×50×2=400 行；其余 transformed 条目不混用。
- R16 独立 pilot 来源：新项目 `Reports/evidence/r16_pilots.json`；重复数据读取原完整 R16 package/selection。使用配置中的真实 16 个 ID，不猜为前 16 个。
- 2026-09-25 作者决定：Parameter-weighted D-TRAK、AbU+、NDA 仅在 related work 中介绍，退出全部实验基线；不纳入主表、manifest、coverage、排名或缺件待办，即使已有评分也不接入本轮评价。保留既有实现与历史资料。Journey val 为不适用，不列待补实验。

## 5. 时间与交付边界

本轮 A、B 的新增 GPU 卡时均为 0。A 的 CPU 墙钟暂预留半天窗口，不是实测 ETA；用首个正式 C2/C10 面板计时，按各 N、方法/查询数与读盘时间更新，不另做冒烟。B 在 A 表交付后继续，避免诊断代码拖住主表。

交付两份任务书：`TASK_CODE_SNR_LDS_V1.md`、`TASK_JINXU3_SNR_LDS_CPU_V1.md`。运行书当前是 planned；唯一实质前置是新算法及完整调用包尚未实现。Coder 交付可运行包后按既定 CPU 范围直接执行，不再添加研究审批或重做已通过代码测试。
