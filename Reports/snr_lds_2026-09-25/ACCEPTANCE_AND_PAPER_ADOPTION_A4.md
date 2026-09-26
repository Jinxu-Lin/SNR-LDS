# SNR a4 验收与论文采用（2026-09-25）

本轮由 Experimenter 验收，并按研究者直接授权兼任 Author 回填 `/path/to/BA-LDS/Paper`。
不是新的实验运行：没有重训、评分、拟合或 GPU 使用。

## 验收

- 运行依据：`TASK_JINXU3_SNR_LDS_CPU_V4.md`，r4 快照，Executor 回执 `RUN_JINXU3_SNR_LDS_a4.md`。
- 输入只读复用 `CFA/_Data/results/snr_lds_20260925/a3`；正式后处理输出 `a4`。
- 已读回 `verify_a.json`（pass，verified_queries=24300，errors=[]）、summary、E-DEL 和图文件。
- 接受可用输入范围的 243 格，不声称 248 格齐全。119 个方法/平台/轨道汇总，476 个阈值敏感性汇总。
- 24,300 物理逐查询记录中 16,946 有效、7,354 拟合失败，3,876 有效空选择。共享验证输入去重后的表格分母与物理记录总数不同，均已说明。
- 5 个缺输入格：DDPM TracInCP/GAS gen/val、Journey gen。Journey val 是 n/a。像素/CLIP 在当前规则下全失败，论文写 NA，不拿旧值填充。

## 采用位置

- `Paper/Sections/05_Experiment.tex`：主表及排序表述、E-DEL 方法选择、与来源检索比较的范围。
- `Paper/Sections/XY05_EvaluationExperiments.tex`：AB2/DDPM 表、覆盖表、阈值敏感性、两张固定拟合/分布图、删除效用及点位95%配对区间。
- 数值表由 `render_paper_tables.py` 从 a4 JSON 生成，位于 `Paper/Figures/evaluation/snr_{cifar,platforms,coverage,sensitivity}.tex`。
- a4 两张图复制至 `Paper/Figures/snr_lds/`。没有覆盖旧 B 诊断图，旧图继续明确标为历史 density-based 结果。
- PDF 已用 latexmk 编译，新增表/图无 overfull 或未定义引用；全文仍有与本轮无关的旧 notation 引用缺失 `eq:loo-target`、`eq:lds-prediction`、`eq:lds-response`，未擅自改写理论符号。

## 结果边界

- CIFAR-2 FM gen：FMAS 44.54、D-TRAK 44.50；val：DAS 50.45 最高。
- CIFAR-10 FM gen/val：DAS 40.51/44.07 最高。AB2 gen/val：DAS 30.91/38.48 最高。
- DDPM gen/val：FMAS 34.94/44.10，IF 34.69/44.00，差距不写为显著。
- 上述各方法均值使用各自有效查询，不能冒充跨方法同查询配对检验。
- E-DEL 使用四方法共同有效集（seed42/123/456 分别83/75/81），Full 选 DAS、SNR 选 FMAS；随后读取原有全部50个查询的删除效用。
- k300：DAS .007500393、FMAS .010237530，差 .002737137，95% CI [.002130861,.003396835]。
- k1000：DAS .010899235、FMAS .019352716，差 .008453480，95% CI [.007099821,.010070689]。
- B/R16 校准、固定筛选 head/squaring 诊断没有运行，本轮不验收或更新其结论。

## DDPM 后续

已备 `TASK_JINXU3_DDPM_SNR_CPU_V1.md` 和 `run_ddpm_tracin_snr.py`，复用既有评分器及 r4 SNR，不改科学算法。
适配器普通测试3项通过：四格路由/保留重训来源、错误val ID拒绝、缺方法拒绝；没有正式启动。
TracIn/GAS 四格待 xuchang3 现行双卡特征回传后由 jinxu3 CPU 执行。
Journey 新生成图与旧 GT 不是相同查询；其已有特征不等于已有可比 SNR 输入，独立保留缺口。
当前会话没有可用的跨 task 发信工具，任务书已落在 jinxu3 共享文件系统，但未声称已送达 Executor 或已运行。
