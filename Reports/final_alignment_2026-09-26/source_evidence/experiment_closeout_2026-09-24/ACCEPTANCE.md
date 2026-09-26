# 2026-09-24 实验收尾验收

验收人：Experimenter；主库：`anonymous-lab3:/path/to/CFA/_Data`。
时间：2026-09-24 10:33 AEST。本轮不新增训练、GPU打分或KDE；执行已有CFM回传/CPU合表并只读复核其他结果。论文采用统一记 **pending**，不代改Paper。

## 1. 本轮结论

| 内容 | 状态 | 本轮处理与边界 |
|---|---|---|
| CFM 最终审核v2：11方法查询特征、评分、来源检索 | accepted | 5类新500查询特征、11份50000×500分数、4400逐test记录；100val/400test，每概念10/40；不混旧v1 |
| CFM FMAS / EK-FAC IF：100val＋400test | accepted | XC3四lane均exit0；本轮完成回传、主库读块验证与CPU合表；100val块＋400test块、两份50000×400矩阵、800逐test记录 |
| CFM 当前13方法共同400检索表 | accepted | 上述两批同一v2查询身份，合计5200逐query；13方法汇总见下表；旧v1表不作为当前结果 |
| C10 TracInCP/GAS：seed123/456中途特征与双轨评分/BA | accepted | 六个checkpoint单元、18原始producer登记均匹配主库；8新增评分矩阵有manifest；800新增Full/BA逐query复算通过 |
| C10 TracInCP/GAS：三种子合表 | accepted | 复用seed42的400条，合计1200条、12单元、4组均值及样本SD；5个合法拟合失败，1194个合法空集合 |
| DDPM Journey新生成与特征 | accepted_features_only | 原模型重新生成100张、各10个真实DDPM轨迹点；train5000×4096、query100×4096有限。尚无新query对应GT或最终评分/LDS |
| DDPM非核心CPU/共享GPU线 | partial / blocked | 18份有限评分矩阵已在库、gen面板900条；val面板编号错位，不能将整线标完成。保留可复用特征/分数 |
| RunPod DDPM FMAS/IF | produced / transfer_pending | 远端完整500块已通过原verify；25GiB回传仍在进行，尚未做主库完整验收/native import；不是已入库闭合 |
| J2 DDPM重训练 | ready / no_start_evidence | 本次未见约定输出目录或worker，亦无新执行回执；不宣称完成/失败 |
| 旧CPU E-BENCH | failed / obsolete_rule | 原job已exit1，末尾为native score shape/finite mismatch；旧规则结果不覆盖当前已验收BA，不盲目重启 |

## 2. CFM 最终审核v2：可交Author的共同400表

输入既有身份：`cb0f6c7240e814c17c945abaa53e9dd0633b7bfcdb99ea5a3ab75773ea6c6cd7`。
全部方法均用100val选择参数，400test报告；全库AP和Recall@200、等概念权重。
这不是旧CFM400，也不是旧common200。FMAS选参1.0、IF选参1e-9，来自新100val，未用test调参。

| 方法 | AP global (%) | Recall@200 global (%) |
|---|---:|---:|
| das_T100 | 21.162299 | 23.157500 |
| dtrak_T100 | 23.489718 | 24.786250 |
| trak_T100 | 7.452720 | 11.815000 |
| pixel_dot | 0.425591 | 0.098750 |
| pixel_cos | 1.252607 | 2.207500 |
| clip_dot | 6.798116 | 8.646250 |
| clip_cos | 12.531519 | 14.285000 |
| grad_dot_T100 | 0.533242 | 0.591250 |
| grad_cos_T100 | 0.618718 | 1.002500 |
| relative_if_T100 | 7.259059 | 11.898750 |
| renorm_if_T100 | 6.813276 | 10.796250 |
| ekfac_if | 23.498299 | 26.072500 |
| fmas_raw | 25.310884 | 27.673750 |

11方法原始逐query：
`_Data/scores/<method>/cifar10_inj4_inject/seed_42/inject_result.json`。
汇总：`_Data/results/cfm_reviewed500_20260923/verified_summary.json`。

新增两方法原始逐query、fine/coarse与逐概念：
`_Data/results/cfm_test400_20260923/analysis/{fmas_raw,ekfac_if}/result.json`；
矩阵：同目录`test400_scores.npy`；总表：`analysis/SUMMARY.json`。
原100val：`_Data/results/cfm_val100_20260923/incoming/{xc3-fmas,xc3-if}/`。
两方法是隔离结果根的正式交付，不将400列矩阵冒充标准500列inject_scores，也不声称全库manifest闭合。

CFM四lane合计42.6838单卡墙钟小时；前序100val为10.6895小时，合计53.3732个4090卡时。
11方法query阶段另约13分21秒；其CPU阶段不计GPU时。本次收尾新增GPU计算为0。

## 3. C10 补缺及正式性

主库：`_Data/results/c10_cpu_closeout_20260923/`；
`PER_QUERY.json`保留1200条，`SUMMARY.json`保留12个单元/4组，
`TABLE.md`可直接交Author。新增800条的原分数、删除mask、GT、
Full预测、gamma>0.5硬筛选、保号原值聚合、BA预测和Spearman LDS均独立复算。
存储mask是keep-mask；删除预测使用1-mask。

六档特征均已回库；按原producer记录核实18条manifest，无新增哈希。
8个新增score路径有标准manifest。沿用XC1已完成的特征数值检查与本次prepare实际检查，
不再次读取/哈希整套大型特征。

三种子1195/1200 BA有效，1194个有效查询为空集合，按当前定义LDS=0；
其余5个是fit_failed，保留NA。这是已测负结果，不能写成“BA普遍改善”，也不能将NA按0填入。
论文表使用各方法自身有效query的配对Full、等seed均值和样本SD，不把1200条当独立seed。

## 4. 尚未整体验收的边界

### DDPM非核心BA

CPU四方法及共享梯度五方法均在val出现相同错误：
要求q0..q99，实际panel为q0..q94,q96..q100；q95缺失、q100标missing_query。
任务wrapper的panel/score/response查询坐标没有统一。
不能仅把q100改名为q95；应按实际原查询ID核实分数列与GT列再重做受影响val BA。
修复责任为Experimenter组织Coder修正任务局部接入；已算特征/18份scores保留，
不因BA接入错误重新提梯度或重新打分。本轮没有启动修复运行。
gen面板900条（795有效/105拟合失败）已产出；本次做了覆盖/有限性检查，
尚未将其描述为完成两轨任务或完成全部逐query预测独立复算。

### Journey与DDPM新训练

Journey新产物：`_Data/results/ddpm_gaps_20260923/journey_j3/`，
执行08c4451；100PNG、100×10轨迹、两个特征矩阵实际读回有限且形状正确。
单卡墙钟约7.1144h，期间获准与其他轻显存作业共享，不当作独占吞吐基准。
新噪声种子20260923、DDIM50/eta0；新query身份不能配旧query的GT列。
验收限于任务书要求的生成/轨迹/特征交付，**不表示Journey BA-LDS已补齐**。

J2原重训练书bd07351依然存在；只读查看anonymous-lab2，
约定`/mnt/hdd/anonymous/CFA_ddpm_retrain_20260923/run`尚不存在且未见对应worker。
可能尚未转发/启动，不能从缺少记录断言训练崩溃。本轮未自行启动。

### RunPod

2026-09-24 09:55 AEST远端verify通过，两方法val100/test400完成；
本机monitor仍从`/workspace/ddpm_return_20260923`回传25GiB包，
主库有complete.json不代表传输已完成。本轮未改动monitor或并发覆盖其目标目录。
待回传结束后执行已交付import/native登记与本地复验，再更新为accepted。
Pod目前仍在线、无GPU计算进程；数据回传和验收完成后需按研究者授权释放，
**本轮未关闭/删除Pod**。

## 5. 检查与执行记录

- 只读验证脚本：本目录`verify.py`；结果`VERIFY.json`。
- Python：`/path/to/CFA-envs/e3c-torch260/bin/python`；
  `CUDA_VISIBLE_DEVICES=''`，OMP/MKL/OPENBLAS均2线程。
- 实际CPU核验：
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /path/to/CFA-envs/e3c-torch260/bin/python /path/to/CFA-worktrees/experiment-closeout-20260924/_Data/reports/experiment_closeout_2026-09-24/verify.py`。
- 已有CFM收尾命令：
  `ssh xuchang3 'cd /tmp/cfa-cfm-test400-xc3 && bash _Data/reports/cfm_test400_2026-09-23/run_host.sh finish'`。
  代码ebbc59f；执行前确认四lane exit0且无重复finish，命令exit0；
  含4lane本机verify、定向rsync、J3两套val verify与CPU merge。
  每个test方法/lane均50块，通过后主库merge再次逐块读回。
- 验证脚本早期检查将keep-mask误当删除mask，已按生产定义改为1-mask；
 另纠正只读脚本的`bench/`定位为非核心结果的`ba/`。
 这些是验收脚本修正，未改实验工件或科学参数。

原执行证据：CFM11方法回执13ade47；XC3四卡任务/进度959ccb9/f747801；
C10任务8264909及launcher兼容偏差cf9781c、实际job exit0；
DDPM两失败回执a380385/d3226c5；
XC1特征验收7dfc9d9；Journey bd07351/08c4451及主库result/launcher。
可信已有检查与本次读取相互补充，不把执行者自检称作新的独立GPU重跑。

## 6. 登记与交接

已向当前工作区`Experiment/experiment.qmd`及`Experiment/experiment_v2.md`
追加本日状态和本报告入口；旧段落保留历史并由新条覆盖。
未修改Author正在编辑的Paper，也未把整个dirty工作树提交。
本报告/验证材料单独发布；台账新增段落另附`LEDGER_UPDATE.patch`，避免混入未提交的Author重构。
