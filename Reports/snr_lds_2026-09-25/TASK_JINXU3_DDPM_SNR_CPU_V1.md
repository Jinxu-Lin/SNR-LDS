# jinxu3：DDPM 非核心基线 CPU 续接（当前 SNR 规则）

task_id: DDPM_SNR_CPU_JINXU3_20260925；revision: 1；attempt: j3-ddpm-snr-a1。Executor: jinxu3。
状态更新（2026-09-25 16:18 AEST）：研究者授权本会话直接执行，已由 `run_ddpm_recovery_j3.py` 在jinxu3 cuda0接管GPU缺口，之后自动执行CPU `run_local`。**不再单独执行本书旧launch，也不等待原incoming全量回传，防止重复启动。** 详见同目录 `RUN_DIRECT_DDPM_RECOVERY_J3_a1.md`；以下原命令作为V1历史保留。
本书替代旧 `TASK_JINXU3_CPU_V2.md` 的 density-BA 后处理，不重复 GPU 特征，不运行旧 BA。

## 已完成与本次增量

- DDPM 已有的 26 个方法×轨道输入已包含在 SNR a3/a4，结果已填入论文；不重算这些拟合。像素/CLIP 的 NA 是既定拟合失败，不是待补任务。
- 只接续 TracInCP、GAS 的 gen/val 四格，共 400 条逐查询记录。12 份梯度特征由两方法共享。
- 2026-09-25 早间查见 xuchang3 两条特征 worker 仍在运行：cuda2 PID 2698066、cuda3 PID 2698065；15:23复查时均已退出，step2000/6000的6份完整特征可复用，step4000/8000尚缺。不抢占、不在 jinxu3 复制 GPU 工作。
- Journey gen 是独立缺口：已生成的 Journey 图为新 namespace，并非旧 GT 的查询。此次不把新轨迹配到旧响应，不启动额外生成/GT计算。Journey val 为 n/a。

## 输入与版本

主库 `/path/to/CFA/_Data`；CPU 2 线程，GPU 0 卡时。不等待 cuda0 空闲。
评分代码 `19c31140c300d344f827f15e04d5ff460ad9f6ac`；SNR 复用已验收快照 `/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4`。
小型调用脚本与本任务书同目录：`run_ddpm_tracin_snr.py`，只调用既有 ingest、score、SNR batch/summary/verify。

已有上游 lane 自动将下列四类文件按相对路径回传至
`results/ddpm_tracin_gas_20260924/incoming/cuda2/` 与 `incoming/cuda3/`：
每 checkpoint 的 FEATURE_META、TRAIN_FEATURES、QUERY_FEATURES gen/val；cuda2 为 step2000/4000，cuda3 为 step6000/8000。
两边合计 12 个矩阵和 4 份 meta。原 lane 自带 collect；无需全库 rsync 或复制 manifest.db。

若某条 lane 已明确完成，但仅回传失败，原 collect 命令即可重传，不重提特征：

```bash
ssh xuchang3 'cd /tmp/cfa-ddpm-tracin-gas-xc3-v2 && /path/to/miniconda3/envs/da/bin/python _Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py collect --gpu 2'
ssh xuchang3 'cd /tmp/cfa-ddpm-tracin-gas-xc3-v2 && /path/to/miniconda3/envs/da/bin/python _Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py collect --gpu 3'
```

## 直接启动

两组 incoming 已完整后，检查本任务没有活跃 worker，然后执行。不要对仍在产出的源反复复制或提前运行 ingest。

```bash
git -C /path/to/CFA worktree add --detach /tmp/cfa-ddpm-tracin-gas-j3-snr-v1 19c31140c300d344f827f15e04d5ff460ad9f6ac
/path/to/CFA-envs/e3c-torch260/bin/python /path/to/BA-LDS/Reports/snr_lds_2026-09-25/run_ddpm_tracin_snr.py launch
```

若指定 worktree 已存在且正是该提交，直接复用，不重复 add。监督器脱离会话并持久化退出码。顺序为：
1. 既有 ingest 验证并登记 12 特征＋4 meta；
2. 既有 TracInCP/GAS 评分器、四 checkpoint 等权均值，生成四份 `(5000,100)` 矩阵；
3. 使用 a3 已修正的 mask/response/query ID 清单，接入四份新分数；val 使用实际原 ID，不按列号冒充 ID；
4. SNR batch（零均值 Gaussian，默认 ζ=3，敏感性1/2/3/4）→summary→400条逐query source-algebra verify。

CPU 排期暂留1–2小时（不是本配方实测承诺）；首个正式方法完成后更新 ETA，不额外冒烟。

## 状态、产物与验收

```bash
/path/to/CFA-envs/e3c-torch260/bin/python /path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools/e3c_launch.py status --output-root /path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/a1/supervised
tail -n 30 /path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/a1/supervised/launcher/*.log
```

新输出：`/path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/a1/`，含 inputs/manifest.json、panels、tables、verify.json、supervised。
标准分数仍在既有 `cifar2_das_retrain_20260924` namespace，不改成原模型。
必须保留 `completed_source_identity.scoring_config.retrained_trajectory=true`：这些梯度来自重训轨迹，而非原训练丢失的中途 checkpoint；查询与 GT 则是原查询。回执明确该差异，不能把结果描述成原 checkpoint 恢复。

验收：四矩阵有限、gen/val 原 ID 对齐；400/400 逐query记录及 verify pass；拟合失败=NA、有效空选择=0。报告 Full/SNR 同有效集均值、覆盖与阈值敏感性。Author 最终采用时应保留重训轨迹说明。
不触碰 a3/a4、不重跑 E-DEL、不启动 B、不自动修改论文。

如发生外部中断，确认旧 worker 已结束后，允许一次原命令续跑；ingest/score/batch 复用完整产物。确定性代码错误记录并报告，不改阈值。回执写 `RUN_JINXU3_DDPM_SNR_CPU_a1.md`，完成后交 Experimenter 验收。
