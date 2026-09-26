# DDPM CPU 阻断的上游核实与更正

检查时间：2026-09-25 15:21–15:23 AEST。检查者：Experimenter；只读 SSH 和工件读回，没有运行实验、搬运、重启或停止进程。
对应回执：`RUN_JINXU3_DDPM_SNR_CPU_a1.md`。保留原回执，以下追加更正库存事实。

## 已确认

- jinxu3 CPU 暂停正确：incoming 两组尚未齐套，不能开始四 checkpoint 评分。
- xuchang3 两 worker 2698066/2698065、两 supervisor 均已退出。正确 lane output-root 下的状态是 `lost`、exit_code=null，而不是没有历史作业。`status=[]` 不能用来判断无历史运行。
- 两日志分别最后更新于09-25 08:25:07、08:21:26 AEST；step4000/8000 的 train 日志到约1100/5000行，末行截断，没有退出码或明确异常栈。机器 uptime 起点为09-01，排除这次机器重启；不能据此断言 OOM 或人为终止。
- 特征位于 `/path/to/CFA/_Data/featurize/trak_T100/cifar2_das_retrain_20260924/ddpm/seed_42/`，不是任务 `results/` 子目录。

| checkpoint | train | gen | val | meta |
|---|---|---|---|---|
| 2000 | 已完成(5000,4096) | 已完成(100,4096) | 已完成(100,4096) | 已存在 |
| 4000 | 未保存 | 未生成 | 未生成 | 已存在，仅表示配方已写入 |
| 6000 | 已完成(5000,4096) | 已完成(100,4096) | 已完成(100,4096) | 已存在 |
| 8000 | 未保存 | 未生成 | 未生成 | 已存在，仅表示配方已写入 |

已逐一用 torch.load 读回2000/6000共6份矩阵，形状与有限性通过；meta的T100/p4096及val IDs 0..94,96..100正确。剩余6份未完成，不是12份全部不存在。

## 后续处理

既有 `TASK_XUCHANG3_V2.md` 的a2恢复命令会自动复用2000/6000的完整张量，再完成4000/8000并自动collect。不需要修改评分或SNR代码。
当前 featurize 仅在整份矩阵结束后保存，**不能从1100行续算**；4000/8000的train从0开始。按中断前约9.52秒/训练行外推，每卡剩余约14小时（含两份查询的粗略同速估算），两卡合计约28卡时；新负载下会变化，不是墙钟承诺。

截至本次检查，xuchang3 cuda0–3均有其他Python任务占用（各约13–15GiB），本轮未启动恢复。原卡释放时按现有恢复授权执行，或研究者指定改派后更新机器命令；不自动抢卡、不在jinxu3额外启动GPU。
jinxu3任务状态是 `blocked_upstream`，而不是可立即开始的CPU任务；齐套回传后按V1原命令执行，无需新的科学裁决。

## 可复用的精确命令（由 xuchang3 Executor 执行）

查看原任务状态：

```bash
/path/to/miniconda3/envs/da/bin/python /path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools/e3c_launch.py status --output-root /path/to/CFA/_Data/results/ddpm_tracin_gas_20260924/supervised/xc3-cuda2-a1
/path/to/miniconda3/envs/da/bin/python /path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools/e3c_launch.py status --output-root /path/to/CFA/_Data/results/ddpm_tracin_gas_20260924/supervised/xc3-cuda3-a1
```

对应物理卡释放、确认无重复worker后，沿用原V2的一次a2恢复；各卡独立，不要求同时空闲：

```bash
cd /tmp/cfa-ddpm-tracin-gas-xc3-v2
/path/to/miniconda3/envs/da/bin/python _Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py launch --scope features --gpu 2 --attempt xc3-cuda2-a2
/path/to/miniconda3/envs/da/bin/python _Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py launch --scope features --gpu 3 --attempt xc3-cuda3-a2
```

不执行当前不完整lane的collect，不用一半checkpoint提前生成正式评分。不改写历史a1运行回执。
