# jinxu3：SNR-LDS CPU 正式重评执行书 V2

> 修复交付后续：r3注册修复已核对，A的新执行依据为 [V3](TASK_JINXU3_SNR_LDS_CPU_V3.md)，使用j3-snr-a3及a3输出。本版r2命令不得再次执行。

> 2026-09-25运行更正：r2的A已在prepare确定性失败，当前状态改为 **blocked**，下文ready和launch仅保留为当时授权记录，不能再次执行。根因为独立prepare入口未导入已有cifar2_das注册模块；见 `CODER_FIX_A_DATASET_REGISTRATION_V1.md`。无SNR产物，旧日志保留。等待修正版执行依据，不原样重试。

task_id: SNR_LDS_CPU_JINXU3_20260925  
revision: 2；attempt_id: j3-snr-a1；owner: jinxu3 Executor；acceptance: Experimenter。  
状态：**A ready，可以按本书启动；B 待接入修正，不自动串接。**

本书替代 V1 的启动约束和 Coder 附录中“等待另行放行”的条款。用户已要求在 jinxu3 执行该 CPU 任务，A 不再等待新的 commit、发信或人工门控。代码中的 `formal_launch_authorized=false` 是旧交付记录，launcher 不读取它；无需修改只读快照。本文没有启动作业。

## 1. 已落实的执行依据

- 固定代码目录：`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2`。**不用无后缀版、r1 或仍可能编辑的 Codes 目录。** 当前项目没有 Git 元数据，版本按该只读快照与 `CODER_RECEIPT.md` 记录，不编造 execution commit。
- Python：`/path/to/CFA-envs/e3c-torch260/bin/python`。
- 实际入口：快照内 `tools/run_snr_lds_a.sh`，具有 launch/status/resume/verify；已核对源码、CLI 帮助与 shell 语法，未跑正式实验或冒烟。
- Coder 已交付普通测试：326 passed、1 skipped，layer check 通过。Executor 不再重跑同一代码测试或重新设计流程。
- 输入全部从 `/path/to/CFA/_Data` 读取；**无需 scp、整库迁移、模型下载或环境重装。** 四份来源 manifest、AB2 DAS 恢复所需 train/gen/val 特征和 error_train、E-DEL 效用文件本机可读。
- 固定配置 `configs/snr_lds_j3.json` 是交付说明；实际 shell 的固定参数和实参才是本次运行命令。不可只改 JSON 便以为 launcher 的路径/线程随之改变。

本次核对时主机 anonymous-lab3，约25 GiB可用RAM、155 GiB磁盘余量；未发现本任务worker，输出根尚未创建。这是出单时观测，Executor 启动前只需一次现场检查。

## 2. A 的计算范围

1. 四平台：C2 FM/C10 FM seeds42/123/456；AB2 SD3.5/C2 DDPM seed42；gen/val各100原query。共16个面板，CIFAR64子集、AB2 32子集。
2. 在役方法沿用现有16方法清单，Journey val不适用。最多248个适用方法格、24,800个method/query记录；实际源缺失分开列，不宣称全部格完成。Parameter-weighted D-TRAK、AbU+、NDA已退役，不接入也不列待办。
3. 零均值 Gaussian SNR；strict abs(x)/sigma_x > zeta；zeta=1/2/3/4共用一次拟合，主结果为3。DAS用signed t拟合，t²原生聚合；其余方法保号原值。保留旧模型、原评分参数、GT和原ID。
4. 输出主表、敏感性、coverage、逐query/预测/拟合、固定C2 seed42 gen q0/q1图。
5. E-DEL从C2 gen三seed benchmark共同有效集选一个全局候选，再读原50query×k300/1000实测效用。四候选不变，不重训、不重排实际删除名单、不按效用选赢家。

DDPM val九非核心方法使用修复后的来源覆盖清单（原IDs 0..94,96..100）。DDPM TracIn/GAS若本次清单没有其新训练轨迹的完整评分，记missing，不转去提特征，也不把新模型评分强塞为原模型；后续单独补入。缺评分、fit_failed、合法空集合和n/a分别记录。

## 3. 直接执行：检查一次，然后启动

仅CPU：单worker、4线程，CUDA隐藏。预计单面板内存低于16 GiB；为串行输出预留10 GiB，首正式面板更新实际峰值/ETA。半天为原排期窗口，不是实测承诺，不因超时估计自动停作业。

### Step 1：现场检查（无计算、无全库hash）

```bash
hostname
free -h
df -h /path/to/CFA/_Data
ps -eo pid,ppid,lstart,args | rg '[r]un_snr_lds_a.sh|[b]alds.cli.ba batch|[r]un_snr_lds_b.sh'
```

确认是 anonymous-lab3，且没有同任务worker；若已运行则接管监控，不重复launch。不要求GPU空闲。正常资源下几分钟内直接进入Step 2，不等待接单回执发布。

### Step 2：一次正式启动 A

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2
./tools/run_snr_lds_a.sh launch
```

launcher 自动设置 `PYTHONPATH=<该快照>/src`、`CUDA_VISIBLE_DEVICES=''`、OMP/MKL/OPENBLAS/NUMEXPR各4线程、PYTHONUNBUFFERED=1，并通过nohup脱离会话。不要再包一层自写launcher或同时启动其他面板。

**链内已包含全部命令，不另手动 prepare：**

1. 生成显式input manifest；必要时从旧特征在CPU恢复AB2 lambda1 signed t，仅写新inputs。
2. `balds.cli.ba batch --rule snr_zero_mean_gaussian_v1 --zetas 1,2,3,4 --primary-zeta 3 --save-fits`。
3. summary表 → benchmark级E-DEL → 固定图 → source-algebra verify。

完整实参保存在只读 `tools/run_snr_lds_a.sh`；回执记录该入口、快照和实际启动命令即可，不让Executor临场拼命令。旧fit/gamma/selected均不作为新拟合缓存。

### Step 3：确认进度和状态

```bash
cat /path/to/CFA/_Data/results/snr_lds_20260925/a1/launcher_a/pid
tail -n 40 /path/to/CFA/_Data/results/snr_lds_20260925/a1/launcher_a/run.log
```

prepare尚未生成manifest时，上述log是准确信息；不要过早运行status后把“manifest还没有”误判为失败。manifest生成后使用：

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2
./tools/run_snr_lds_a.sh status
```

`launcher_a/pid`是运行shell PID，不一定是当前Python子进程。按该PID查看子进程并记录实际Python PID；保存命令如下，不用宽泛进程名发信号：

```bash
SNR_PID=$(< /path/to/CFA/_Data/results/snr_lds_20260925/a1/launcher_a/pid)
ps -p "$SNR_PID" -o pid,ppid,lstart,etime,args
ps --ppid "$SNR_PID" -o pid,ppid,lstart,etime,args
```

首正式面板属于原任务，正常即继续。报告处理量/阶段、wall time、内存和按C2/C10分开的ETA；不另做试跑，不高频反复扫描整库。

### Step 4：结束后验收、提交回执

```bash
cat /path/to/CFA/_Data/results/snr_lds_20260925/a1/launcher_a/exit_code
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2
./tools/run_snr_lds_a.sh verify
./tools/run_snr_lds_a.sh status
```

所有输出已在jinxu3主库，无回传步骤。A完成即可交付；**不要接着运行r2的B launcher**，原因见§6。

## 4. 中断恢复——保留日志，不重做有效query

允许一次已识别的临时退出恢复，逻辑attempt记j3-snr-a2；源/规则/阈值不改。先确认旧shell及其Python子进程已结束；若存活就监控它，不launch/resume。确定性代码错误、错误ID或内容冲突停止受影响链并上报，不能原样反复重启。

`resume`内部实际再launch：query checkpoint可复用，但现有脚本会覆盖run.log，且旧exit_code在新worker结束前可能仍在。因此恢复前保存小日志，恢复后以活进程和当前log判断，不把旧exit_code当新完成：

```bash
SNR_LAUNCHER=/path/to/CFA/_Data/results/snr_lds_20260925/a1/launcher_a
SNR_RECOVERY=$(mktemp -d /path/to/CFA/_Data/results/snr_lds_20260925/a1/recovery.XXXXXX)
for SNR_FILE in run.log pid exit_code; do
  if [ -f "$SNR_LAUNCHER/$SNR_FILE" ]; then
    cp -p "$SNR_LAUNCHER/$SNR_FILE" "$SNR_RECOVERY/$SNR_FILE"
  fi
done
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2
./tools/run_snr_lds_a.sh resume
```

不删输出、不修改只读快照、不改manifest绕过身份检查；不恢复历史BA缓存。resume后同样报告实际PID和新进度。

## 5. 交付与验收边界

根：`/path/to/CFA/_Data/results/snr_lds_20260925/a1/`。

- `inputs/benchmark_manifest.json`：实际来源、ID、表示与不适用项。
- `panels/`：逐query原子checkpoint、per_query/summary/predictions/fits/completion；失败原因与缺项明示。
- `tables/summary.{json,csv}`：四阈值Full/SNR、有效数、seed SD及query CI。
- `edel/edel_selection.json`：全局候选、共同有效benchmark IDs、完整50query两预算真实效用。
- `figures/`、`verify_a.json`、`launcher_a/{pid,run.log,exit_code}`。
- 回执：`/path/to/BA-LDS/Reports/snr_lds_2026-09-25/RUN_JINXU3_SNR_LDS_a1.md`，含实际命令、时间、PID、范围、缺格、偏差；按现有流程追加jinxu3台账。

verify命令会核对query源数据、mask和聚合代数，并确认表/E-DEL/固定图文件存在；**不把文件存在检查说成全部科学验收**。Executor核对产物数量与状态，Experimenter负责表格/选择/配对统计的最终验收。非DAS逐query Full应保持同源值；DAS用t²核对而非旧t/t；有效空/常数为0，失败为NA。E-DEL CI跨零或LDS变差照常交付，不作为失败。

`produced/verified`不等于Paper已采用。Executor不改论文、全局实验状态或旧科学结果。A完成后通知Experimenter，即使B仍待修正也不得扣住A交付。

## 6. B 的实际阻塞：不是等一句放行

本次只读检查发现r2的 `tools/run_snr_lds_b.sh` 尚不能接主库完整R16：

1. 写死的 `/path/to/CFA/_Data/results/repeatability_r16` 不存在；历史package/repeats分布在e3c原目录和narrative的R-A/B/C回传目录，不能用只有前4次的目录冒充完整16次。
2. 写死的 `Reports/evidence/r16_pilots.json` 是 `records[]` 来源报告，`summarize()`却要求 `{method:{track:npy_path}}`，且NPY应为已按原16 IDs对齐的pilot矩阵。不能直接传入。
3. 新代码 `prediction_variance` 当前算的是各subset预测方差再取均值，不是任务B要求的 `Var_r(P[r,m,q]-P[r,n,q])`。两者不同，需要补预测差方差字段及普通测试，不能改名验收。

这些只影响B；详见 `CODER_FOLLOWUP_B_RUNTIME_V1.md`。Coder提供修正后的独立B命令包后再更新B执行依据，不改A正在使用的r2快照。未发布B修正版前本书不授权运行r2 B，不新增任何GPU任务。
