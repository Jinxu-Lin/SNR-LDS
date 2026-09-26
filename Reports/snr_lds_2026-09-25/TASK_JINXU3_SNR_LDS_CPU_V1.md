# jinxu3：SNR-LDS 全平台 CPU 重评

> 2026-09-25：本版已由 [V2执行书](TASK_JINXU3_SNR_LDS_CPU_V2.md) 替代。以下 planned／等待放行和Coder附录保留为历史。当前 A ready；B 有实际接入修正项，不能按本版附录直接运行。

task_id: SNR_LDS_CPU_JINXU3_20260925；revision: 1；owner: jinxu3 Executor；acceptance: Experimenter。

**状态：planned，等待配套 Coder A 实现；本文件现在不是可执行发布包。** 不要用现有 `balds batch` 的旧密度比规则启动。Coder 须在本文件后附实际命令/代码版本，将已实现部分标 ready；不另要求科研审批。B 可晚交，不阻塞 A。

## 1. 范围与资源

- anonymous-lab3，本机 CPU，单worker，4线程；CUDA_VISIBLE_DEVICES 为空，不占GPU，不停止在跑任务。
- 代码：`/path/to/BA-LDS/Codes` 的新 SNR 交付，正式运行用 Coder 提供的固定代码快照/真实提交，不边跑边编辑。
- 输入根：`/path/to/CFA/_Data`，只读。不把新项目尚空的 `_Data` 当已迁入主库。
- 输出根：`/path/to/CFA/_Data/results/snr_lds_20260925/`；大工件不放 Git，不覆盖旧 BA。
- 回执：`/path/to/BA-LDS/Reports/snr_lds_2026-09-25/RUN_JINXU3_SNR_LDS_a1.md`；本机台账按现有流程追加。
- 资源规划：可用RAM16 GiB、可用磁盘10 GiB作为串行评估预计余量；内存按一个方法/面板加载，禁止把全部C10方法同时转成float64常驻。正式首面板报告实际峰值/ETA，不做冒烟。A先预留半天CPU窗口，非保证完成时间；超过预计汇报真实阶段/剩余量，不因估时偏差终止有效工作。

## 2. 顺序

### A1 输入与主表

执行 Coder 已交付的 prepare 调用：汇集计划中各已完成源，生成本次显式 score/train/query/response-ID 清单及 missing_inputs，不迁移整库。若 AB2 DAS signed lambda1缺文件，用现成 CPU 恢复入口写到本次 inputs，不重提梯度。DDPM采用修正后的val原ID，不继承旧错位。

串行完成 C2 FM、C10 FM、AB2 SD3.5、C2 DDPM 的 gen/val；FM seeds42/123/456，AB2/DDPM seed42。每轨100queries，CIFAR64子集、AB2 32子集。覆盖当前实际已有的在役方法。2026-09-25 作者将 Parameter-weighted D-TRAK、AbU+、NDA 退出实验基线：即使已有评分也不接入，不计入主表、coverage、排名或缺件待办。Journey val=n/a，不是待办；其他在役方法缺评分明确记录，不为补齐自动启动GPU。

固定零均值拟合、101点、SNR严格>3；DAS t拟合/t²聚合。新拟合不读取旧BA fit参数/选择。每query一次拟合，然后复用到阈值1/2/3/4；主表默认3不变。完成一个平台就落盘，缺格不阻塞其余平台。

### A2 表格、图和删除效用

导出 Full/SNR 配对表、各阈值敏感性、coverage/失败/空集合/常数统计、原query逐条数据和固定q0/q1拟合图。

按新版论文，从C2 generation三seed benchmark共同有效集选一个全局候选，分别产生Full与SNR选择。直接读取已有四候选×50queries×2预算的native实测效用；全50queries给出k300/1000选择后效用与差值CI。不得重训、重新生成、换删除名单、按效用择优或改成每query选法。

### B 附录CPU分析（对应代码交付后）

原head/平方控制的SNR固定选择重评，以及现有FMAS/D-TRAK R16独立pilot校准、8/8交叉诊断，按代码任务B执行。使用原package IDs与完整重复，不能将新选择拟合在用于验证的repeat上。不得补 IF/DAS GPU repeats 或MC预算干预；缺口单列。A完成可以先交付，不等待B。

## 3. 命令包交付要求——由 Coder 在实现后填实，不交 Executor 临场开发

本版本不列伪造的 SNR CLI。Coder 必须交付并附录以下**真实完整命令**：

1. 固定代码/导入路径、已有Python环境；输入清单 prepare、AB2 DAS必要的CPU恢复。
2. detached launch A、status、resume、verify；launcher落盘真实PID、退出码与阶段日志。可复用已有supervisor或简单既有机制，不新增队列/平台。
3. 单独 launch B、status、resume、verify；B交付后自动接续的具体依据，不隐藏在A命令中。
4. 表/图/回执的完整输出位置，哪些输入missing、哪些阶段结束、剩余哪些可补入；不通过文件存在猜完成。

环境固定值：CUDA_VISIBLE_DEVICES=''；OMP_NUM_THREADS=4；MKL_NUM_THREADS=4；OPENBLAS_NUM_THREADS=4；NUMEXPR_NUM_THREADS=4；PYTHONUNBUFFERED=1。现有可用Python候选为 `/path/to/CFA-envs/e3c-torch260/bin/python`，由Coder针对新包确认并写实际路径；无需改变运行中的环境或下载模型。

命令包缺失是真实代码前置，不是要求Executor补一轮研究设计。已交付测试不重复跑；现场检查一次目标主机/资源/重复worker，准备输入后直接正式启动并报告首个有效进度，无冒烟、无启动前全库hash。

## 4. 恢复与验收

任务中断后排除同任务旧worker仍在运行，按已交付resume仅继续未完成新规则query/面板。旧BA缓存不作为resume命中；不能把部分输出标完成。确定性代码错误报告Coder，不原样循环重启；不更改阈值、科学参数或源ID来绕过错误。

验收内容：

- 每个已有输入组合均有逐query记录或明确失败原因；缺输入与不适用不算拟合失败。
- 原ID/GT一致；DDPM非连续val ID正确；分数与响应无任何重算/修改。
- 非DAS Full逐query与同源历史值一致；DAS按native t²核对，不强制复现旧t/t Full。
- 新fit是零中心二参数回归；保留数符合abs(x)/sigma_x>zeta；DAS两侧入选、平方一次。全零、合法空及常数为0，拟合失败为NA。
- 同方法Full/SNR配对有效集、全局选法四候选共同集、跨seed权重、共享val与2000query-bootstrap均可从输出复算。
- E-DEL全局选择和既有400条native效用对应；50query结果完整。任何CI跨零如实报告，不以提升作为验收条件。
- 输出含 source_manifest、逐query表、fit摘要、预测、主表/敏感性/图、E-DEL、missing/status清单、实际命令与退出码；B另给诊断表图及pilot/repeat对应关系。

完成后Executor报告 produced/verified；Experimenter再独立验收与登记，Author更新论文。不得声称目前已计算出SNR-LDS，也不自动改写Paper或旧实验结果。

## 5. Coder 交付附录（2026-09-25）

状态：A/B 代码均已实现并通过普通测试；**正式 launch gate 仍关闭，Coder 未启动任何重评**。

- 固定只读代码：`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2`
- 快照包：`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2.tar.gz`
- 快照包 SHA-256：`2034f7e3c35bbb08eedd0682a75471cb0d53096ef2a8480aa92d56138376fcfb`
- 快照逐文件清单聚合 SHA-256：`8e09f31ef405e164dc3848943926d21012493fc15008b65a5014e6857353b203`
- Python：`/path/to/CFA-envs/e3c-torch260/bin/python`
- 固定配置：`configs/snr_lds_j3.json`；其中 `formal_launch_authorized=false` 是本次交付状态记录，不是 Executor 的启动授权替代品。

### A：launch / status / resume / verify

研究者或 Experimenter 另行打开正式门后，Executor 使用固定快照中的串行封装：

```bash
CODE=/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2

# detached launch；写 launcher_a/{pid,run.log,exit_code}
"$CODE/tools/run_snr_lds_a.sh" launch

# 状态包含每个 method 的原子 query checkpoint 数、missing/pending/complete，
# 同时报告真实 worker 与退出码。
"$CODE/tools/run_snr_lds_a.sh" status

# 先确认旧 PID 不存活；同一命令仅命中身份完全相同的新规则 query checkpoint。
"$CODE/tools/run_snr_lds_a.sh" resume

# 从原 scores/masks 重放 D=1-K、严格阈值及普通/DAS聚合，并检查表、E-DEL、固定图。
"$CODE/tools/run_snr_lds_a.sh" verify
```

`run_snr_lds_a.sh` 固定 `CUDA_VISIBLE_DEVICES=''`、单 worker、各 BLAS/OpenMP 4 线程，并依次执行：显式来源清单与 AB2 lambda=1 signed-t 必要恢复、SNR batch、跨 seed 表、benchmark-level E-DEL、固定 C2 seed42 gen q0/q1 图、验收重放。AB2 恢复只写本次 `a1/inputs`；已有一致 manifest/恢复数组不会覆盖，冲突身份会失败。旧 BA fit/selection 从不作为 resume 命中。

A 输出根为 `/path/to/CFA/_Data/results/snr_lds_20260925/a1/`：

- `inputs/benchmark_manifest.json`：来源及原 train/query/response IDs；
- `panels/*/*/queries/query_*.npz`：单文件原子 checkpoint，含规则身份、fit 摘要、四阈值选择和预测；
- `panels/*/*/{summary.json,per_query.json,predictions.npz,fits/,completion.json}`；
- `tables/{summary.json,summary.csv}`：Full/SNR、seed SD、query bootstrap 与 ζ 敏感性；
- `edel/edel_selection.json`：三 seed benchmark 级选择和完整 50-query 两预算效用；
- `figures/`、`verify_a.json`、`launcher_a/`。

缺失评分写为 `missing_input`，Journey validation 写在 manifest/panel 的 `not_applicable`，数值拟合失败单独写 `fit_failed`；三者不会互相冒充。

### B：独立 launch / status / resume / verify

B 不隐藏在 A launcher 中，只有 A 表交付后才单独启动：

```bash
CODE=/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r2
"$CODE/tools/run_snr_lds_b.sh" launch
"$CODE/tools/run_snr_lds_b.sh" status
"$CODE/tools/run_snr_lds_b.sh" resume
"$CODE/tools/run_snr_lds_b.sh" verify
```

B 输出固定-mask head/平方控制到 `results/snr_lds_20260925/b1/controls/`；它只读既有 `results/repeatability_r16` package/repeats，并将独立 pilot SNR 诊断原子写入新的 `results/snr_lds_20260925/b1/repeatability/<method>/<track>/{snr_statistics.json,snr_coordinates.npz}`，不覆盖历史分析。`verify_snr_appendix.py` 检查规则、ζ=3 和四个 FMAS/D-TRAK × gen/val 的完整 8/8 双向逐 repeat 覆盖。B 不运行 repeat scorer，不补 IF/DAS repeats，不占 GPU。

### 代码验证回执

```text
PYTHONPATH=src /path/to/CFA-envs/e3c-torch260/bin/python -m pytest tests -q
326 passed, 1 skipped, 2 warnings in 68.60s

PYTHONPATH=src /path/to/CFA-envs/e3c-torch260/bin/python tools/check_layers.py
Dependency layers passed (92 modules).

bash -n tools/run_snr_lds_a.sh tools/run_snr_lds_b.sh
passed
```

唯一 skip 为既有可选环境条件；两条 warning 为 PyTorch scheduler 的既有弃用提示。以上仅为普通代码测试和静态 launcher 检查，不是正式实验或正式冒烟。
