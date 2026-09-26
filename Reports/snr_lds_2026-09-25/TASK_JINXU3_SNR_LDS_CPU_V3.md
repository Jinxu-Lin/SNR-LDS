# jinxu3 SNR-LDS：A 修复后正式运行 V3

> 后续：a3已完成243格后在汇总/verify阻塞，原记录保留。使用 [V4后处理任务](TASK_JINXU3_SNR_LDS_CPU_V4.md)，只读a3，输出a4；不要再按本版launch重跑整链。

task_id: SNR_LDS_CPU_JINXU3_20260925；revision: 3；attempt_id: j3-snr-a3。  
owner: jinxu3 Executor；acceptance: Experimenter；状态：**A ready**。本书是修复后的执行依据，不再等待另一次放行。本次仅发布命令，尚未启动正式实验。

## 修复接受与范围

已核对 `CODER_FIX_A_DATASET_REGISTRATION_RECEIPT.md` 和r2→r3实际差量：只补prepare的已有DAS注册模块导入，以及a3输出/attempt路径；未改loader、分数、ID选法或SNR公式。Coder普通测试327 passed、1 skipped；本次读回其临时真实来源manifest：16面板，DDPM val IDs确为0..94,96..100，响应为原DDPM val GT。

这是**代码修复及输入解析验证完成**，不是SNR实验完成；验收时a3目录尚不存在。不要复用旧a1目录或重启r2。旧a1日志和recovery证据保留；第一次空日志退出原因仍未知。

- 代码：`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3`，只读快照，无Git commit。
- Python：`/path/to/CFA-envs/e3c-torch260/bin/python`。
- 输入：`/path/to/CFA/_Data`；本机已有，无传输或新下载。
- 输出：`/path/to/CFA/_Data/results/snr_lds_20260925/a3/`。
- CPU单worker、4线程，CUDA隐藏；不停止或干扰其他任务。不重复代码测试、不做冒烟、不做全库hash。
- 范围沿用V2 §2：四平台16面板的在役方法，zeta1/2/3/4共用拟合、主结果3；主表/coverage/固定图/全局选法E-DEL。DAS t拟合/t²聚合；不新增训练、梯度、评分或GPU工作。
- B不属于本次启动。r3仅修复A，不能据此运行B；B修正仍按 `CODER_FOLLOWUP_B_RUNTIME_V1.md`。

## 1. 现场检查一次

```bash
hostname
free -h
df -h /path/to/CFA/_Data
ps -eo pid,ppid,lstart,args | rg '[r]un_snr_lds_a.sh|[b]alds.cli.ba batch'
```

应在anonymous-lab3，无同任务旧shell/Python子进程；若有则先核实，不重复启动。以约16 GiB可用RAM、10 GiB可用磁盘为规划余量。无需等GPU空闲。资源正常就进入正式启动，不等回执发布。

## 2. 正式启动

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3
./tools/run_snr_lds_a.sh launch
```

脚本自动nohup脱离会话，写真实shell PID、attempt及退出码。依次prepare（含AB2 signed-t必要CPU恢复）→SNR batch→tables→E-DEL→figures→verify。不要另行手工prepare，不使用Coder临时清单替代正式清单，不原地修改快照。

## 3. 监控

```bash
cat /path/to/CFA/_Data/results/snr_lds_20260925/a3/launcher_a/pid
tail -n 40 /path/to/CFA/_Data/results/snr_lds_20260925/a3/launcher_a/run.log
```

manifest生成后再运行status；prepare期间manifest未生成不等于失败。

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3
./tools/run_snr_lds_a.sh status
SNR_PID=$(< /path/to/CFA/_Data/results/snr_lds_20260925/a3/launcher_a/pid)
ps -p "$SNR_PID" -o pid,ppid,lstart,etime,args
ps --ppid "$SNR_PID" -o pid,ppid,lstart,etime,args
```

首次有效进度报告实际Python PID/阶段；首正式面板更新ETA，不另试跑。正常继续到A全部阶段结束，缺输入明确列出，不自行提特征或填0。

## 4. 完成与交付

```bash
cat /path/to/CFA/_Data/results/snr_lds_20260925/a3/launcher_a/exit_code
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3
./tools/run_snr_lds_a.sh verify
./tools/run_snr_lds_a.sh status
```

产物位于a3的inputs、panels、tables、edel、figures、verify_a.json和launcher_a。回执写 `Reports/snr_lds_2026-09-25/RUN_JINXU3_SNR_LDS_a3.md`，记录命令、耗时、PID、输出量、fit_failed/empty/missing/n/a、验证和偏差；追加原jinxu3台账。本机产物无需回传。

验收沿用V2 §5科学标准：源ID与GT一致，Full/SNR配对集合正确，DAS平方一次、严格双侧阈值，合法空/常数0而失败NA；四候选benchmark全局选法和原50query两预算效用对应。verify的文件存在检查不是最终科学验收；Experimenter复核后登记。A先交付，不能等B或声称B已完成。

## 5. 有界恢复

确定性代码/数据错误不原样重试。已识别临时退出且旧shell/Python均结束时，允许一次同a3产物根的resume，回执另记恢复次数与新PID。执行前保存现有小日志（脚本会覆盖run.log；旧exit_code在新worker结束前不代表新完成）：

```bash
SNR_LAUNCHER=/path/to/CFA/_Data/results/snr_lds_20260925/a3/launcher_a
SNR_RECOVERY=$(mktemp -d /path/to/CFA/_Data/results/snr_lds_20260925/a3/recovery.XXXXXX)
for SNR_FILE in run.log pid exit_code attempt_id; do
  if [ -f "$SNR_LAUNCHER/$SNR_FILE" ]; then
    cp -p "$SNR_LAUNCHER/$SNR_FILE" "$SNR_RECOVERY/$SNR_FILE"
  fi
done
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3
./tools/run_snr_lds_a.sh resume
```

只复用同身份已完成的新规则query checkpoint，不删旧证据、不改参数绕过失败。不得执行r2/a1的resume。
