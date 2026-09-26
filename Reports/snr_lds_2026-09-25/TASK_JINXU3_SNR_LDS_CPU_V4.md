# jinxu3 SNR-LDS V4：只收尾，不重新拟合

task_id: SNR_LDS_CPU_JINXU3_20260925；revision: 4；attempt_id: j3-snr-a4。  
Executor: jinxu3；状态：ready；本版本替代V3后续执行依据。B仍不运行。

## 已修复与复用范围

代码快照：`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4`。
Python沿用 `/path/to/CFA-envs/e3c-torch260/bin/python`。

- 全失败SNR方法仍按SNR schema汇总，数值保留NA，不能回退ba_lds字段。
- Full query CI与SNR使用相同有效查询；既有Full/SNR逐query数值不变。
- verifier按生产端float64后平方、逐subset点积次序重放；仍为rtol=0、atol=1e-12，不放宽阈值或容差。
- 用 `nohup setsid` 脱离启动会话，避免再次依赖执行者手动加setsid。

源 `a3/inputs/benchmark_manifest.json` 和 `a3/panels/` **只读**；保留243格、24,300条已完成记录及全部历史日志。输出仅写 `a4/`。启动后不执行prepare、batch、KDE拟合、模型、特征、评分或GPU操作。

5个缺格是cifar2_das这个DDPM平台的TracInCP/GAS双轨、Journey gen；不是das_native_sq方法缺失。当前源清单没有这些评分，本次保留missing，不改模型身份、不伪造、不为填格启动GPU。Journey val仍n/a。完成收尾不等于248格全齐。

## 启动

本机确认无a3/a4同任务worker存活，CPU/磁盘正常即可执行，不做冒烟、不重新核对整库。单worker、4线程、CUDA隐藏；不要求GPU空闲。

```bash
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh launch
```

**必须带 `SNR_STAGE=closeout`**。默认full模式保留给历史入口兼容，不用于本次修复。脚本依次：汇总表→E-DEL→固定图→全部已有query的只读source-algebra验证。没有任何重新拟合步骤，不等待B修复。

## 状态与验收

```bash
tail -n 40 /path/to/CFA/_Data/results/snr_lds_20260925/a4/launcher_a/run.log
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh status
```

status里的243/248是源面板完成数，不是后处理完成信号；实际后处理以当前worker、log及本次exit_code为准。PID文件是shell身份，记录当前Python子进程PID。

全部结束后：

```bash
cat /path/to/CFA/_Data/results/snr_lds_20260925/a4/launcher_a/exit_code
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh verify
```

交付a4内 `tables/summary.{json,csv}`、`edel/edel_selection.json`、`figures/`、`verify_a.json`和`launcher_a/`；回执 `RUN_JINXU3_SNR_LDS_a4.md` 链接a3源和本次输出，说明缺5格及拟合失败数。verify检查聚合及文件存在，科学统计/论文采用仍由Experimenter/Author分别处理，不把全失败方法填0或删掉。

## 恢复

确定性错误不原样重试。已知临时退出后确认旧shell和子进程结束，允许一次恢复；先保存日志，因为launch会覆盖run.log：

```bash
SNR_LOG=/path/to/CFA/_Data/results/snr_lds_20260925/a4/launcher_a
SNR_BACKUP=$(mktemp -d /path/to/CFA/_Data/results/snr_lds_20260925/a4/recovery.XXXXXX)
for SNR_FILE in run.log pid exit_code attempt_id; do
  if [ -f "$SNR_LOG/$SNR_FILE" ]; then cp -p "$SNR_LOG/$SNR_FILE" "$SNR_BACKUP/$SNR_FILE"; fi
done
cd /path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4
SNR_STAGE=closeout ./tools/run_snr_lds_a.sh resume
```

恢复只重新执行后处理，不触碰a3。旧exit_code在新worker结束前可能仍在，不据此宣告成功。无需搬运或再次请求启动许可。
