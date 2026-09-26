# Direct DDPM recovery — jinxu3 a1

研究者2026-09-25明确授权本会话直接处理代码并执行收尾。本轮兼任Experimenter/Coder/Executor。
接管原xuchang3两条lost lane剩余工作；原worker/supervisor均不存活，没有执行恢复命令，不再等原卡释放。xuchang3当前其他任务不动。

资源：jinxu3 cuda0（启动前1MiB、0%），CPU2线程。无新增训练、生成或科学参数变动。
执行代码 `/tmp/cfa-ddpm-tracin-gas-j3-snr-v1` @ `19c31140c300d344f827f15e04d5ff460ad9f6ac`，解释器 `/path/to/CFA-envs/e3c-torch260/bin/python`（torch2.6.0+cu124）；SNR r4。
变更仅为接管调用封装：将已完成step2000/6000六矩阵与meta登记到主库，既有features自动复用它们并补4000/8000，随后既有score与SNR串行执行。CPU入口增加run_local跳过不再适用的incoming全量ingest，科学核心不变。

## 实际准备命令

```bash
git -C /path/to/CFA worktree add --detach /tmp/cfa-ddpm-tracin-gas-j3-snr-v1 19c31140c300d344f827f15e04d5ff460ad9f6ac
mkdir -p /path/to/CFA/_Data/results/ddpm_tracin_gas_20260924/recovered_xc3
rsync -a xuchang3:/path/to/CFA/_Data/featurize/trak_T100/cifar2_das_retrain_20260924/ddpm/seed_42/step_2000 xuchang3:/path/to/CFA/_Data/featurize/trak_T100/cifar2_das_retrain_20260924/ddpm/seed_42/step_6000 /path/to/CFA/_Data/results/ddpm_tracin_gas_20260924/recovered_xc3/
```

## 启动与状态

```bash
/path/to/CFA-envs/e3c-torch260/bin/python /path/to/BA-LDS/Reports/snr_lds_2026-09-25/run_ddpm_recovery_j3.py launch
/path/to/CFA-envs/e3c-torch260/bin/python /path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools/e3c_launch.py status --output-root /path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/recovery_j3_a1/supervised
```

输出：既有标准featurize和score namespace；SNR结果 `/path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/a1`。监督器保存实际PID、阶段日志和退出码。未完成矩阵只能从头，已完成矩阵复用；正式首单元更新ETA。仅补TracInCP/GAS四格，不冒充Journey已补齐。
状态：已于2026-09-25 16:18:09 AEST启动。监督器3516072，串行worker3516074，实际GPU特征子进程3516458。
启动记录：`/path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/recovery_j3_a1/supervised/launcher/repeat_0_2b4f495cee63.json`；同名`.log`记录全过程。
已看到两次RESTORED（step2000/6000共6份特征＋2份meta登记主库）、step2000三项REUSE_FEATURE，以及START_FEATURE step4000 split=train。16:18:40 GPU约1258MiB，计算子进程存活；正式50行吞吐尚待日志。
调用封装普通测试已通过：恢复只执行一次，GPU features使用CUDA_VISIBLE_DEVICES=0，后续run_local CPU使用空CUDA可见列表；这是命令路由测试，不是正式评分已完成。核心featurizer、评分器、r4 SNR未改。
研究者本次直接执行授权覆盖迁移至空闲jinxu3 cuda0；原xuchang3 a2恢复授权对本任务暂停，避免双机重复。后续CPU阶段已串入同一监督链，不需要再次分发。
