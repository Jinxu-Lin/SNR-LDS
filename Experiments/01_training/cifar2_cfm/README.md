# CIFAR-2 CFM：主模型三个 seed

本档案对应后续机制分析及 CIFAR-2 FM 评价的三个完整训练集模型。
训练 seed 为 **42 / 123 / 456**；不是三份不同源代码，而是相同实现和配方的三个随机初始化。

## 代码、配方与执行

- 数据：CIFAR-2，automobile / horse 各 2,500，32×32；固定数据抽样 seed 42。
- 模型：无条件 CFM U-Net，base channels 128。
- 训练：200 epochs，batch 128，AdamW，学习率 1e-4，weight decay 1e-6，10% warmup，cosine schedule，dropout 0.1，随机水平翻转。
- 训练总步数 7,812；中间检查点步数 1,953 / 3,906 / 5,859，另保留 final。
- [config.json](config.json) 显式冻结本实验关键配置；[run.py](run.py) 为当前 CLI 重建的复现入口。
- 训练实现：[training.py](../../../Codes/src/balds/workflows/training.py)、[模型训练](../../../Codes/src/balds/models/train.py)；其他默认设置见 [defaults.yaml](../../../Codes/src/balds/conf/defaults.yaml)。

```bash
# 默认只显示三条命令，不训练、不写入既有产物。
python Experiments/01_training/cifar2_cfm/run.py --data-root /path/to/new-data
# 确认输入和资源后执行全部三个 seed；也可用 --seed 42 单独执行。
python Experiments/01_training/cifar2_cfm/run.py --data-root /path/to/new-data --execute
# 训练后生成每模型 100 个查询，Euler 100 steps。
python Experiments/01_training/cifar2_cfm/run.py --data-root /path/to/new-data --stage generate --execute
```

训练不使用 `--force`；已有 final 检查点时拒绝执行，防止将跳过训练误报为新运行。
执行记录写入本目录忽略的 `runs/`，记录命令、时间、Git 版本、依赖版本和 stdout/stderr。
生成随机 seed 为 123 + 模型 seed，即 165 / 246 / 579；详见共用流程。

## 已有运行证据（不是本次重新训练）

[evidence/artifacts.json](evidence/artifacts.json) 登记三个 seed 的 final、中间检查点及 loss history 的数据相对路径、大小、SHA-256。
每个 seed 子目录保存原始 `loss_history.json` 的副本。检查点标量元数据直接从现存文件读取，未复制模型权重到 Git。

| Seed | Final 检查点步数 | 最后一次 loss 日志步数 | 日志 loss |
|---|---:|---:|---:|
| 42 | 7,812 | 7,000 | 0.184040 |
| 123 | 7,812 | 7,000 | 0.205463 |
| 456 | 7,812 | 7,000 | 0.193863 |

这里是训练过程中的日志值，不是 final checkpoint 的验证损失，也不是论文 query loss。

这些文件确认现有产物身份及训练步数，但无法单独证明原始优化器配置或训练时的软件环境。
原始训练命令、stdout 和训练时 Git commit 未在本档案中核实；上面的命令由当前实现重建，不冒称原始日志。
`record_artifacts.py` 可对另一个数据根生成新的检查清单；默认只输出 JSON，避免覆盖已有证据。
