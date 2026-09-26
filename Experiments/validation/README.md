# 验证证据与已知缺项

本次重组只整理代码、报告和小型证据，没有重新训练模型、采样梯度或生成图像。
当前测试结果见 `checks.json`。`public-wheel-environment.json` 是之前 CPU 安装验证的环境记录，
不是原始 GPU 训练环境。

2026-09-26 本次整理验证：54 个 Markdown 链接、6 条训练/生成命令解析及15个训练产物登记均通过。
`PYTHONPATH=Codes/src python -m pytest Codes/tests -q`：**329 passed**，90.27秒。
此结果是当前代码的测试，不是一次新的 GPU 实验。可用下面的命令重新检查档案：

```bash
PYTHONPATH=Codes/src python Experiments/validation/check_dossiers.py
```

保留的历史核验按实验存放：

- [SNR 数组核验](../05_snr_lds/evidence/benchmark_verify.json)：24,300 个 query 记录；含缺少输入的明确清单。
- [SNR 固定筛选控制](../05_snr_lds/evidence/figure3_controls.json)。
- [head tie 核验](../04_reweighting/evidence/head_tie_audit.json) 及曲线比较。
- [删除图像统计](../06_deletion/evidence/deletion_visual.json)。
- [检索表格核验](../07_source_retrieval/evidence/retrieval-verification.json)：572 个数值；不是重训证明。
- [R16 小型统计](../03_sampling_noise/evidence/)：原接受统计，不是重新生成的 64 份分数矩阵。

上述 JSON 保留历史数字和图表编号，以便追溯；仅对个人路径和工作区名称做脱敏。
当前图表编号以实验总索引为准。已移出的机器任务书、旧实验、源码副本及排版备份不再参与执行。
当前源码配置 `core_provenance.json` 内的旧 `Reports/...` 字符串仅记录历史来源，不是可执行依赖。
