# 最终论文的数据整理

本目录接续 2026-09-22 的迁移调查，以当前扁平结构的 `Paper/` 为准。实际范围见 [论文引用图](../paper_include_graph.json) 和 [论文范围审计](../PAPER_SCOPE_AUDIT.md)。旧迁移报告仍保留历史，但不再决定当前默认数据集。

默认清单包含最终实验结果及其必要输入、模型、特征和曲率；复制是独立文件复制，不是硬链接、符号链接或移动。CFA 保持只读，既有 BA-LDS 文件不覆盖。文件内容不计算新哈希，也不重新执行实验。

- [selection.json](selection.json)：按最终实验划分的选择规则、退休项与未完成项。
- [inventory.json](inventory.json)、[files.tsv](files.tsv)：实际文件级来源、目标、大小、修改时间和用途。
- [files.txt](files.txt)：默认 `analysis,regeneration` 文件清单。
- [benchmark_dependency_check.json](benchmark_dependency_check.json)：16 个 SNR 面板的原始 query IDs、分数空间、实际缺失项与输入覆盖。
- [COPY_RECEIPT.json](COPY_RECEIPT.json)：复制结束后的文件数量、体积和逐文件元数据核对；在复制完成前不存在。
- `files.optional_retraining.txt`、`files.optional_pretrained.txt`：未默认复制的再训练权重与外部预训练模型。

`build_selection.py` 从既有验收 manifest 建立本次规则；`migrate.py` 复用旧工具的文件清单、只读源索引和复制逻辑。默认动作是计划，只有 `copy --execute` 或 `registry --execute` 会改变目标数据目录。目标剩余空间不足 30 GiB 时拒绝复制。

本轮实际命令的源项目根为 `/path/to/CFA`，目标根为 `/path/to/BA-LDS`。工具通过参数接收这些路径；换机器时由调用者提供当地的源、目标目录。

```bash
python Reports/final_alignment_2026-09-26/migration/build_selection.py --source /path/to/CFA
python Reports/final_alignment_2026-09-26/migration/migrate.py inventory --source /path/to/CFA
python Reports/final_alignment_2026-09-26/migration/migrate.py plan --source /path/to/CFA
python Reports/final_alignment_2026-09-26/migration/migrate.py copy --source /path/to/CFA --execute
python Reports/final_alignment_2026-09-26/migration/migrate.py verify --source /path/to/CFA
python Reports/final_alignment_2026-09-26/migration/migrate.py registry --source /path/to/CFA --execute
```

最后一步新建过滤后的 `manifest.db`，或只追加源库中既有、已复制且大小匹配的缺失条目；既有目标身份冲突时停止，不覆盖。未登记产物继续由明确路径清单寻址；不凭空创建旧 hash、身份或验收状态。历史 JSON 中的原路径保持原样，由公开代码的路径解析转换到当前数据根。

默认不复制可重新训练得到的约 115.9 GiB 子集／删除模型权重，也不重复复制约 45.5 GiB 的 SD3.5 预训练缓存。完整路径仍在可选清单中，源库保留。论文表图的复核使用已选入的原始 mask、实测响应、删除效用和图像结果；从模型重新测量的入口及成本应与这些已存结果的复核区分。
