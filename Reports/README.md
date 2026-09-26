# SNR-LDS 论文复现与交付记录

当前项目统一命名为 **SNR-LDS**，安装包名为 `snr-lds`；当前入口是 **2026-09-26 定稿对齐版**。方法与实验以本地 `Paper/` 为依据，`Codes/` 已迭代为 0.2.0，所选实验产物位于本地 `_Data/`。论文和数据不随 Git 仓库发布。

FMAS 是归因方法名，不是项目名。`balds` 命令、导入路径及 `BALDS_*` 环境变量为兼容现有脚本保留。历史报告中的 CFA/BA-LDS、旧机器路径和协议标识仅用于来源追溯，不代表当前项目名称；历史数值和原始回执不作重新标注。

完整命名规则与安装兼容说明见 [PROJECT_NAMING.md](PROJECT_NAMING.md)。

匿名发布说明：报告中的个人主目录已替换为 `/path/to`，同名主机和其他目录标识也已匿名化。其后的项目、实验和文件相对路径保留；这些绝对路径是示例占位符，运行历史命令或脚本前须替换为实际位置。此次处理不改写 Git 历史，也不代表所有作者身份线索均已移除。

1. [本次交付与验证](final_alignment_2026-09-26/RELEASE_REPORT.md)：方法、代码、数值复核及仍需说明的差异。
2. [最终论文范围核对](final_alignment_2026-09-26/PAPER_SCOPE_AUDIT.md)与[实验索引](EXPERIMENT_INDEX.md)：有效图表及执行报告。
3. [实际产物迁移报告](final_alignment_2026-09-26/migration/FINAL_MIGRATION_REPORT.md)：已复制文件、可选大型产物与缺项。
4. [安装和运行](../Codes/README.md)、[架构](../Codes/ARCHITECTURE.md)、[数据说明](../_Data/README.md)。
5. [共用生产管线](experiments/00_SHARED_PIPELINE.md)：训练、query、特征、曲率、子集测损与评价。

最终 SNR 定义为零均值背景拟合、双侧严格阈值筛选；DAS 始终拟合平方前 t、聚合原生 t²。删除实验使用三 seed 全局方法选择，检索使用每模型 reviewed500 的 val100/test400，R16 使用 VarRatio。

`RELEASE_REPORT.md`、`migration/` 及其他旧日期回执是早期版本的历史记录，不代表本次完成状态。历史代码与被替换报告备份在 `final_alignment_2026-09-26/before/`；其中服务器名和旧绝对地址只用于溯源，不是当前代码运行依赖。
