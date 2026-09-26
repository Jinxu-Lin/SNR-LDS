# BA-LDS 论文整理交付

本目录记录 2026-09-22 对当前论文、代码和实验依赖的整理。CFA 是只读来源；本次没有迁移实验产物，也没有启动论文实验。

建议依次阅读：

1. [交付与验证报告](RELEASE_REPORT.md)：目录、架构、已完成工作、验证范围和待完成事项。
2. [实验总索引](EXPERIMENT_INDEX.md)：当前论文有效图表与实验的一一对应。
3. [共用生产管线](experiments/00_SHARED_PIPELINE.md)：训练、query、特征、曲率、子集模型、测损和评价。
4. [代码安装说明](../Codes/README.md)与[架构说明](../Codes/ARCHITECTURE.md)。
5. [产物迁移报告](migration/MIGRATION_REPORT.md)与[逐文件新旧地址](migration/files.tsv)：CFA 实验结束后的搬运依据。

实验报告使用相对数据根及可配置命令。`evidence/` 保留论文引用、协议属性和验收来源；历史版本与原路径是溯源信息，运行时不依赖那些服务器位置。

**DAS 的公开版定义是 fit(t)、aggregate(t²)。** 历史 V3 实际执行 fit(t)、aggregate(t)；相关 LDS、E-DEL 选法与表中数字需要按论文定义重新计算和验收。原文及历史证据保持原样，此次整理不替换科学结果。
