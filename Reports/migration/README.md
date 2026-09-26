# 产物迁移说明

本目录记录从 CFA 到 BA-LDS 的**待执行迁移方案**。本次只读取源文件、数据库和文件大小/时间，没有复制实验产物、连接其他服务器、运行实验或计算新哈希。`BA-LDS/_Data/` 暂时只有说明文件。

入口为 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)。机器可读选择规则在 [selection.json](selection.json)，盘点快照在 [inventory.json](inventory.json)，逐文件映射在 [files.tsv](files.tsv)。[files.txt](files.txt) 是默认 `analysis,regeneration` 配置的源文件清单；实际复制必须使用 TSV/JSON 中的目标映射，不能直接对根目录无差别 rsync。

```bash
cd /path/to/BA-LDS
python Reports/migration/migrate_artifacts.py
```

默认命令只汇总保存的清单，不访问源机器、不传输数据。详细的未来刷新、计划、复制、核对和路径重定位命令见迁移报告。

**DAS 的最终要求：在有符号平方前分数 `t` 上拟合和筛选，保留坐标使用 `t²` 聚合。** CFA 的 V3 结果在拟合和聚合两处都使用了 `t`，因此此批旧结果不属于该要求下的最终结果。清单只将其小型配置/汇总作为历史证据，另将 1,200 个 DAS 拟合缓存列为可选复用输入；不默认搬运整批 BA 缓存。必须重新聚合、重算 Full/BA-LDS，并重做 E-DEL 方法选择。

导入 DDPM 的最小原始身份闭包已显式保留：两份有序 split 索引和两份模型/scheduler JSON，共 4 文件、19,218 字节。其余原始 DAS archive 继续排除；曲率使用已选入的自含 imported checkpoint。ArtBench 保留两种当前风格各完整 5,000 train/1,000 test 图像，确保固定抽样与行序不变。
