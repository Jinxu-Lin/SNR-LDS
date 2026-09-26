# 迁移方案的本轮核对

执行日期：2026-09-22。这里只核对迁移规则、实际文件元数据和新运行 JSON 的转换，未复制二进制、未生成实验结果、未计算 hash。

- Python 脚本通过语法解析。
- `inventory` 成功扫描当前本地源树；逐文件目标路径唯一。
- 当前默认 `analysis,regeneration` 为 **16,408 文件 / 180,010,276,460 字节 / 167.648 GiB**。
- 实际 `plan --source /path/to/CFA --target /path/to/BA-LDS` 返回 16,408 项 `would_copy`，没有源变更或目标冲突。
- R16 清单恰有 **64 个 scores.npy**：复用的 R0–R3 加新 R4–R15，覆盖两方法两轨。旧 R4 独立分析与大缓存不在该集合。
- 当前 analysis 清单不含 W-A/W-B 区域采样产物；未选入整批 V3 的逐查询结果 payload。
- DAS 可选拟合缓存精确为 1,200 文件，未选择所有方法的 BA fit cache。
- 对实际 bench/edel JSON 进行内存转换并截获写入函数，核对 **190 条方法记录**都有明确 `score_space`，其中 **13 条 DAS 配置**均为 `das-presquare` 且引用原 `scores/das_T100/...` 的 t。BA 的 mask/response/score 路径全部相对数据根；旧有歧义字段已经从新副本内容移除。此项未产生 runtime JSON 文件。
- 源 `manifest.db` 当前有 2,643 行，417 个登记路径本地缺失；没有据此删文件或伪补登记。

可复查的元数据命令：

```bash
cd /path/to/BA-LDS
python Reports/migration/migrate_artifacts.py plan --source /path/to/CFA --target /path/to/BA-LDS
python Reports/migration/migrate_artifacts.py relocate-json --source /path/to/CFA --target /path/to/BA-LDS
```

这些核对不能证明历史论文数字符合新的 DAS 聚合语义。正确的重聚合、LDS 和 E-DEL 选择仍是独立待完成工作；迁移文件完整性也必须在未来实际复制之后再次核对。

追加范围修正：现役 M3 的 C2 ekfac_mean/ekfac_msl2 三种子双轨共 12 份原始分数已选择；仅为机制诊断，不加入主表。AB2 运行 manifest 转换明确设置 n_subsets=32。DDPM 保留不同分数/response 的原始列 ID 映射。

原始输入闭包追加核对：DAS 的两份索引分别有5,000/1,000行，顺序保留；两份小型模型配置同时保留 archive 根布局。curvature 加载已选择的 self-contained checkpoint，无需原始权重副本。ArtBench 清单保留两个固定风格各5,000训练/1,000测试原图，满足排序后固定抽样的候选池要求。
