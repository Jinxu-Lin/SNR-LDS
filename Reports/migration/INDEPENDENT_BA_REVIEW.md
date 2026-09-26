# 新 BA 评价链的独立有限范围复核

日期：2026-09-22。复核者未修改新评价器代码；本次仅阅读 `Codes/src/balds/evaluation/background.py`、`Codes/src/balds/workflows/benchmark.py`、共用 LDS 计算与对应测试，并使用截获文件读写的内存例检查一个覆盖率问题。没有读取大规模分数重新评分，没有正式实验或产物迁移。

## 已核对的科学行为

- `evaluate_das(t, ...)` 明确调用 `evaluate(t*t, ..., fit_scores=t)`。背景拟合与双侧筛选在有符号 t 上完成，Full 和 BA 用原生 t²；保留分数不再乘 suppression 权重，也没有再次平方为 t⁴。
- 拟合失败的 BA-LDS 是 `None`；合法空集合/全零向量的预测是零，不把失败补零。汇总 Full 与 BA 使用同一组 BA 有效查询。
- 样本子集使用保留指示 mask，预测对删除行 `1-mask` 求和。Full/BA 共用响应和 mask。
- batch 按方法自己的 `query_ids` 以及响应的 `response_query_ids` 显式匹配，不假定投影/曲率矩阵的前 100 列具有相同身份。
- 四候选删除选择使用共同有效查询；精确并列候选取实测效用均值，同一个方法选择用于两个删除预算。

## 发现并已由实现者修复的问题

原 batch 在“请求 2 个查询、分数只含 1 个查询”的部分缺失场景中，逐行结果保留了 missing_query，但方法摘要的 n_total 继承 evaluator 的 1，缩小了覆盖率分母。

已向实现者报告。实现者随后将摘要拆为 `n_total`、`n_evaluated`、`n_missing_query`。在更新后的源码上独立运行了小型内存例：requested IDs 为 `[8,4]`，方法只有 `[8]`；结果为 n_total=2、n_evaluated=1、n_missing_query=1、complete=false，缺失行保留 ID4。所有加载/写出接口均被截获，没有创建输出目录。

该有限范围没有发现其他需要阻止交付的 DAS 聚合或缺失值处理问题。它不替代正式数据重聚合与逐平台验收，也不宣称已经验证全部训练/特征管线。

## 与迁移输入配套的修正

现役 M3 的 `ekfac_mean` 和 `ekfac_msl2` C2 三种子双轨分数保留为机制输入，不扩大主表方法集合。旧 AB2 panel 的 `n_subsets=64` 由 runtime 适配明确改为真实的 32；不继续依赖旧 reader 的 min-slicing。投影、曲率、响应的原始列 ID 保持不变。

AB2 DAS 的有符号平方前输入还须按实际同配置 λ=1 来源确认/恢复。旧全平台 manifest 是历史身份依据，不因完成路径重定位就自动获得正确的 t；默认转换列表没有擅自启用未确认的 AB2 评分输入。
