# E-SOURCE 曲率评分入口交付

交付日期：2026-09-22。新增 [source_scoring.py](../../Codes/src/balds/workflows/source_scoring.py)、[score_retrieval.py](../../Codes/tools/score_retrieval.py) 和 [CPU 测试](../../Codes/tests/test_source_scoring.py)。既有 CFA 源码、工件和运行目录均未修改，也未执行模型、迁移二进制或启动远程任务。

## 来源与数值契约

原始实现来自 CFA 工作树 `e5-j3-F0-20260920` 的 `Codes/tools/e5_common200.py`、`Codes/cfa/app/inject_subset.py`、`Codes/cfa/app/curvature.py` 和 `Codes/cfa/store/isolated.py`。公开入口移除了私有远程启动、主机、设备分配及绝对目录配置，复用公开包中的曲率求解、训练行随机数、分块存储与恢复函数；原始 query 子集映射在独立工作流中完成。

本次直接只读验证了以下两个既有 `identity.json`，新实现的 `scientific_identity(...)` 输出与它们的**全部字段及数值完全一致**：

- `_Data/results/e5_common200_20260920/fmas_raw/identity/identity.json`
- `_Data/results/e5_common200_20260920/incoming/xc2_I0/ekfac_if/identity/identity.json`

可公开审阅的原始属性快照见 [retrieval_common200_properties.json](../evidence/retrieval_common200_properties.json)。两者实际配置均为 **MC250/250**；100 是 query chunk，不能把 MC 改成 100 后声称重现这些结果。共同配置包括 fit/eig epochs 125/125、grad chunk 125、row chunk 1000、bfloat16 query coordinates、训练侧 train mode/hflip，以及原始 100 validation + 200 test 的 300 个全局位置。FMAS 为 stratified-antithetic/blockshrink，IF 为 iid/global，阻尼网格保持原值。

原有 query 标识为 `fe3004f1dd5cc10321d51e864c78479f4a101230ac614f108103ca6d4f2baf95`，直接读取已有字段，未计算新哈希。Query 梯度随机数使用原始 500-query 文件中的位置，而不是重排后的 0…299；训练行仍以原始行号生成随机数。分块额外校验 original query IDs 与 grad chunk，避免将不同 iid 抽样配置混用。

## 执行与产物契约

先从 `Codes/` 安装项目及训练依赖。以下命令在项目根目录运行；输入根与输出根由使用者设置。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
OUTPUT="$BALDS_DATA_ROOT/results/paper/source_cfm_blocks"
SELECTION=results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json

python Codes/tools/score_retrieval.py prepare \
  --input-data-root "$BALDS_DATA_ROOT" --output-root "$OUTPUT" \
  --selection "$SELECTION" --method fmas_raw

python Codes/tools/score_retrieval.py run \
  --input-data-root "$BALDS_DATA_ROOT" --output-root "$OUTPUT" \
  --selection "$SELECTION" --method fmas_raw \
  --device cuda:0 --query-chunk 100 --rank 0 --world 1 \
  --row-rank 0 --row-world 1 --cpu-threads 2

python Codes/tools/score_retrieval.py status \
  --input-data-root "$BALDS_DATA_ROOT" --output-root "$OUTPUT" \
  --selection "$SELECTION" --method fmas_raw
```

将 `--method` 改成 `ekfac_if` 执行另一方法；两者可以共用输出根，方法的身份和评分块地址彼此独立。准备阶段检查现有 checkpoint、曲率 factors、query、injection metadata、selection、CIFAR10/100 cache，不训练或复制因素。重建实验须先使用主工作流的训练与 `ekfac fit` 入口生成这些输入，或迁入验收过的对应输入。

`run` 只生成评分块、身份文件和执行记录。`status` 只读身份和 block metadata，给出覆盖缺口；`blocks_complete` 仅表示 metadata/coverage 齐全。最终矩阵数值、有限性、validation-only lambda 选择及 test 指标由 [assemble_retrieval.py](../../Codes/tools/assemble_retrieval.py) 检查和发布。把上述 `$OUTPUT` 传给它的 `--sources`，并传入原有 selection 和 comparison manifest。多个工作目录的完成块可在装配阶段合并；各目录必须使用完全一致的科学身份。

现有完成块可在相同身份下复用，已有原始 provenance 保留；修改 query chunk 会改变分块身份，使用新输出目录。`--code-version` 可记录执行版本，但不代替科学身份校验。公开入口不修改已有 lambda/native score 工件，也不会将块覆盖完成误写成最终检索结果。

## 已完成验证与边界

在 Python 3.12 的现有 CPU 环境，运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Codes/src python -m pytest -q \
  Codes/tests/test_source_scoring.py Codes/tests/test_source_assembly.py
```

结果：**8 passed**。覆盖了原始 query RNG key、行号和分片参数传递、MC/grad chunk、训练与查询侧 mode/hflip、阻尼矩阵合成、原子块恢复跳过、错误身份拒绝、缺少输入、无身份旧块拒绝、CLI 到运行工作流，以及 CPU 装配接口。额外以实际 query/selection 文件构造新身份，并对上述两个真实验收 identity 逐字段断言相等；两个断言均通过。

仅使用确定性合成 CPU scorer 验证流转，没有真实模型评分、GPU 测试或论文结果重算。最终全库检查与包构建由集成任务统一执行。兼容性依据是保存的科学身份、原始索引与复用的数值内核；完成数值复现仍需在输入迁移后执行正式评分与装配验收。
