# Coder 紧急修复：A prepare 遗漏 cifar2_das 注册

日期：2026-09-25；范围：现有 SNR A 入口修复，不改变实验。优先于 B 的接入修正，A可独立交付。

## 已确定原因

来源：`RUN_JINXU3_SNR_LDS_a1.md` 及主库 `results/snr_lds_20260925/a1/launcher_a/run.log`。

`tools/prepare_benchmarks.py` 调用 `workflows.common._load_ds(cfg,'cifar2_das')`，但此独立入口未导入负责注册的 `workflows.das_import`。注册函数**已经存在**，并非缺少数据集实现或主库数据：

- `src/balds/workflows/das_import.py::_load_cifar2_das` 从原 DAS archive 读取训练/验证索引，保持原行序；
- 同模块已有 `DATASETS.add(DATASET, _load_cifar2_das)`；
- 原 container 通过导入此模块完成注册，新 prepare 入口绕过了 container。

Experimenter 已在r2独立Python进程仅导入模块验证：导入前 `DATASETS.get('cifar2_das')` 报错，导入 `balds.workflows.das_import` 后返回 `_load_cifar2_das`。没有加载图片、拟合或生产科学产物。这是确定的入口初始化遗漏。

## 最小修改

1. 在prepare入口使用该数据集前显式初始化已有DAS注册模块，保持层依赖，不新写loader、不复制注册实现、不引入完整训练container。
2. 增加独立进程/真实入口路径的普通回归测试：不能依靠其他测试提前import container污染全局registry后才通过。用小型替身隔离图像加载，但保留实际注册路径；验证DDPM原ID映射及16面板构造。无需冒烟评分。
3. 针对正式prepare的真实主库来源做只读输入解析核对，临时清单可写独立临时目录；不执行SNR batch，不恢复模型，不下载数据。保留val原IDs 0..94,96..100及原DDPM响应，不用cifar2_5k替代。
4. 新快照的A输出设置为 `/path/to/CFA/_Data/results/snr_lds_20260925/a3/`，同时更新launcher ROOT、derived-input-root和说明配置，避免只改其中一个。a1的日志及recovery保留；a3不复用a1科学缓存，因为本次尚无manifest或SNR产物。新attempt名 `j3-snr-a3`，明确区别首次launch和已用过的一次恢复。

## 交付与恢复

- 不修改正在作为证据的r2只读快照。交付修正代码快照的真实路径和普通测试结果、完整launch/status/resume/verify命令；无需新hash流程。
- 不捆绑等待B，不改sigma拟合、阈值、方法清单或科学参数。
- Experimenter核对修正交付后发布A恢复依据；Executor不要再次启动r2，不在旧目录手动import后试跑。
- 首次空日志退出原因仍未确定，不声称它也由registry报错造成。当前有完整trace的失败才归因为注册遗漏。
- 本单不授权Coder启动正式重评；只交付修复与命令包。

## Coder delivery

2026-09-25 已交付并完成普通验证；正式重评未启动。固定修正版为
`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r3`，新 attempt
为 `j3-snr-a3`，输出根为
`/path/to/CFA/_Data/results/snr_lds_20260925/a3/`。完整证据和命令见
`CODER_FIX_A_DATASET_REGISTRATION_RECEIPT.md`。r2及a1证据未修改。
