# 主模型训练档案

| 平台 | 模型 seed | 当前生产入口 / 身份 |
|---|---|---|
| CIFAR-2 CFM | 42, 123, 456 | [三 seed 专项档案](cifar2_cfm/README.md)；无条件模型 |
| CIFAR-10 CFM | 42, 123, 456 | [共用流程第 1 节](../00_pipeline/README.md)；条件模型，`p_uncond=0` |
| ArtBench-2 SD3.5 Medium | 42 | [共用流程第 5 节](../00_pipeline/README.md)；LoRA 微调，先准备 latent |
| CIFAR-2 DDPM | 42（存档标识） | [共用流程第 5 节](../00_pipeline/README.md)；导入参考存档，不等价于重新训练 DDPM |
| CIFAR-10 注入 CFM / DDPM | 各 42 | [来源检索第 2 节](../07_source_retrieval/README.md)；两个独立模型 |

所有训练平台的公共参数、生成查询和下游特征计算均在共用流程中登记，避免多份配方漂移。
CIFAR-2 CFM 档案进一步提供独立执行脚本、三个 seed 的 loss history、检查点校验和及元数据。
其他平台目前保留配方及输入路径，不宣称已经补齐原始 stdout、硬件环境或训练时的源码版本。
子集模型与删除模型是独立的重训任务，分别在共用流程第 2 节和删除实验中说明。
