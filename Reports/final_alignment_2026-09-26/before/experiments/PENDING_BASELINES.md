# 退役基线历史配置与既有输入记录

**状态更新（2026-09-25）：作者已将 Parameter-weighted D-TRAK、AbU+、NDA 从实验基线中退役，仅在论文 related work 中介绍。** 三者不进入任何当前实验表、manifest、coverage、方法排名或缺件待办。下面的配置、命令和旧“待完成”描述仅作历史记录，不构成当前执行任务；既有代码和工件保留。

AB2 IF 属于原有核心方法，不在本次退役范围内。本文件末尾保留其早期输入快照，当前状态以 [SNR-LDS 执行计划](../snr_lds_2026-09-25/PLAN_SNR_LDS_V1.md) 为准。

## 参数加权 D-TRAK：首个正式C2seed42单元

已明确配置：C2seed42，无条件，N5000，旧gen/val各100，复用D-TRAK T100训练特征；独立学习1000生成图，Euler100、gen_seed20260922，旧eval gen_seed165不变。304可训练参数张量组，先mask再共享原cuda_jl投影p4096/seed0；每query梯度T100、batch1；query贡献分块16；ridge1、mean_abs。softmax权重，AdamW10epochs、lr.01、cosine、topk10、logit decay0、init seed0。单一学习权重用于gen/val，不使用GT训练权重。

历史a1只生成1000图，首贡献块前fast_jl batch容量错误，0/63贡献，无weights/scores/BA。a2修复`36788a7a785b16b27fa1ad10dde0b2aa8d8bb43a`将内部kernel容量映射8，实际batch仍1，已ready；本次读到的台账不证明最终完成。

完整新环境流程如下，保存为独立tag，不覆盖普通D-TRAK：

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
TAG=pw_c2_s42_l1000_k10_wd0_v1
balds-run --data-root "$BALDS_DATA" --device cuda:0 generate \
  --dataset cifar2_5k --process cfm --uncond --seed 42 \
  --Q 1000 --ode-steps 100 --gen-seed 20260922 --batch 50 \
  --artifact-name weight_learning_samples_pw_c2_s42_v1
CFG=(--set parameter_weighting.feature=dtrak_T100
 --set parameter_weighting.learning_generation=weight_learning_samples_pw_c2_s42_v1
 --set parameter_weighting.learning_query_count=1000
 --set 'parameter_weighting.learning_query_ids=[]'
 --set parameter_weighting.ridge=1.0 --set parameter_weighting.kernel_normalize=mean_abs
 --set parameter_weighting.proj_dim=4096 --set parameter_weighting.proj_seed=0
 --set parameter_weighting.projection=cuda_jl --set parameter_weighting.batch_size=1
 --set parameter_weighting.contribution_query_chunk=16 --set parameter_weighting.query_seed=42
 --set parameter_weighting.epochs=10 --set parameter_weighting.lr=0.01
 --set parameter_weighting.scheduler=cosine --set parameter_weighting.top_k=10
 --set parameter_weighting.weight_decay=0.0 --set parameter_weighting.initialization_seed=0)
for STAGE in prepare fit; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 "${CFG[@]}" parameter-weighting "$STAGE" \
    --dataset cifar2_5k --process cfm --uncond --seed 42 --config-tag "$TAG"
done
for TRACK in gen val; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 "${CFG[@]}" parameter-weighting score \
    --dataset cifar2_5k --process cfm --uncond --seed 42 --config-tag "$TAG" --query-type "$TRACK"
  balds-run --data-root "$BALDS_DATA" --device cpu "${CFG[@]}" evaluate \
    --method dtrak_param_weighted_T100 --dataset cifar2_5k --process cfm --uncond --seed 42 \
    --config-tag "$TAG" --query-type "$TRACK"
done
```

公开`generate --batch 50`已按历史helper保留一次完整CPU初始noise抽样，再依次50行调用Euler，记录`generation_batch_size`；不把每块重新设seed或重新抽noise当同一学习集。已迁入匹配的1000图可直接复用。贡献/权重和score存于带`cfg_<tag>`的相对地址；BA按[E-BENCH](E_BENCH.md)读取该tag对应矩阵和原GT，native无DAS平方特殊处理。原score meta中的val真实ID需与GT列对齐。

## AbU+

当前主表空白。已裁定query梯度MC250、更新前后测训练loss10个时间点；这是项目预算适配，不称原文4000/10完整复现。原始计划需确定参数子空间、EK-FAC/预条件器实际接入、曲率缩放、步长。旧默认diagonal-only用例不能靠写一个ekfac配置伪装为原文方法。

公共接口`balds-run abu prepare/score --config-tag ...`存在，但**科学配置仍不完整，不能生成一份带任意默认步长的“论文正式命令”**。需补的具体项为参数域、预条件器及缩放、step size、每平台score身份；然后prepare→每query有限更新→匹配MC前后loss差→Full/BA。现有已批准MC250/10保持不变。

## NDA

当前主表空白。无模型梯度；原生图像patch空间，在全训练patch范围归一化，不因分块变为局部归一化。AB2仍图像空间，不静默换latent或shortlist。正式分辨率、noise kernel、时间/尺度配方、patch sizes、空间聚合待确认。公共`balds-run nda score --config-tag ...`是能力入口，不是这些参数已确定的证据。

## AB2 IF

已有全曲率和两轨MC250 query eigen-coordinates；训练行0–5000划8单元，每单元625行、每25行原子块、同时覆盖gen/val各100。U0–U2已验收共1875行，U3–U7待完成。收齐→CPU装配全阻尼5000×100→原native选参/Full→BA；缺任一块不能promote为正式完整score。FMAS曲率可共享不等于FMAS梯度分数可替换IF。

## 来源

源`Experiment/experiment.qmd:23–32,92–106,151–153`；`_Data/reports/weighted_dtrak_2026-09-22/{TASK_JINXU3_WEIGHTED_DTRAK_C2_V1.md,TASK_JINXU3_WEIGHTED_DTRAK_C2_V2.md}`；`closeout_dispatch_2026-09-21/{BASELINES_IMPLEMENTATION_DESIGN_V2.md,BASELINE_PARAMETER_AUDIT_V1.md}`；`ab2_if_joint_2026-09-21/DESIGN_AND_COMMANDS.md`；`ab2_remaining_2026-09-22/DISPATCH.md`。新增结果完成后按独立新验收更新本报告，不改历史失败或计划状态。
