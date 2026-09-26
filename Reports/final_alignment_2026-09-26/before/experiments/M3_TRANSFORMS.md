# M3：固定分数变换、平方与跨平台对照

标签完整清单见[实验索引](../EXPERIMENT_INDEX.md)。对应Figure1(b–d)、附录D幂变换/平方/同名单/符号及跨平台κ结果。

## 固定参数

C2三模型×双轨四核心构造，另C2十三构造幂曲线及C10四核心峰值。每query先除max(abs(score))；folded=`abs(t)^p`、signed=`sign(t)*abs(t)^p`，p=0,.2,…,4，另p6/8锚点。p0 folded为单位，signed只保留符号。固定原评分参数；局部抛物线peak只是描述。

四核心FMAS raw、IF、D-TRAK、DAS的普通square均从**一次项**出发。native-head比较时DAS用t²；同名单square比较的head固定由signed t最大值选取，两种读出使用完全相同坐标；不能把signed head和abs head混用。κ=5/10/20/30/40/50/60/70/80/90/100%。

matched DAS跨C2/C10/AB2六轨：一次项、普通square、signed square，各自重排head及固定一次项head。AB2使用native lambda1的一次项恢复值，不是旧lambda.5。paired bootstrap2000，validation跨seed同步抽query、generation每seed内抽；单模型AB2是条件query区间。

## 执行线

[共用上游](00_SHARED_PIPELINE.md)→固定scores/masks/GT→变换或head mask→deleted-row sum→逐query Spearman→跨seed/paired统计→图表。无新增模型、曲率、梯度采样或响应重训。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/Figure/tab_3_1_square.py
python Codes/Figure/tab_C1_battery.py
python Codes/Figure/fig1_power_diagnosis.py
python Codes/Figure/tab_4_2_square_gain.py
python Codes/Figure/compute_square_heads.py
python Codes/Figure/render_figure1.py
```

Figure1最后一步还需M1的density/head JSON和M2a正式统计。不要运行旧`make_all.py`，它包含退役实验。`compute_square_heads.py`写同名单结果`square_heads.json`和表，保留source/selection描述。

跨平台接受分析来源为`results/lds_nextwave_20260920/j2/a1/transforms/{panels,summaries.json,paired_differences.json,missing.json,completion.json}`，54有效panel。原生κ/paired/sign扩展使用既有E4：`results/lds_e1e5_wave1_20260917/jinxu2/e4_c10_ab2_summary.json`及纠正后的长表。不要用已退役E1/A2或D2–D4分析填这些标签。

54-panel跨平台CPU算法已经整理为便携入口，与S3共享一次计算：

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/tools/aggregation_controls.py transforms \
  --data-root "$BALDS_DATA" \
  --output "$BALDS_DATA/results/paper/controls/transforms"
```

输出保留panel、paired differences、missing和summary；源历史接受输出与重新运行结果放不同目录。此入口不包含主机调度，不增加科学条件。

## DDPM共同100及存档1000

DDPM独立C2样本、38.3M模型、DDIM50生成；原存档M128只读前64，响应前三重训×三噪声平均。两轨共同100：gen原前100；val按balanced query索引映射到存档列。投影D-TRAK/DAS regularization=10；IF lambda1e-12；FMAS扩展rho grid1e-8至1，选gen1e-7/val1e-8。三个readout在原始LDS选参后固定，不能逐变换重选。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
balds-run --data-root "$BALDS_DATA" --device cpu import-das --root "$BALDS_DATA/raw/das_archive" --model
balds-run --data-root "$BALDS_DATA" --device cuda:0 ekfac fit --dataset cifar2_das --process ddpm --seed 42
for TRACK in gen val; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 --set lds.Q_by_dataset.cifar2_das=100 --set 'ekfac.blockshrink_grid=[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]' ekfac score --method fmas_raw --dataset cifar2_das --process ddpm --seed 42 --query-type "$TRACK"
  balds-run --data-root "$BALDS_DATA" --device cuda:0 --set lds.Q_by_dataset.cifar2_das=100 ekfac score --method ekfac_if --dataset cifar2_das --process ddpm --seed 42 --query-type "$TRACK"
done
python Codes/Figure/tab_C3_ddpm.py
```

导入保留1000列 projected特征和响应；以上Q100覆盖只用于曲率。`tab_C3_ddpm.py`同时处理共同100与完整1000，勿把全量投影query重存为100。原DDPM1000表只含D-TRAK/DAS；原score lambda均10。另64子集有明显批次偏移，不用于任何论文LDS。

## 已接受边界与证据

C2 gen原始LDS FMAS .3937、IF .3847、D-TRAK .3922、DAS .3841；ordinary square分别 .5109/.4824/.4735/.4694。同原signed支持head，DAS .4550→.4468，不是按native重新选头的.4386。C10/AB2一些head仍有正square增益，不推广“平方优势一律消失”。这些机制控制与当前BA t拟合/t²聚合是不同实验。

源证据：`lds_nextwave_2026-09-20/ACCEPTANCE_J2_CPU_j2-a1.md`（1f2e6d1，54panel）；`lds_e1e5_wave1_2026-09-17/ACCEPTANCE_JINXU2_j2-a3_20260917.md`（E4）；`ddpm_c3_2026-09-14/{RESULTS.md,ACCEPTANCE_DDPM_C3_A.md,ACCEPTANCE_DDPM_C3_B.md,RUN_DDPM_C3_B2_j3-c3B2-a1.md}`。精确四样本反例保留为推导，不恢复随机模拟。
