# M1：分数幅值与聚合范围

标签：Figure1(a,e)、`tab:mech-concentration`、`fig:mech-density-full`、`fig:mech-head-full`、`tab:mech-band-lds`；论文`XY04_MechanismExperiment.tex:61–156`。

## 固定设计

C2 FM三模型seed42/123/456，gen/val各100，N5000，M64，三链×三噪声平均响应；FMAS raw、EK-FAC IF、D-TRAK T100、DAS **平方前**四种固定构造。上游[共用管线](00_SHARED_PIPELINE.md)阶段1–4；本实验不再训练、不重新提特征/拟合曲率、不重选正则化。

密度逐query除以RMS后汇集，每seed每track500000分数；每cell robust Silverman带宽取中位数作为共同Gaussian带宽，8001 bins，零边界Gaussian平滑。原接受绘图范围±8 RMS（所有cell绝对值99.9分位向上取整），带宽0.0454605734077672 RMS；线是三seed密度均值，带是seed范围。这是M1绘图协议，不替代BA逐query KDE。这里的低幅值tail指靠近0的训练坐标，不是统计分布的远尾。

集中度先在每query内计算top5%/bottom50%的绝对值和份额，以及同一原始幅值名单的平方和份额，再跨query/seed平均。幅值head按原分数绝对值排名选5/10/20/30/40/50/60/70/80/90/100%，保留原符号原值；分带5–50%、50–100%单独求和。不能把support-direction head曲线当absolute-magnitude head曲线。

## 执行线

1. 读取`scores/{fmas_raw,ekfac_if,dtrak_T100,das_T100}/cifar2_5k[_val]/seed_{42,123,456}/scores.npy`。
2. 读取`subsets/cifar2_5k_masks.pkl`和`results/gt_matrix_fm_cifar2_5k[_val]_seed_<seed>.npy`。
3. 逐query集中度、RMS密度、幅值head和分带LDS；逐seed平均，最终mean/sampleSD。
4. 导出JSON/表，再重绘现稿图；不以图的平滑值替代计算点。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/Figure/fig_score_magnitude.py
python Codes/Figure/fig2_head_profile.py
python Codes/Figure/tab_3_1_bands.py
```

公开整理版的`tab_3_1_bands.py`只保留当前M1分带分析；旧训练侧256坐标R4重复统计不属于该命令的目标。`_Data/results/paper_figures/legacy/fig_score_magnitude.json`和`fig2_head_profile.json`用于现稿组合图；完整Figure1还依赖M2a和M3，见各报告。

## 已接受的数值锚点

FMAS gen full=.3937、absolute-top5%=.4789，val .4433→.5183。FMAS gen top5%占绝对和21.7%，同一名单占平方和51.1%；bottom50%由17.8%降到4.1%。这些只说明所测score/response面板上的聚合性质，不是小分数真实影响为零的证明。

## 历史证据与重现边界

源 `Codes/Figure/{_common.py,fig_score_magnitude.py,fig2_head_profile.py,tab_3_1_bands.py}`；`Paper/ICLR/Figures/section31/{figure1_sources.json,magnitude-head-table.tex}`；`_Data/reports/author_fig1_completion_2026-09-21/`。保留原选定分数，尤其DAS不得将native平方数组覆盖原signed数组。完整图的布局源和数值JSON已随Paper复制；数据根迁移后再执行上述命令。
