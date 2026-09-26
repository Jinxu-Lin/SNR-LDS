# 终稿M1：分数密度与低幅值区域

有效标签：Fig1(a) `fig:motivation`、Fig4 `fig:mech-density-full`。依据`Paper/Sections/XY03_MechanismExperiment.tex:50–63`。旧concentration表和单独FMAS band-only Table3已退役；abs-head结果现在归入[M3](M3_TRANSFORMS.md)。

## 固定输入与参数

C2 FM三模型seed42/123/456，gen/val各100，N5000。四核心FMAS raw、EK-FAC IF、D-TRAK T100及DAS **平方前t**。复用[共用上游](00_SHARED_PIPELINE.md)原score，不再训练、采样梯度、fit曲率或重选正则化。

每query score除自己的RMS，保留符号与0点，每seed/track汇集500000个值。每cell robust Silverman带宽取中位数作为共同Gaussian带宽；8001 bins、零边界Gaussian平滑。接受范围±8 RMS，带宽.0454605734077672 RMS。曲线为3seed均值，带为seed范围；Fig1/Fig4虚线标记对应轨道FMAS pooled score中央50%（主图约±.502）。该绘图KDE不等于SNR逐query零均值噪声fit。

输入位于`scores/{fmas_raw,ekfac_if,dtrak_T100,das_T100}/cifar2_5k[_val]/seed_{42,123,456}/scores.npy`。密度图不需要GT/mask；原始DAS signed score不能被native平方数组覆盖。

## 基础密度计算

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
python Codes/Figure/fig_score_magnitude.py
python Codes/Figure/render_fig1_density.py
python Codes/Figure/render_appendix_density.py
```

基础入口仅计算终稿所需共同带宽与范围，写入`results/paper/figures/metadata/fig_score_magnitude.json`；两个renderer分别输出`Fig1/motivation-density.pdf`与`Fig4/density-all.pdf`及PNG。未重新计算时，renderer显式读取迁入`results/paper_figures/legacy/`的接受metadata。所有新产物都在`BALDS_FIGURE_ROOT`（默认`_Data/results/paper/figures`）内，不回写Paper。

## 来源与解释边界

`Reports/figure_split_2026-09-25/`保存终稿图重排及VarRatio替换记录；实际图形样式以当前Paper assets为准。这里低幅值是接近0的坐标，不是分布远尾；集中密度不证明真实影响严格为0。SNR拟合的额外噪声模型见[E-BENCH](E_BENCH.md)。
