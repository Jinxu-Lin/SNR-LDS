# SNR-LDS 终稿Figure便携化与真实复算回执

2026-09-26。代码位于`Codes/Figure`，输入为本轮已独立迁入 SNR-LDS 的`_Data`，新输出为`_Data/results/paper/figures`。历史源仓与Paper未改写。原入口/README已存`before/Codes/Figure`。详细stdout、stderr、耗时与首轮失败记录完整保留于[figure_validation.json](figure_validation.json)，并没有只保留成功日志。

## 最终范围及已运行结果

- Fig1(a)密度、Fig1(b)VarRatio、Fig4双轨密度、Fig5双轨幅值头、Fig6四核心幂图：renderer实际运行通过。
- Fig2相同幅值头的linear/square逐seed计算完成；稳定head profile重新生成后，独立head结果严格一致，renderer通过，输出两张PDF与八行TeX。
- Table5同分数linear/folded/signed三seed分析实际运行完成。
- Table10固定参数matched100分析实际运行完成，保存八行逐query结果及paired bootstrap。既有IF/FMAs四行显示数值和区间与Paper一致；投影重建差异见下节。
- R16十个500行幅值bin的VarRatio均值：19.701613、2.942026、1.035636、.517569、.316320、.208042、.145644、.104164、.086333、.045170。与9月25接受数值一致。五个独立pilot band的四method/track统计也已重新导出CSV。
- 所有Figure Python文件AST解析通过，当前实验报告与Figure README中所有bash fenced block通过`bash -n`。代码中无服务器/源checkout绝对路径，无Paper写入目的地。

本轮是已保存研究工件的CPU读出与排版验证，没有训练模型、重新生成query、采样新梯度或重新拟合曲率。Table10唯一重建score的步骤是对既有投影features运行固定lambda10 kernel；没有改动或重选它的参数。

## 必须保留的两个数值差异

### 1. 主机制图的head tie规则

终稿`Paper/Sections/XY03_MechanismExperiment.tex:43`明确写：

> The main aggregation controls rank samples by absolute pre-square score and retain $k=\lceil\kappa N\rceil$ samples, breaking ties by training index.

旧Figure实际使用`np.argpartition`。在相同迁入score/mask/GT上重新执行旧算法，全部12个method/seed top5% linear值与归档JSON精确一致，因此不是数据身份变化。稳定排序改变边界名单的query数如下（每seed100query）：

| 方法 | seed42 | seed123 | seed456 |
|---|---:|---:|---:|
| FMAS | 55 | 68 | 49 |
| EK-FAC IF | 60 | 48 | 53 |
| D-TRAK | 0 | 0 | 0 |
| DAS | 0 | 0 | 0 |

完整逐seed旧值、新值、变动行数见[head_tie_audit.json](head_tie_audit.json)。FMAS 30% head旧42.202209、新42.245926 LDS×100；该FMAS曲线最大已观察差0.0437164。全部四方法、11头比例、两readout的最大差为IF20% linear：旧42.452577、新42.529745，差0.0771680 LDS×100；完整88格见[head_curve_comparison.json](head_curve_comparison.json)。新代码遵循终稿稳定index规则，`fig2_head_profile.py`与`compute_abs_heads.py`一起重算；`render_fig2.py`仍保持独立数值相等断言。第一轮混用旧curve与新table的断言失败被保留在日志中，而不是放宽容差或抄回旧均值。

重新生成的Fig2/5因此与Paper旧点略有差别。Paper资产和legacyJSON保留；此处不主张它们验证了终稿所写稳定tie规则。

### 2. Table10投影feature的CPU重建

新入口只报告两个轨道各100 matched query、前64子集和四方法。固定参数为投影lambda10、IF1e-12、FMAS gen1e-7/val1e-8。原1000-query投影features保留用于同一kernel完整计算后再取100列；它不是退役1000-query表的恢复。

IF/FMAs的四行使用接受per-damping score矩阵，均值和区间复现Paper所有显示值。DTRAK/L1没有接受的固定lambda10完整score矩阵，需要由float32 features重建；本次CPU结果与Paper部分单元相差约0.01–0.02 LDS×100。例如DTRAK gen Original新24.38/稿24.37，L1 gen Folded新32.13/稿32.12；DTRAK val Folded新43.48/稿43.49。具体历史数值环境差异尚未定位，不把差异称为已证明的BLAS原因。

新结果保存在`metadata/tab_C3_ddpm_matched100.json`，旧`results/paper_figures/legacy/tab_C3_ddpm.json`保留接受均值。没有把旧均值覆盖到新逐query计算中，也没有宣称全表打印值严格复现。

## 路径与生产约定

`BALDS_DATA_ROOT`选择输入，`BALDS_FIGURE_ROOT`选择图表输出；均可为新checkout之外的路径。历史索引里的绝对`_Data`路径只重定位到所选输入根，不回退CFA。`BALDS_R16_ROOT`可选新`balds-repeat summarize`的analysis根，读取四套`coordinates.npz`；未指定则复用原索引及64重复矩阵。

完整执行线、来源及依赖见[Figure README](../../Codes/Figure/README.md)、[M1](../experiments/M1_MAGNITUDE.md)、[M2a](../experiments/M2A_REPEATABILITY.md)、[M3](../experiments/M3_TRANSFORMS.md)。其余最终SNR/删除/匹配DAS图表由`Codes/tools`入口产生，验证由根任务单独记录；来源检索replay已核对26方法过程与572表格数值，见[E-SOURCE](../experiments/E_SOURCE.md)。
