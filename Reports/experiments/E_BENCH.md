# SNR-LDS 主表、阈值与覆盖

终稿标签：Tables1/15 `tab:exp-snr-cifar` / `tab:exp-snr-platforms`；Table14 `tab:snr-coverage`；Table16 `tab:snr-sensitivity`；Fig9/10拟合图；Fig3(a)固定筛选读出。依据 `Paper/Sections/XY04_EvaluationPipelines.tex:13–60` 及 `XY05_EvaluationExperiments.tex:73–84,134–172`。9月22 BA版本在日期快照中保留，不是当前待重评配方。

## 精确规则

每query拟合原signed score（DAS为native平方前t），除RMS。Gaussian KDE带宽为`0.9*min(sampleSD,IQR/1.34)*N**(-1/5)`；MAD绕median计算，但101点拟合区间**以0为中心**，从`-1.4826MAD`到`+1.4826MAD`。以KDE密度作权重拟合`log f(x)=A+C*x²`，没有线性项。`C<0`时`sigma_x=sqrt(-1/(2C))`；原单位sigma=rms*sigma_x。截距不进入选择，不计算pi0、gamma或混合密度。

共同硬规则为`abs(x)/sigma_x>zeta`，严格大于。默认zeta3；1/2/3/4共用同一sigma。普通方法保留入选原值与符号；DAS按signed t选择、聚合`t²`一次。选择对所有subset固定，不读GT进行拟合或阈值选择。

全零向量直接合法空集合。非正带宽/中心宽度、秩亏/非有限fit、C>=0、非正或非有限sigma失败NA。合法空集合及常数预测/响应记0，分类报告。Full/SNR同方法使用相同有效query；跨方法own-valid均值不是配对检验。

## 数据与正式状态

[共用上游](00_SHARED_PIPELINE.md)产出scores(N,Q)、keep masks(M,N)和GT(M,Q)。C2/C10FM三seed双轨各100，AB2/DDPM单seed双轨各100；M=64/64/32/64。保留16标准方法，Journey只gen。AB2 IF采用Full LDS的逐轨oracle选择：gen lambda1e-7、val1e-6，原响应同时用于选参与评价；不可写成独立test选参。

正式归档来自`results/snr_lds_20260925/a3`逐query/fit/mask与`a4`后处理，243格、24300物理query；16946有效、7354失败、3876空选择。像素/CLIP全fit失败也要保留原输入与失败证据。共享validation表格只计一次。Table1 DDPM TracInCP/GAS四格汇总由作者9月26提供，不具有已发表逐querycoverage；Journey DDPMgen没有匹配旧GT的正式结果。新结果若补齐，独立登记，不覆盖历史覆盖统计。

## 完整便携执行线

从仓库根执行。配置`Codes/configs/paper.json`明确SNR规则、阈值、平台及原生输入manifest；这些旧名称manifest只是输入ID/路径证据，不意味着运行旧BA。迁移后需保留其引用的实际scores和GT。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
python Codes/tools/reproduce_benchmarks.py prepare --data-root "$BALDS_DATA_ROOT" --output results/paper/snr
python Codes/tools/reproduce_benchmarks.py evaluate --data-root "$BALDS_DATA_ROOT" --output results/paper/snr --device cpu
python Codes/tools/reproduce_benchmarks.py postprocess --data-root "$BALDS_DATA_ROOT" --output results/paper/snr
python Codes/tools/reproduce_benchmarks.py status --data-root "$BALDS_DATA_ROOT" --output results/paper/snr
python Codes/tools/reproduce_benchmarks.py verify --data-root "$BALDS_DATA_ROOT" --output results/paper/snr
python Codes/tools/plot_snr_benchmarks.py \
  --data-root "$BALDS_DATA_ROOT" \
  --manifest "$BALDS_DATA_ROOT/results/paper/snr/inputs/benchmark_manifest.json" \
  --results "$BALDS_DATA_ROOT/results/paper/snr/panels" \
  --output "$BALDS_DATA_ROOT/results/paper/snr/figures"
```

prepare核对/对齐ID，处理DDPM balanced val100与原1000列、AB2M32及lambda1 signed DAS恢复，不训练。evaluate执行新SNR拟合，默认同时保存四阈值；postprocess只读逐query输出，生成summary、sensitivity、coverage与全局E-DEL；status读取完成记录；verify核对选择与预测代数。不得把整链称为本轮迁移已运行。

若只重现已接受统计，应以实际迁移后的a3 portable manifest和a3 panels作为`postprocess --manifest ... --results ...`输入，输出新目录；不为画图重复拟合。历史manifest若含绝对路径，应由迁移清单统一重定位后使用，不能静默指回CFA。

单个DAS单元的明确入口：

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
balds evaluate \
  --scores "$BALDS_DATA_ROOT/scores/das_T100/cifar2_5k/seed_42/scores.npy" \
  --score-space das-presquare \
  --masks "$BALDS_DATA_ROOT/subsets/cifar2_5k_masks.pkl" \
  --responses "$BALDS_DATA_ROOT/results/gt_matrix_fm_cifar2_5k_seed_42.npy" \
  --n-subsets 64 --rule snr_zero_mean_gaussian_v1 --zetas 1,2,3,4 --primary-zeta 3 \
  --device cpu --save-fits --output "$BALDS_DATA_ROOT/results/paper/snr_single_das"
```

显式`--fit-scores signed_t.npy --scores native_t_squared.npy`也受支持，但同一调用不可再要求das-presquare重复平方。

## 汇总与诊断

输出包括逐query SNR、Full、fit状态、各阈值保留数/预测、sigma与原query IDs。每seed先按该方法有效query平均，再等权跨seed并报sampleSD；shared pixel/CLIP val只有身份及逐query结果一致时才去重。缺输入、失败、合法空集合和常数分开。

Fig3(a)复用zeta3 mask，与原abs(pre-square)各5/10/20/30/40/50/60/70/80/90/100% head取交集，在线性/ordinary-square读出间保持同名单和sigma。C2 gen三seed、四核心；不重拟合、不改阈值。接受分析为`Reports/snr_lds_2026-09-25/fig3_snr_controls.json`与对应acceptance；便携入口如下；读取既有拟合结果，不重复拟合。

Fig9固定C2 seed42 gen q0/q1四核心；Fig10仅画有效记录的sigma/保留率，失败仍进入Table14。绘图输出与Paper分开，不能用旧BA拟合图替代。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
FIGURE_OUT="$BALDS_DATA_ROOT/results/paper/figures"
python Codes/tools/render_snr_tables.py \
  --summary "$BALDS_DATA_ROOT/results/paper/snr/tables/summary.json" --output "$FIGURE_OUT"
python Codes/tools/render_fig3_snr_controls.py \
  --data-root "$BALDS_DATA_ROOT" --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels" \
  --output "$FIGURE_OUT"
python Codes/tools/plot_snr_benchmarks.py \
  --results "$BALDS_DATA_ROOT/results/snr_lds_20260925/a3/panels" \
  --manifest "$BALDS_DATA_ROOT/results/paper/snr/inputs/benchmark_manifest.json" \
  --data-root "$BALDS_DATA_ROOT" --output "$FIGURE_OUT"
```

对应Tables1/14/15/16、Fig3(a)、Figs9/10。若采用新`evaluate`结果，两个绘图的`--results`改为同一新结果根，保持manifest一致。作者aggregate-only补充独立读取`Codes/configs/paper_aggregate_supplement.json`，不合成per-query记录。

## 来源

- `Reports/snr_lds_2026-09-25/{TASK_JINXU3_SNR_LDS_CPU_V4.md,RUN_JINXU3_SNR_LDS_a4.md,ACCEPTANCE_AND_PAPER_ADOPTION_A4.md}`。
- `Reports/snr_lds_2026-09-25/ACCEPTANCE_FIG3_SNR_CONTROLS.md`。
- `Paper/Figures/Table{1,14,15,16}/table.tex`；Table1明确作者汇总四格来源限制。
