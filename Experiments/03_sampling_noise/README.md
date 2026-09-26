# M2a：完整 R16 重复性

终稿标签：Figure1(b) `fig:motivation`、Table3 `tab:app-r16-bands`。本报告中的命令用于迁入输入后的重新执行，本次整理未运行重复实验。

## 参数与定义

C2 unconditional FM seed42，全部5000训练坐标，预选gen16/val16原query IDs，FMAS raw及D-TRAK T100。R=16，每repeat每方法每track一个float32(5000,16)，合计64矩阵。训练与query Monte Carlo都独立重抽；模型、投影、正则化保持固定。

FMAS MC250/250、stratified-antithetic、rho=.01、已有曲率固定；train mode及hflip开启，grad chunk125，query coords bfloat16，row block50。D-TRAK T100/grid/uniform/full读出、p4096、projection seed0、`torch_chunked`、batch16、feature block160、逐梯度normalize、lambda=.05，每repeat重建训练kernel和mean-abs归一化。这里不能套用其他实验的`cuda_jl`或更换MC batch。新重复不重训模型、不重新fit曲率、不优化lambda。

配置固定于`Codes/configs/repeatability.json`。query原ID为：

- gen：`6,7,13,14,22,34,36,41,49,56,57,61,66,79,90,95`。
- val：`3,7,14,16,19,25,32,49,50,60,78,79,80,84,86,88`。

原选取规则是PCG64 seed20260917，依次gen/val从100无放回选16并排序。RNG base42、stride1000000；train repeat offset1、query offset101；method offsets FMAS0/D-TRAK20000、track offsets gen0/val10000。代码按原训练行/query ID派生FMAS phase seed；D-TRAK保留原batch内位置与最后partial batch身份。完整派生规则保存在配置中。

**当前统计是VarRatio，不再报告旧Rep。** 在每query/bin内计算`sum(sample_variance)/sum(repeat_mean**2)`，sample variance使用ddof1；再对16query等权平均。分母是observed repeat mean平方之和，不能改为median中心化或逐坐标ratio平均。

主Fig1(b)用FMAS gen完整16×5000×16分数，按每query的abs(repeat mean)稳定升序排序，分成10个500样本等大bin，低幅值到高幅值；不使用pilot决定这10个bin。接受VarRatio依次19.7016、2.9420、1.0356、.5176、.3163、.2080、.1456、.1042、.0863、.0452。

Table3仍用**独立pilot**按原abs分数固定0–1%、1–5%、5–20%、20–50%、50–100%五带，四method/track分别统计VarRatio；band-LDS每query先平均16repeat LDS，再平均16query。该表从高幅值到低幅值，与主图方向相反。原完整R16数据保持不变，旧Rep值只存历史。FMAS波动条件于固定曲率，不包含反复fit曲率的总变异；R16不是SNR评价器的必要输入。

## 历史输入依赖

- `results/e3c_full_repeats_20260918/repeat_{0,1,2,3}/`：完整R16的前四次，必须迁入。
- repeat4–15 按三个批次保存在外部数据根；历史路径由当前 `render_relative_variation.py` 的索引解析，不能仅以新目录名替代身份核验。
- 独立pilot清单：`reports/narrative_gpu_wave_2026-09-21/PILOT_M2A_FOUR_TRACKS_V1.json`；四轨完整100列score也必须迁入。下面的准备代码明确各轨查询索引和来源。
- 已接受分析：`results/narrative_wave_20260921/analysis/m2a_r16_20260921/`，四轨statistics、分带LDS及逐query数据。

历史多机调度不作为公开执行接口；本报告给出当前代码的单 worker 复现流程。原始执行时的源码对象未全部保留，不能宣称当前 Git 包含这些历史版本。

## 从已有上游重新执行完整 R16

先完成[共用上游](../00_pipeline/README.md)的C2seed42模型、查询、GT及FMAS曲率。以下先准备独立pilot的指定16列，再执行prepare→16 repeats→状态核对→统计。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python - <<'PY'
from pathlib import Path
import json,os
import numpy as np
root=Path(os.environ['BALDS_DATA'])
cfg=json.loads(Path('Codes/configs/repeatability.json').read_text())
out=root/'results/repeatability_r16_pilots'
out.mkdir(parents=True,exist_ok=True)
manifest={}
for method in ['fmas_raw','dtrak_T100']:
  manifest[method]={}
  for track in ['gen','val']:
    ds='cifar2_5k' if track=='gen' else 'cifar2_5k_val'
    filename='scores_lambda_1.00e-02.npy' if method=='fmas_raw' else 'scores.npy'
    source=root/'scores'/method/ds/'seed_42'/filename
    a=np.load(source)
    assert a.shape==(5000,100), (source,a.shape)
    name=f'{method}_{track}.npy'
    np.save(out/name,a[:,cfg['selection'][track]])
    manifest[method][track]=name
(out/'pilot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
balds-repeat prepare --data-root "$BALDS_DATA" --output results/repeatability_r16
for REPEAT in {0..15}; do
  balds-repeat run --data-root "$BALDS_DATA" --output results/repeatability_r16 \
    --repeat "$REPEAT" --device cuda:0
done
balds-repeat status --data-root "$BALDS_DATA" --output results/repeatability_r16
balds-repeat summarize --data-root "$BALDS_DATA" --output results/repeatability_r16 \
  --pilot-manifest "$BALDS_DATA/results/repeatability_r16_pilots/pilot_manifest.json"
```

pilot须保留已接受的原独立分数，不能改为16次中的一次，不能从16次均值反向选带。历史FMAS/D-TRAK pilot的代码对象不全，已接受provenance通过日志/manifest/rename map及对应源语义重建；详见精简证据，不能宣称旧commit均可检出。

输出位于`results/repeatability_r16/analysis/<method>/<track>/{statistics.json,coordinates.npz}`与`analysis/summary.json`。新的summarize要求同一已prepare输出目录有16次完整身份；它不直接消费历史分散四个根。复用历史64矩阵时可直接读已验收分析，或先按迁移方案统一索引并核对package身份；不要仅复制文件夹后伪称新任务已prepare。

## 数值与绘图交付

新`summarize`输出使用VarRatio、pilot-band LDS以及mean-ranked decile结果。历史分散repeat0–15和接受coordinate_stats可以复算相同统计；不要把旧`rep_median`字段重命名为VarRatio。接受小型数值证据见 [deciles](evidence/fmas_varratio_deciles.csv) 和 [pilot bands](evidence/pilot_band_varratio.csv)。

终稿绘图输出与Paper分开。复用已接受64矩阵及索引：

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
python Codes/Figure/render_relative_variation.py
```

消费本报告新生产的完整R16分析时，显式设置分析根：

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
export BALDS_R16_ROOT="$BALDS_DATA_ROOT/results/repeatability_r16/analysis"
python Codes/Figure/render_relative_variation.py
unset BALDS_R16_ROOT
```

输入为各method/track的`coordinates.npz`（新）或`coordinate_stats.pt`（历史）；新分析根不需要旧分散score索引。输出`results/paper/figures/Fig1/motivation-repeatability.pdf`及metadata下两个CSV。旧`render_figure1.py`的六面板布局已移出公开入口。独立SNR重复校准B线不在当前论文，不能为此重新生产额外repeat。

## 历史验收

`_Data/reports/narrative_gpu_wave_2026-09-21/ACCEPTANCE_M2A_R16_V1.md`确认64/64矩阵、四轨pilot齐备；`closeout_dispatch_2026-09-21/RESULTS_INTEGRATION_20260921.md`完成CPU统计；`author_accepted_batch_2026-09-21/REPORT.md`与`author_r16_simplification_2026-09-21/`登记采用。旧文档的“R16尚缺repeat4–15”是更早快照，不代表当前缺件。
