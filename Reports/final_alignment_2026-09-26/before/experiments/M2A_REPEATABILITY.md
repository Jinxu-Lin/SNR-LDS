# M2a：完整 R16 重复性

标签：Figure1(f)、`tab:app-r16-bands`；论文`XY04_MechanismExperiment.tex:158–180`。本报告中的命令用于迁入输入后的重新执行，本次整理未运行重复实验。

## 参数与定义

C2 unconditional FM seed42，全部5000训练坐标，预选gen16/val16原query IDs，FMAS raw及D-TRAK T100。R=16，每repeat每方法每track一个float32(5000,16)，合计64矩阵。训练与query Monte Carlo都独立重抽；模型、投影、正则化保持固定。

FMAS MC250/250、stratified-antithetic、rho=.01、已有曲率固定；train mode及hflip开启，grad chunk125，query coords bfloat16，row block50。D-TRAK T100/grid/uniform/full读出、p4096、projection seed0、`torch_chunked`、batch16、feature block160、逐梯度normalize、lambda=.05，每repeat重建训练kernel和mean-abs归一化。这里不能套用其他实验的`cuda_jl`或更换MC batch。新重复不重训模型、不重新fit曲率、不优化lambda。

配置固定于`Codes/configs/repeatability.json`。query原ID为：

- gen：`6,7,13,14,22,34,36,41,49,56,57,61,66,79,90,95`。
- val：`3,7,14,16,19,25,32,49,50,60,78,79,80,84,86,88`。

原选取规则是PCG64 seed20260917，依次gen/val从100无放回选16并排序。RNG base42、stride1000000；train repeat offset1、query offset101；method offsets FMAS0/D-TRAK20000、track offsets gen0/val10000。代码按原训练行/query ID派生FMAS phase seed；D-TRAK保留原batch内位置与最后partial batch身份。完整派生规则保存在配置中。

独立pilot按原幅值固定0–1%、1–5%、5–20%、20–50%、50–100%带。每坐标Rep=`abs(mean_repeat(score)-median_query(mean_repeat(score))) / sample_std_repeat(score)`；保留零方差处理和原汇总顺序。分带LDS每repeat用同一GT计算后按原统计汇总。当前稿只展示Rep和分带LDS；不恢复旧Rep<1列、排序一致率或M2b采样干预。

## 历史输入依赖

- `results/e3c_full_repeats_20260918/repeat_{0,1,2,3}/`：完整R16的前四次，必须迁入。
- `results/narrative_wave_20260921/incoming/xuchang3/R-A/repeat_{4,5,6,7}/`；同根R-B为8–11、R-C为12–15。
- 独立pilot清单：`reports/narrative_gpu_wave_2026-09-21/PILOT_M2A_FOUR_TRACKS_V1.json`；四轨完整100列score也必须迁入。精简属性见[独立pilot证据](../evidence/r16_pilots.json)。
- 已接受分析：`results/narrative_wave_20260921/analysis/m2a_r16_20260921/`，四轨statistics、分带LDS及逐query数据。

旧机器名称仅是历史相对目录段，不用于连接远端。历史正式执行`554e8a73da837fad97f8ea2fa90f3171e7d0873f`基于0484977；旧`narrative_wave.py prepare/launch --lane R-A`内含主机调度，公开版不提供该旧调度接口。

## 从已有上游重新执行完整 R16

先完成[共用上游](00_SHARED_PIPELINE.md)的C2seed42模型、查询、GT及FMAS曲率。以下先准备独立pilot的指定16列，再执行prepare→16 repeats→状态核对→统计。

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

## 图表阶段

先运行M1/M3的CPU表图命令。组合图默认读取现稿已接受R16统计，因此迁入历史结果即可重现原图。若使用新跑的R16分析，必须显式指定新统计根，不能默认图中已经使用新结果。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/Figure/render_figure1.py
```

若使用上述新跑结果，在执行渲染前设置`export BALDS_R16_ROOT="$BALDS_DATA/results/repeatability_r16/analysis"`；不设置时读取已接受历史根。此脚本排版已有完整R16统计，不运行重复估计。输出会重写目标仓Paper内对应图表，原样复制版本应在审核新结果前保留。

## 历史验收

`_Data/reports/narrative_gpu_wave_2026-09-21/ACCEPTANCE_M2A_R16_V1.md`确认64/64矩阵、四轨pilot齐备；`closeout_dispatch_2026-09-21/RESULTS_INTEGRATION_20260921.md`完成CPU统计；`author_accepted_batch_2026-09-21/REPORT.md`与`author_r16_simplification_2026-09-21/`登记采用。旧文档的“R16尚缺repeat4–15”是更早快照，不代表当前缺件。
