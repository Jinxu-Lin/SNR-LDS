# M2a 完整 R16 独立验收 — 2026-09-21

结论：**评分产物 accepted；完整 R16 CPU 统计 ready、尚未执行。**
本验收者未参与该评分实现或运行。读回主机为 anonymous-lab3 正式 _Data，未修改原始输入、旧 R4、新 R16 或运行中的 M2b 包。

## 范围与结果

- 旧 repeat0–3 从 results/e3c_full_repeats_20260918 复用；新 repeat4–15 从 incoming/xuchang3/R-A,R-B,R-C 读取；无重复来源、无缺 repeat。
- 16 repeats × FMAS/D-TRAK × gen/val = **64/64** 矩阵，其中新增48，旧16。
- 每份原始矩阵为 float32(5000,16)，全 finite。身份、固定科学配置、输入清单、训练行、原始查询列及 RNG streams 由已接受 execution 554e8a7 的 score_index / checked_repeat / load_score 实際核对。
- 四个 method/track 各16矩阵两两非相同，不把同一数组复制件算独立重复。这里只检查未误复用，随机流独立性来自设计和记录，不从差异反推。
- 每 repeat 的 progress.stage=complete；runtime 均 torch2.6.0+cu124；新12次主机为 xuchang-lab3。具体来源、路径与版本见 VERIFY_M2A_R16_V1.json。
- MC：FMAS 训练与查询各250，D-TRAK T100，训练与查询均重抽；不是旧训练侧 repeat_scores 的扩充。R16原配置、模型、曲率、固定32查询（gen16+val16）与原生方法不变。

## Pilot 接收

FMAS既有两轨证据保持原样；D-TRAK恢复了原始seed42特征、原始λ=.05评分、行列映射与随机流。
新独立清单：PILOT_M2A_FOUR_TRACKS_V1.json。
实际 load_pilots 已成功加载四轨，各选出(5000,16)；与R0–R15的有效MC seed集合不相交。
D-TRAK的详细证据和旧commit不可解析边界见 PILOT_DTRAK_PROVENANCE_V1.md。
没有新增GPU、不重新选λ、不用本轮R16选pilot、不改M2b的 packages/pilot_manifest.json。

## 下一步

可用已接受 R16 工具对三个 canonical incoming roots 及新四轨 manifest 做正式CPU分带/矩估计。
本轮只做产物验收与溯源，没有运行正式统计，因此不宣称M2a的统计图表/论文结论已经完成。
不需要新GPU、图片审核或Coder改动；用真实源路径，不复制/搬动正在消费的区域输入。

## 实际验收命令

运行于 /path/to/CFA；CUDA_VISIBLE_DEVICES为空，OMP/MKL各2线程。耗时约1秒；GPU卡时0。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/path/to/CFA-worktrees/r16-xc3-20260921/Codes /path/to/miniconda3/envs/da/bin/python - <<'PY'
from cfa.app.narrative_m2 import score_index,load_pilots,config
from pathlib import Path
import json,numpy as np
D=Path('/path/to/CFA/_Data');sources=[D/'results/narrative_wave_20260921/incoming/xuchang3'/x for x in ('R-A','R-B','R-C')]
v,idx=score_index(D,sources)
rows=[]
for r in range(16):
 p=D/'results/e3c_full_repeats_20260918'/f'repeat_{r}' if r<4 else sources[(r-4)//4]/f'repeat_{r}'
 progress=json.loads((p/'progress.json').read_text());runtime=json.loads((p/'runtime.json').read_text())
 assert progress['stage']=='complete'
 rows.append({'repeat':r,'root':str(p),'stage':progress['stage'],'host':runtime['host'],'torch':runtime['torch'],'matrix_count':len(v[r])})
pilots=load_pilots('/path/to/CFA/_Data/reports/narrative_gpu_wave_2026-09-21/PILOT_M2A_FOUR_TRACKS_V1.json',config())
assert len(pilots)==4
print(json.dumps({'complete':idx['complete'],'missing_repeats':idx['missing_repeats'],'matrices':len(idx['scores']),'shape_per_matrix':[5000,16],'dtype':'float32','all_finite':True,'pilots_loaded':[m+'/'+t for m,t in pilots],'repeats':rows,'distinct_matrix_counts':{m+'/'+t:len({v[r][m,t].tobytes() for r in range(16)}) for m,t in pilots},'score_index':idx['scores']},indent=2))
PY
```

命令输出保存于 VERIFY_M2A_R16_V1.json。读取已有manifest历史字段，无新hash、无冒烟。
