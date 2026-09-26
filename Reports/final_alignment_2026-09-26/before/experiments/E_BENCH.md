# E-BENCH：主表与跨平台 BA-LDS

论文位置：主文 `tab:exp-ba-cifar`（别名`tab:exp-c2`），附录`tab:exp-ba-platforms`（别名`tab:exp-platforms`），以及四个平台coverage表。依据 `Paper/ICLR/Sections/05_Experiment.tex` 和 `XY05_EvaluationExperiments.tex:171–258`。

## 目标、输入和当前口径

比较相同背景筛选规则应用于各方法后对删除子集损失的预测。固定模型、查询、原评分配置和响应；不根据BA结果重选lambda或阈值。上游完整生产流程见[共用管线](00_SHARED_PIPELINE.md)。

每个方法输入score矩阵(N,Q)、keep mask(M,N)、响应(M,Q)。C2/C10各3模型×gen/val100；AB2及DDPM各1模型×双轨100。核心FMAS=`fmas_raw`。保留pixel/CLIP dot/cos、gradient dot/cos、TracInCP、GAS、Journey、Relative IF、Renorm IF、TRAK、D-TRAK、DAS、IF和FMAS。2026-09-25 作者已将 Parameter-weighted D-TRAK、AbU+、NDA 退出实验基线，仅保留 related work 介绍，不再列为缺格或待补实验；旧配置见[退役记录](PENDING_BASELINES.md)。

**本次用户明确的DAS规则：** `fit_scores=t`，`scores=t²`。t是同一原生配置下平方之前的有符号向量；不能用sqrt(t²)恢复丢失的符号。Full=sum(D*t²)，BA=sum(D*H(t)*t²)。筛选允许正负t两侧；不再给t取负或乘连续gamma。

逐query RMS标准化t（其他方法标准化native score）；Gaussian KDE带宽`0.9 min(sample_SD,IQR/1.34) N^(-1/5)`；median±1.4826MAD内101等距点，density-weighted log-quadratic fit；gamma=max(0,1−b/f)，严格gamma>.5，等价f>2b。有限正pi0>1合法。非法fit=NA；合法空集合、常数预测/响应=0，分别计数。

## 现有结果与必须重算的边界

历史C2/C10 V3执行d9d3a66/f686bd8：17800已产出，17507有效、293NA、4708空/常数0；800条C10 TracInCP/GAS seed123/456双轨缺输入。**该V3的DAS使用t拟合且t聚合，与当前论文定义不同。其DAS Full/BA及依赖它的E-DEL选择结果不能直接搬成当前定义的结果。** 非DAS相同输入/规则结果可复用。需重新生成DAS Full/BA预测、LDS、统计表、E-DEL选法和CI；保留原结果及新结果的独立身份。

AB2/DDPM旧双侧native规则3700条现有输入已验收，3511有效、189拟合失败、361空选择；2500条缺输入。旧DAS在平方空间拟合，仍需使用有符号原项按本次口径重评。AB2原生lambda=1的一次项是匹配源；lambda=.5旧缓存不适用。

AB2 IF在源台账2026-09-22时U0–U2 accepted，1875/5000行；U3–U7尚待收尾。必须等最终5000×100 gen/val完整矩阵、配置及选参验收后补BA，不能以incoming块数量推定全表完成。

## 1. 重新训练/评分（仅当确实缺输入）

按[共用管线阶段1–4](00_SHARED_PIPELINE.md)完成主模型→查询→梯度特征/曲率→64或32个子集重训→query损失→GT→原分数。已有完整分数无需重新训练/采样。文件列表与缺件从 `../evidence/bench_input_properties.json` 读取；该文件是源V3输入的路径与ID属性摘要，不是新口径数值。

## 2. C2/C10已有分数的可移植评价

在仓库根执行。这个完整循环只消费已有方法；缺格写入清单，不填0。`das-presquare`明确传入t、由评价器派生t²，等价显式`--scores t_squared.npy --fit-scores t.npy`。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python - <<'PY'
import json, os, subprocess
from pathlib import Path
root=Path(os.environ['BALDS_DATA'])
methods=['pixel_dot','pixel_cos','clip_dot','clip_cos','grad_dot_T100','grad_cos_T100',
         'tracincp_T100','gas_T100','journey_trak_T100','relative_if_T100','renorm_if_T100',
         'trak_T100','dtrak_T100','das_T100','ekfac_if','fmas_raw']
missing=[]
for ds in ['cifar2_5k','cifar10_v2']:
  for seed in [42,123,456]:
    for track in ['gen','val']:
      key=ds if track=='gen' else ds+'_val'
      masks=root/'subsets'/f'{ds}_masks.pkl'
      responses=root/'results'/f'gt_matrix_fm_{key}_seed_{seed}.npy'
      for method in methods:
        if method=='journey_trak_T100' and track=='val':
          continue
        scores=root/'scores'/method/key/f'seed_{seed}'/'scores.npy'
        paths=[scores,masks,responses]
        absent=[str(p.relative_to(root)) for p in paths if not p.exists()]
        if absent:
          missing.append(dict(dataset=ds,seed=seed,track=track,method=method,missing=absent))
          continue
        output=root/'results/paper/ba'/ds/f'seed_{seed}'/track/method
        command=['balds','evaluate','--scores',str(scores),'--masks',str(masks),
                 '--responses',str(responses),'--output',str(output),'--n-subsets','64','--device','cpu']
        if method=='das_T100':
          command+=['--score-space','das-presquare']
        subprocess.run(command,check=True)
out=root/'results/paper/ba'
out.mkdir(parents=True,exist_ok=True)
(out/'missing_inputs.json').write_text(json.dumps(missing,indent=2)+'\n')
PY
```

CPU KDE可耗时较长；`--device cuda:0`使用相同精确浮点KDE，仅改变计算设备。原N=50000下的正式资源记录不能按C2的16秒pilot直接外推。首次科学重评属于新结果，不在本次迁移中执行。

显式分离拟合/聚合输入的单格版本：

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python - <<'PY'
import os
from pathlib import Path
import numpy as np
root=Path(os.environ['BALDS_DATA'])
p=root/'results/paper/inputs/cifar2_5k/seed_42/gen'
p.mkdir(parents=True,exist_ok=True)
t=np.load(root/'scores/das_T100/cifar2_5k/seed_42/scores.npy').astype(np.float64)
np.save(p/'das_native_squared.npy',t*t)
PY
balds evaluate --fit-scores "$BALDS_DATA/scores/das_T100/cifar2_5k/seed_42/scores.npy" \
  --scores "$BALDS_DATA/results/paper/inputs/cifar2_5k/seed_42/gen/das_native_squared.npy" \
  --masks "$BALDS_DATA/subsets/cifar2_5k_masks.pkl" \
  --responses "$BALDS_DATA/results/gt_matrix_fm_cifar2_5k_seed_42.npy" \
  --n-subsets 64 --device cpu --save-fits \
  --output "$BALDS_DATA/results/paper/ba/cifar2_5k/seed_42/gen/das_explicit"
```

全平台清单生成入口会显式处理C2/C10三seed、AB2单seed/M32、DDPM单seed/M64前缀，以及DDPM balanced validation原ID和100/1000列覆盖。AB2原生lambda1的有符号一次项由已保存的train/query特征及原inverse kernel恢复，不重新采样梯度：

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/tools/prepare_benchmarks.py \
  --data-root "$BALDS_DATA" \
  --datasets cifar2_5k cifar10_v2 cifar2_das artbench2_256 \
  --restore-ab2-das \
  --output "$BALDS_DATA/results/paper/inputs/benchmark_manifest.json"
balds batch --manifest "$BALDS_DATA/results/paper/inputs/benchmark_manifest.json" \
  --data-root "$BALDS_DATA" --output "$BALDS_DATA/results/paper/ba_all_platforms" \
  --device cpu --save-fits
python Codes/tools/summarize_benchmarks.py \
  --manifest "$BALDS_DATA/results/paper/inputs/benchmark_manifest.json" \
  --results "$BALDS_DATA/results/paper/ba_all_platforms" \
  --output "$BALDS_DATA/results/paper/benchmark_tables"
```

该入口为每格声明`score_space=das-presquare`或`native`，训练行、原query IDs、response列ID及模型身份。DDPM验证集标签本身必须可读；AB2 restore必须有实际lambda1特征/核输入，不能用sqrt恢复符号。没有这些数据时报告缺件，不能仅给CLI传一个Q=100就声称匹配原协议。`batch`对缺方法score保留missing，不计入0或替代拟合失败；已有历史V3绝对路径manifest不直接作为新manifest运行。

## 3. 汇总与交付

每格输出`summary.json`、`per_query.json`、`predictions.npz`；`--save-fits`另留诊断。先在每seed的有效query求均值，再跨有效seed等权mean/sampleSD。共享pixel/CLIP validation只报告一份估计；C10 checkpoint方法仅seed42。Full/BA同方法同query配对；不同方法own-valid均值不称严格配对排名。逐query必须含拟合状态、空选择/常数、保留数、Full/BA和query IDs；同时报告实际输入缺失。

最后的`summarize_benchmarks.py`只读逐query结果及manifest身份，输出`benchmark_tables/{summary.json,summary.csv}`，不会重新拟合。pixel/CLIP validation只有逐query指标一致时才合并为共享单次估计；身份或数值不一致须保留差异，不能无条件去重。

历史接受数值与当前重新执行结果必须放不同目录。最终主表回填前需新的DAS表示核对及选择效用验收；本报告不制造新BA数值。

## 历史证据

- `Experiment/experiment.qmd:56–106`，`experiment_v2.md:183–217,249–262`。
- `_Data/reports/ba_das_presquare_2026-09-22/{TASK_JINXU3_C2_C10_EDEL_V3.md,prepare_manifests.py,ACCEPTANCE_V3_20260922.md,VERIFY_V3_ACCEPTANCE.json}`。
- `_Data/results/ba_c2_c10_das_presquare_20260922/inputs/bench_manifest.json`，旧结果根同名。
- `_Data/reports/gpu_wave_five_2026-09-21/{ACCEPTANCE_OVERNIGHT_20260922.md,ACCEPTANCE_C10_NATIVE_HARD_20260922.md}`。
- `_Data/reports/ab2_remaining_2026-09-22/DISPATCH.md`；`ab2_if_joint_2026-09-21/DESIGN_AND_COMMANDS.md`。

这些是源CFA相对路径，历史主机/进程不是公开复现所需配置。
