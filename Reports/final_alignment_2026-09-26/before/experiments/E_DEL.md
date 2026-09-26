# E-DEL：评价选法与真实删除效用

标签：`tab:exp-selection`、`tab:app-delete-existing`、`tab:app-edel-methods`；历史pilot为`tab:app-ba-feasibility`、`tab:app-ba-utility`、`fig:app-ba-fits`。论文`05_Experiment.tex:35–91`、`XY05_EvaluationExperiments.tex:38–44,78–167`。

## 固定设计

C2 unconditional FM seed42，原generation query0–49，N5000，64-mask响应。候选FMAS raw、D-TRAK T100、DAS native square、EK-FAC IF。每候选每query按**原生支持排名**删top300/top1000，原配方重新训练并测原query损失；每预算5随机删除集共享于50queries。BA筛选集合不是实际删除集合。

定义U为删除模型query损失减full模型query损失。Full/BA分别取逐query最大LDS候选，精确平局对并列候选U平均，不用U打破评价平局；两个预算共享同一评价选择。仅在四候选BA全有效的共同集比较。2000次paired query-bootstrap，seed20260920；配对差BA−Full。随机组的250 query/set观测不等于250个独立训练模型。

## 1. 原干预生产链

主模型、查询、分数和原mask响应见[共用管线](00_SHARED_PIPELINE.md)。删除名单从原生分数构造→每名单重训→固定query测损→逐query效用归档。既有干预已完整，不因本次DAS BA拟合定义改变而重训。

公开接口为`balds-run counterfactual {build,retrain,regenerate,analyze}`。显式列出四候选和random：FMAS arm `fmas`读取raw `fmas_raw`，DAS arm `das_native_sq`读取有符号t并平方一次后排名，D-TRAK/IF保留原始有符号值。默认旧双方法列表不用于此表。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
ARMS=fmas,dtrak_T100,das_native_sq,ekfac_if,random
balds-run --data-root "$BALDS_DATA" --device cuda:0 counterfactual regenerate \
  --dataset cifar2_5k --process cfm --uncond --seed 42 --arm reference
for DELETE_K in 300 1000; do
  balds-run --data-root "$BALDS_DATA" --device cpu counterfactual build \
    --dataset cifar2_5k --process cfm --uncond --seed 42 --arm "$ARMS" \
    --k "$DELETE_K" --n-queries 50 --n-random 5
  for ARM in fmas dtrak_T100 das_native_sq ekfac_if random; do
    balds-run --data-root "$BALDS_DATA" --device cuda:0 counterfactual retrain \
      --dataset cifar2_5k --process cfm --uncond --seed 42 --arm "$ARM" \
      --k "$DELETE_K" --rank 0 --world 1
    balds-run --data-root "$BALDS_DATA" --device cuda:0 counterfactual regenerate \
      --dataset cifar2_5k --process cfm --uncond --seed 42 --arm "$ARM" \
      --k "$DELETE_K" --rank 0 --world 1
  done
  balds-run --data-root "$BALDS_DATA" --device cuda:0 counterfactual analyze \
    --dataset cifar2_5k --process cfm --uncond --seed 42 --arm "$ARMS" --k "$DELETE_K"
done
```

`analyze`默认测loss，不传`--no-loss`。Full reference只生成一次。若指定FMAS固定rho，以`build --fmas-rho 0.01`读取对应raw缓存；不可从EB/shrink分数补出raw。迁入历史干预后直接用已有效用即可，无需再次重训。

新运行summary包含原query ID、loss_cf、loss_base、dloss。下面导出native效用表，与原历史表分开归档：

```bash
export BALDS_DATA="$PWD/_Data"
python - <<'PY'
from pathlib import Path
import os,json,csv
from balds.workflows.counterfactual import arm_id
root=Path(os.environ['BALDS_DATA'])
out=root/'results/paper/deletion/native_utilities.tsv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
  w=csv.DictWriter(f,fieldnames=['k','query','method','transform','utility'],delimiter='\t')
  w.writeheader()
  for k in [300,1000]:
    source=root/'counterfactual/cifar2_5k/analysis/seed_42'/f'summary_k{k}.json'
    data=json.loads(source.read_text())
    for arm,method in [('fmas','fmas_raw'),('dtrak_T100','dtrak_T100'),('das_native_sq','das_native_sq'),('ekfac_if','ekfac_if')]:
      records=data['records'][arm_id(arm,k)]
      assert len(records)==50 and {r['query'] for r in records}==set(range(50))
      for r in records:
        w.writerow(dict(k=k,query=r['query'],method=method,transform='native',utility=r['dloss']))
PY
```

历史四候选效用汇总位于`results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv`：800行=两预算×四候选×native/signed-square×50query；native部分400行。保留原干预mask/meta、query编号和原始loss证据，避免不同格式导出误配。

历史DAS300曾用旧格式导出；DAS1000、IF300/1000跨不同恢复批次归档。历史CPU汇总代码1f2e6d1核对名单后合表；公开新生产链与历史恢复过程分开记录。

## 2. 按当前DAS定义重算评价

C2三种子/双轨主表重评见E-BENCH。删除面板使用seed42的前50个generation query。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python - <<'PY'
from pathlib import Path
import os,json,subprocess
root=Path(os.environ['BALDS_DATA'])
out=root/'results/paper/deletion'
methods={'fmas_raw':'fmas_raw','dtrak_T100':'dtrak_T100','das_native_sq':'das_T100','ekfac_if':'ekfac_if'}
evaluations={}
ids=','.join(map(str,range(50)))
for utility_key,source in methods.items():
  dest=out/'evaluations'/utility_key
  command=['balds','evaluate','--scores',str(root/'scores'/source/'cifar2_5k/seed_42/scores.npy'),
    '--masks',str(root/'subsets/cifar2_5k_masks.pkl'),
    '--responses',str(root/'results/gt_matrix_fm_cifar2_5k_seed_42.npy'),
    '--n-subsets','64','--query-ids',ids,'--output',str(dest),'--device','cpu']
  if source=='das_T100': command+=['--score-space','das-presquare']
  subprocess.run(command,check=True)
  evaluations[utility_key]=str(dest)
out.mkdir(parents=True,exist_ok=True)
(out/'evaluations.json').write_text(json.dumps(evaluations,indent=2)+'\n')
PY
balds deletion --evaluations "$BALDS_DATA/results/paper/deletion/evaluations.json" \
  --utilities "$BALDS_DATA/results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv" \
  --output "$BALDS_DATA/results/paper/deletion/selection"
```

`das_native_sq`是原效用表的候选键，不能用它暗示旧V3已正确平方。本次评价的fit=t、aggregate=t²由`--score-space das-presquare`决定；原干预仍native t²排名。若使用新产效用，替换`--utilities`为`results/paper/deletion/native_utilities.tsv`。输出交付共同IDs、fit failures、每query两种选法和实际效用、两预算均值/配对差/CI、精确ties与选择一致率；相对增益由配对差除Full所选效用计算，Full为零则不定义。

## 3. 历史数字仅作溯源

旧V3：199/200有效、共同49（排除q17）、22/49一致；删300效用差+.000278251208，CI[−.000095739702,.000684364056]；删1000差+.000311749930，CI[−.000707967176,.001385433250]。**它们基于DAS t/t，不能当本次t/t²结果，且两CI均跨0。** 重新执行可能改变DAS LDS、选择与效用；不保留旧选择来维持原结论。

逐候选原native干预本身不变：50query均值FMAS .010238/.019353、D-TRAK .008121/.013191、DAS .007500/.010899、IF .009952/.018759；随机 .001837/.002520（300/1000）。这些描述的是实际干预，不依赖BA表示。

## 4. 附录早期BA pilot

FMAS/D-TRAK、query0–19、40fits；旧规则只选正score且拒绝pi0>1，有效39、共同19。FMAS valid19/20，D-TRAK20/20。该规则由单独`--pilot`开关保留，不改变主算法默认。LDS@5%为稳定排序的top250，不能用E2的top300/1000替代。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
for METHOD in fmas_raw dtrak_T100; do
  balds evaluate --scores "$BALDS_DATA/scores/$METHOD/cifar2_5k/seed_42/scores.npy" \
    --masks "$BALDS_DATA/subsets/cifar2_5k_masks.pkl" \
    --responses "$BALDS_DATA/results/gt_matrix_fm_cifar2_5k_seed_42.npy" \
    --query-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --n-subsets 64 --pilot --save-fits --device cpu \
    --output "$BALDS_DATA/results/paper/pilot/$METHOD"
done
```

以上重新产生Full/BA逐query与拟合参数。历史39/40及q17 pi0>1的失败必须按该旧规则理解，不能套当前双侧规则解释。固定q0/q1四面板从已保存fit重绘，不挑选其他query；原图及其源参数在`results/ba_lds_feasibility_20260920/j3-a1/`。现稿已复制的pilot图不需要重拟合。

pilot的二候选Full/5%/BA效用表与当前四候选E-DEL不同；`balds deletion`严格要求四候选×50，不用于此20query补充。下面按两方法BA共同有效IDs、stable top250、每预算相同原native U及精确ties均值生成三列结果，不添加历史未报告的显著性检验。

```bash
export BALDS_DATA="$PWD/_Data"
python - <<'PY'
from pathlib import Path
import os,json,csv,pickle
import numpy as np
from balds.evaluation.lds import compute_lds,predicted_influence
from balds.evaluation.background import selected_utility
from balds.workflows.benchmark import write_json
root=Path(os.environ['BALDS_DATA']);out=root/'results/paper/pilot'
methods=['fmas_raw','dtrak_T100'];records={};table=[]
with (root/'subsets/cifar2_5k_masks.pkl').open('rb') as f: masks=pickle.load(f)[:64]
y=np.load(root/'results/gt_matrix_fm_cifar2_5k_seed_42.npy')[:64,:20]
for method in methods:
  a=np.load(root/'scores'/method/'cifar2_5k/seed_42/scores.npy')[:,:20]
  head=np.zeros_like(a,dtype=bool)
  np.put_along_axis(head,np.argsort(-a,axis=0,kind='stable')[:250],True,axis=0)
  head_lds,_=compute_lds(y,predicted_influence(np.where(head,a,0),masks,list(range(64))))
  rows=json.loads((out/method/'per_query.json').read_text())
  for row,h in zip(rows,head_lds):
    row['head5_lds']=float(h);records[method,row['query_id']]=row
  valid=[r for r in rows if r['ba_lds'] is not None]
  table.append(dict(method=method,n_valid=len(valid),**{key:np.mean([r[key] for r in valid]) for key in ['full_lds','head5_lds','ba_lds']}))
common=[q for q in range(20) if all(records[m,q]['ba_lds'] is not None for m in methods)]
with (root/'results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv').open() as f:
  utilities={(r['method'],int(r['query']),int(r['k'])):float(r['utility']) for r in csv.DictReader(f,delimiter='\t') if r['transform']=='native'}
choices=[]
for k in [300,1000]:
  for q in common:
    for metric in ['full_lds','head5_lds','ba_lds']:
      u,winners=selected_utility([records[m,q][metric] for m in methods],[utilities[m,q,k] for m in methods],methods)
      choices.append(dict(k=k,query_id=q,metric=metric,utility=u,winners=winners))
summary=[dict(k=k,metric=metric,n_valid=len(common),utility=np.mean([r['utility'] for r in choices if r['k']==k and r['metric']==metric])) for k in [300,1000] for metric in ['full_lds','head5_lds','ba_lds']]
write_json(out/'pilot_tables.json',dict(lds=table,common_queries=common,utility_summary=summary,utility_per_query=choices))
PY
```

旧`lds_comparison.tsv`、`per_query.tsv`、`utility_comparison.tsv`、`utility_per_query.tsv`及`utility_source.tsv`仍是附录历史表的直接数值证据；新导出与历史文件分开保存。

## 历史证据

- `_Data/reports/lds_e1e5_wave1_2026-09-17/{TASK_XUCHANG3_E2_V1.md,E2_THREEHOST_RECOVERY_20260917.md}`：原干预与恢复，6ae4d89修复。
- `_Data/reports/lds_nextwave_2026-09-20/{TASK_JINXU3_CPU_V1.md,ACCEPTANCE_J3_CPU_j3-a1.md}`：1f2e6d1的800行原始效用/选择表。
- `_Data/reports/ba_das_presquare_2026-09-22/{ACCEPTANCE_V3_20260922.md,VERIFY_DAS_EDEL_IF.json}`：旧V3重新选择。
- `_Data/reports/ba_lds_feasibility_2026-09-20/{RESULTS.md,RUN_ACCEPTANCE_j3-a1.md}`：旧pilot，5d574ac，CPU16.08秒。
