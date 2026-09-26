# 终稿删除实验：全局选法与实际干预

标签：Fig3(b) `fig:snr-diagnostics`、Table12 `tab:app-delete-existing`、Table13 `tab:exp-selection`。依据 `Paper/Sections/XY05_EvaluationExperiments.tex:43–47,86–142`。旧逐query四候选择法与positive-only pilot已退役，原报告在日期快照保留。

## 两套查询集合及精确选择规则

**选方法**：C2 FM gen三模型seed42/123/456，每个seed先取四核心FMAS raw/D-TRAK/DAS/EK-FAC IF共同有效SNR fit的query。Full/SNR分别在同一集合求各方法均值，再对3个seed等权平均，按最高benchmark mean各选一个方法。接受共同query数量83/75/81；Full选DAS、SNR选FMAS。这里不能只读seed42前50，不能按每个query切换候选。

**测干预**：原seed42 generation q0–49，N5000，native top300/top1000每候选每query删除后按原配方重训；两预算各5个random集合。全部50query用于既定FMAS/DAS差，不再受各query fit是否有效影响。SNR选中集合不是实际删除集合。

U=删除模型query loss−full模型query loss。2000次paired query-bootstrap，seed20260920；同一query抽样保留方法/预算。精确benchmark ties保持并列并平均并列候选效用，不用U破同分。随机250个query/set观测不是250次重训。

## 1. 原干预生产链

主模型、查询、分数和原mask响应见[共用管线](00_SHARED_PIPELINE.md)。删除名单从原生分数构造→每名单重训→固定query测损→逐query效用归档。既有干预已完整，不因SNR-LDS评价规则改变而重训。

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

## 2. 从SNR主表选择方法并读取原效用

先运行[E-BENCH](E_BENCH.md)得到三seed generation完整面板；`postprocess`已经执行这一步，也可单独调用：

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
balds deletion \
  --evaluations "$BALDS_DATA_ROOT/results/paper/snr/panels" \
  --utilities "$BALDS_DATA_ROOT/results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv" \
  --output "$BALDS_DATA_ROOT/results/paper/snr/edel"
```

目录入口读取`cifar2_5k_s{42,123,456}_gen/panel.json`四候选记录；当前字段为`snr_lds`，不读旧BA字段。原效用表只取native400行。若从第一节重新生产干预，显式改用新`native_utilities.tsv`，与历史接受表分开。

输出`edel_selection.json`包含逐seed有效query、Full/SNR候选均值、全局选法、两预算50query效用及点态区间。接受Full/SNR DAS=46.70/43.01%、FMAS=39.32/45.04%；这不是主表各方法own-valid均值。预算300的FMAS−DAS=.002737137，95%CI[.002130861,.003396835]；1000为.008453480，[.007099821,.010070689]。均值相对增益36.493249%/77.560305%，不是逐query百分比均值；胜率分别49/50和50/50。

## 3. 图像变化与随机参照

generation原种子165、Euler100、原batchQ100，原图统一取`generations/cifar2_5k/cfm_uncond/seed_42/samples.pt`前50；不能混各机器旧reference距离。FMAS/DAS共200份regen，random共10模型/500份query regen。

pixel L2是`clamp((samples+1)*127.5,0,255).uint8 / 255`图像展平后的欧氏距离；CLIP使用`openai/clip-vit-base-patch32`图像embedding cosine。历史归档为distance=1−cosine时，终稿需转换为cosine，不能只改列名。Random先在同query平均5集合，再平均50query。

正式来源：`results/fmas_das_visual_20260922/{summary.json,per_query.tsv,image_embeddings.npz}`与`results/fmas_das_random_visual_20260922/{summary.json,random_per_set_query.tsv,random_per_query.tsv,image_embeddings.npz}`；原regen和loss路径由其记录给出。固定q0–7图仅是辅助展示，不据效果重新挑query。Table12仍使用全部四候选native损失及CI。主图输出从这些接受统计生成；更大L2/更小CLIP说明图像变化，不直接说明质量下降。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
FIGURE_OUT="$BALDS_DATA_ROOT/results/paper/figures"
python Codes/tools/render_deletion_tables.py \
  --utilities "$BALDS_DATA_ROOT/results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv" \
  --selection "$BALDS_DATA_ROOT/results/paper/snr/edel/edel_selection.json" --output "$FIGURE_OUT"
python Codes/tools/render_deletion_visual.py --data-root "$BALDS_DATA_ROOT" --output "$FIGURE_OUT"
```

分别生成Tables12/13及Fig3(b)TeX，读取既有native/random逐query数据。随机图像统计重新按5set→query→50均值聚合，不从Paper抄表。

## 来源及边界

- `Reports/snr_lds_2026-09-25/ACCEPTANCE_AND_PAPER_ADOPTION_A4.md`：全局benchmark选法及原50query差。
- CFA源`_Data/reports/fmas_das_visual_2026-09-22/{TASK_JINXU3_FMAS_DAS_VISUAL_CPU_V1.md,ACCEPTANCE_20260922.md}`：共同原图、200份regen、loss/L2/CLIP配对分析。
- CFA源`_Data/reports/fmas_das_random_2026-09-22/RESULTS.md`：10random模型/500query-set观测及逐query聚合。

旧BA V3的common49、22/49选法一致率和微小效用差不再是本实验输出。原干预可复用，但评价选择规则按当前终稿执行。本报告不宣称本轮重新训练或生成过图像。
