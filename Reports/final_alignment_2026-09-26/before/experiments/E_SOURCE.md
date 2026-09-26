# E-SOURCE：受控来源检索

标签：主文`tab:exp-retrieval`，附录`tab:app-retrieval-global`、`tab:app-retrieval-concepts`、`tab:app-retrieval-cfm400`、`tab:app-retrieval-host`、`tab:app-retrieval-ddpm-v2`。论文`05_Experiment.tex:201–239`，`XY05_EvaluationExperiments.tex:261–298`。

## 三套查询集合必须分开

CIFAR-10注入4%=2000图，十个foreign概念各200 source，保留50,000行顺序和原host labels；concept query经detector和人工审核，过程独立于attribution scores。主指标每query对完整50,000训练图稳定排序，Recall@200分母200，AP是global AP；先概念内平均，再十概念等权。within-host AP/AUROC/precision@200是单独辅助量。

foreign来源CIFAR-100；替换位置seed和concept抽样seed均20260914，builder为`cifar10-inject-v1`。固定映射：

| CIFAR-10 host label | 外来concept |
|---|---|
| airplane | sunflower |
| automobile | castle |
| bird | butterfly |
| cat | leopard |
| deer | cattle |
| dog | wolf |
| frog | mushroom |
| horse | camel |
| ship | skyscraper |
| truck | tractor |

| 集合 | 查询 / 方法 | 当前用途 | 选参 |
|---|---|---|---|
| CFM common200 | test每概念20，共200；13方法 | 当前主文与完整附录、逐概念表 | 独立100val选pool AP；test IDs按seed20260917预定 |
| CFM original400 | test每概念40，共400；11方法 | 历史附录global及within-host表 | 原100val协议 |
| DDPM review-v2 | 100val+400test，十概念各50总查询；11方法 | 更新后的DDPM附录 | 只用100val，旧DDPM query分数已退役 |

完整13方法=FMAS raw、IF与D-TRAK/DAS/TRAK/RelativeIF/RenormIF/gradient dot/cos/pixel dot/cos/CLIP dot/cos。Journey、TracInCP/GAS和三新增方法不属于此13方法表。DAS在retrieval使用native t²排名，本次BA拟合规则不改变retrieval。

## 1. 从数据构建到审核query

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
balds-run --data-root "$BALDS_DATA" --device cpu inject build --dataset cifar10_inj4
for PROCESS in cfm ddpm; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 train --dataset cifar10_inj4 --process "$PROCESS" --seed 42
done
```

主模型沿C10纯条件p_uncond0配方。Source metadata为`subsets/cifar10_inj4_inject_meta.json`，包含原始source和替换行。对精确复现应迁入**实际审核query包与原CSV决策**，而非重新审核后沿用旧数字。新生成/审核是新数据版本。

历史公开能力：`balds-run inject detector-train` → `inject mine`（CFM Euler100，DDPM DDIM50 eta0）→ `inject review-export` → 人工填写decision → `inject queries --review CSV --version VERSION`。当前审核v2的完整实际mine/detector版本、source query身份由query包及原回执提供；本文不把旧README中的`r50v1/v1`示例伪装成v2现场命令。**缺少已迁入的审核CSV/原query包时，无法从论文中的200/400查询数量唯一还原筛选结果。** 人工审核是明确外部输入，不自动生成“accept”记录。

早期文本称detector训练图与注入图隔离，后续生产存在all-real detector配方；必须保留实际detector的config/meta和版本，而非按论文概述猜测训练集。现稿无需ES-L发生率或ES-D删除实验；它们不是retrieval前置。

## 2. projected特征 → 11方法评分 → retrieval指标

以下命令消费已审核、已划分val/test的query包；CFM用原500总查询，DDPM用review-v2的500总查询。train特征已有时可只计算`--split query`，不可覆盖成另一查询版本。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
for PROCESS in cfm ddpm; do
  for FEAT in das_T100 dtrak_T100 trak_T100; do
    balds-run --data-root "$BALDS_DATA" --device cuda:0 featurize --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --query-type inject --feat "$FEAT" --split both --no-error-weight
  done
  for FEAT in pixel clip; do
    balds-run --data-root "$BALDS_DATA" --device cuda:0 featurize --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --query-type inject --feat "$FEAT" --split both
  done
  for METHOD in dtrak_T100 das_T100 trak_T100 relative_if_T100 renorm_if_T100 grad_dot_T100 grad_cos_T100 pixel_dot pixel_cos clip_dot clip_cos; do
    balds-run --data-root "$BALDS_DATA" --device cuda:0 score --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --query-type inject --method "$METHOD"
    balds-run --data-root "$BALDS_DATA" --device cpu inject evaluate --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --method "$METHOD" --split test --k 200
  done
done
```

`inject_val` selector只最大化100val的pool-AP，不能看test选lambda；输出记录val curve和test结果。上述`inject evaluate`得到400test基础表；它不直接得到CFM common200。CFM原11方法接受lambda：D-TRAK2、DAS.1、TRAK/Relative/Renorm .5；其余无kernel可调参数（表中.01是通用占位网格，不称其经过有效调优）。DDPM原旧表lambda和指标不能带入review-v2，必须读v2 score/meta。

## 3. CFM曲率 → 固定共同200分块评分

`ekfac fit --dataset cifar10_inj4 --process cfm --seed 42`生成共享曲率；两pass125、每侧score MC250，query chunk100、row chunk1000、grad chunk125、query coords bfloat16，train mode/hflip。精确IDs与参数见[接受identity属性](../evidence/retrieval_common200_properties.json)。FMAS和IF只对100val+200test指定原ID评分，所有50000训练行，query/row分块不能改变MC或IDs。FMAS rho通过100val选为1；IF十档lambda通过100val选为1e-8。

历史六单元：FMAS F0=100val、F1/F2=各100test；IF I0=100val、I1a/I1b=前100test互补train行、I2=后100test。IF现有150原子块齐套，不需重跑。分块评分的具体参数/原ID清单来自源`lds_fivegpu_2026-09-20/TASK_*_E5_*.md`及package.json。通用`ekfac score --query-type inject`处理标准审核query包，不能仅以Q=300替代指定的100val+200test原ID排序。

公开`score_retrieval.py`保留原query ID与RNG规则，提供本地prepare/run/status，不依赖历史机器。以下为从已有模型/审核query开始的完整曲率及新块生产命令；复用已验收曲率时跳过fit，复用全部已验收块时直接进入第4节。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
balds-run --data-root "$BALDS_DATA" --device cuda:0 ekfac fit \
  --dataset cifar10_inj4 --process cfm --seed 42
for METHOD in fmas_raw ekfac_if; do
  python Codes/tools/score_retrieval.py prepare \
    --input-data-root "$BALDS_DATA" --output-root "$BALDS_DATA/results/paper/retrieval_blocks" \
    --selection results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json \
    --method "$METHOD" --query-chunk 100
  python Codes/tools/score_retrieval.py run \
    --input-data-root "$BALDS_DATA" --output-root "$BALDS_DATA/results/paper/retrieval_blocks" \
    --selection results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json \
    --method "$METHOD" --query-chunk 100 --device cuda:0 \
    --rank 0 --world 1 --row-rank 0 --row-world 1 --cpu-threads 2
  python Codes/tools/score_retrieval.py status \
    --input-data-root "$BALDS_DATA" --output-root "$BALDS_DATA/results/paper/retrieval_blocks" \
    --selection results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json \
    --method "$METHOD" --query-chunk 100
  python Codes/tools/assemble_retrieval.py \
    --data-root "$BALDS_DATA" --method "$METHOD" \
    --sources "$BALDS_DATA/results/paper/retrieval_blocks" \
    --selection results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json \
    --comparison-manifest results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_common200.json \
    --output "$BALDS_DATA/results/paper/retrieval_common200_new"
done
```

prepare仅核对已有checkpoint、曲率、query、注入meta、selection与数据缓存，不生成曲率。run固定MC250/250，`query-chunk=100`是内存分块列数，不是MC预算。两个方法共用输出根但工件按方法隔离；status核对全部块后，CPU装配才进行val选参与test评价。表格导出使用第4节相同Python逻辑，将根改为`retrieval_common200_new`，不能与历史块混装。

## 4. 已有块的portable CPU装配

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/tools/assemble_retrieval.py \
  --data-root "$BALDS_DATA" --method ekfac_if \
  --sources "$BALDS_DATA/results/e5_common200_20260920/incoming/xc2_I0" \
            "$BALDS_DATA/results/e5_common200_20260920/incoming/xc0_I1a" \
            "$BALDS_DATA/results/e5_common200_20260920/incoming/xc0_I1b" \
            "$BALDS_DATA/results/e5_common200_20260920/incoming/xc2_I2" \
  --selection results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json \
  --comparison-manifest results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_common200.json \
  --output "$BALDS_DATA/results/paper/retrieval_common200"
python Codes/tools/assemble_retrieval.py \
  --data-root "$BALDS_DATA" --method fmas_raw \
  --sources "$BALDS_DATA/results/e5_common200_20260920" \
            "$BALDS_DATA/results/e5_common200_20260920/incoming/xc0_F1" \
            "$BALDS_DATA/results/e5_common200_20260920/incoming/xc3_F2" \
  --selection results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_test200_indices.json \
  --comparison-manifest results/lds_e1e5_wave1_20260917/jinxu3/e5_cfm_common200.json \
  --output "$BALDS_DATA/results/paper/retrieval_common200"
python - <<'PY'
from pathlib import Path
import os,json,csv
root=Path(os.environ['BALDS_DATA'])/'results/paper/retrieval_common200'
a=json.loads((root/'ekfac_if/result.json').read_text())
b=json.loads((root/'fmas_raw/result.json').read_text())
assert a['complete'] and b['complete'] and a['query_ids']==b['query_ids']
methods=dict(a['old11']); methods.update(ekfac_if=a['test'],fmas_raw=b['test'])
assert len(methods)==13
value={'query_ids':a['query_ids'],'methods':methods,'if_best_lam':a['best_lam'],'fmas_best_lam':b['best_lam']}
(root/'common200_table.json').write_text(json.dumps(value,indent=2)+'\n')
with (root/'common200_table.tsv').open('w') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['method','n_test','global_AP','global_Recall_at_200'])
  for name,r in methods.items():
    w.writerow([name,r['n_queries'],r['means']['ap_global'],r['means']['recall_at_k_global']])
PY
```

目录名中的旧机器标识仅是已归档relative layout，不会联系远端主机。上述输入经迁移报告映射；若采用不同目标布局，修改`sources`而不是在代码中恢复服务器名。

## 已验收结果与证据

CFM共同200已接受AP/Recall：FMAS .27229750/.29545000，IF .24902537/.27657500，D-TRAK .25370116/.26952500，DAS .22893309/.25147500。完整13方法见[精简证据表](../evidence/cfm_common200_accepted.tsv)。只报点估计，不添加未检验显著性。IF150块、1000val选参单元、2600test逐query指标独立复核；原产物`results/cfm_if_common200_closeout_20260922/`。

DDPM review-v2 11份50000×500矩阵与11×400指标已验收，来源`results/score_closeout_20260921/ddpm/verified_summary.json`。CFM400来源各方法`inject_result.json`与Paper的`retrieval_global_sources.json`。

源证据：`reports/cfm_if_closeout_2026-09-22/{ACCEPTANCE_20260922.md,VERIFY_ACCEPTANCE.json,TASK_JINXU3_CFM_IF_CPU_V1.md}`；`reports/closeout_dispatch_2026-09-21/ACCEPTANCE_CPU_CLOSEOUT_20260921.md`；`reports/inject_2026-09-14/ACCEPTANCE_INJECT_C10_FEAT_T100_CFM_20260917.md`。CPU最终IF收尾f686bd8，35.680秒，不包含上游GPU成本。
