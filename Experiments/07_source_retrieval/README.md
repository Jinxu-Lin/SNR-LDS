# 终稿E-SOURCE：reviewed500来源检索

有效标签：主文Table2 `tab:exp-retrieval`；附录Tables13/14/15 `tab:app-injection-pairs`、`tab:app-retrieval-by-concept`、`tab:app-retrieval-ddpm-by-concept`。两过程CFM/DDPM均使用最终reviewed500，100val选参、400test报告；各13方法。旧CFM common200、旧CFM-v1 400不是终稿依赖。以`Paper/Sections/XY05_EvaluationExperiments.tex`、最终query包及9月24验收为准。

## 固定数据与度量

CIFAR-10条件生成、seed42，50000训练行中4%=2000行替换为CIFAR-100外来概念，每概念200行，保留host标签及原训练行顺序。位置与概念抽样seed20260914。映射如下，Table13记录同一映射。

| Host | 外来concept | Tier |
|---|---|---|
| airplane | sunflower | coarse |
| automobile | castle | coarse |
| bird | butterfly | coarse |
| cat | leopard | fine |
| deer | cattle | fine |
| dog | wolf | fine |
| frog | mushroom | coarse |
| horse | camel | fine |
| ship | skyscraper | coarse |
| truck | tractor | fine |

各concept50审核query：10val、40test。val只用于within-host pooled AP选lambda/rho；test在完整50000训练行排序，主指标global AP及Recall@200（真source分母200）。先概念内query平均，再十概念等权；fine/coarse各五概念。稳定训练index打破检索分数ties。DAS以native t²排名，不能直接对signed t排序，也不使用SNR选择。

13方法为FMAS raw、EK-FAC IF、D-TRAK、DAS、TRAK、Relative IF、Renormalized IF、gradient dot/cos、pixel dot/cos、CLIP dot/cos。Journey/TracInCP/GAS及退役PW/AbU/NDA不在本实验中。

CFM最终v2替换了旧500中的280张；相同数量、相同0..499整数ID不代表相同图像版本。必须同时保存审核CSV、query张量、metadata、注入source行及打包版本。9月23撤回之后的9月24closeout接受最终query/score；不要依据更早撤回或common200重新拼装。

## 1. 已迁工件的全部13方法CPU复算

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
python Codes/tools/replay_retrieval.py \
  --data-root "$BALDS_DATA_ROOT" \
  --output "$BALDS_DATA_ROOT/results/paper/retrieval"
```

读取各过程11方法50000×500 native矩阵与两曲率方法最终400列矩阵，原100val重新核验所选参数及候选AP；不会加载模型或生成query。有本地论文时可另加 `--paper Paper` 核对TeX打印舍入区间，不写Paper。输出`{cfm,ddpm}/{method}.json`含400逐query、十概念、两tier、均值及选参核对；`summary.json`汇总。可用`--manifest`指定另一个完整归档寻址JSON，用`--train-labels`提供50000行标签数组。默认配置仅引用当前接受工件。

先前的数组重放核验覆盖26个过程/方法，Tables2/14/15共572个数值落在Paper打印舍入区间；两方法两过程共30份val候选矩阵选参一致。记录见 [核验报告](evidence/retrieval-verification.json)。该检验与重新训练或重采样梯度不同，本次整理没有重跑评分。

## 2. 数据、模型与审核query生产

已有接受工件时跳过本节。新训练产生新模型身份，不应沿用旧论文数值。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
balds-run --data-root "$BALDS_DATA_ROOT" --device cpu inject build --dataset cifar10_inj4
for PROCESS in cfm ddpm; do
  balds-run --data-root "$BALDS_DATA_ROOT" --device cuda:0 train \
    --dataset cifar10_inj4 --process "$PROCESS" --seed 42
done
```

主模型沿C10纯条件p_uncond0配方，共用上游见[训练与数据线](../00_pipeline/README.md)。已审核query是外部研究输入：`inject detector-train`→`inject mine`（CFM Euler100，DDPM DDIM50 eta0）→`inject review-export`→人工decision→`inject queries --review CSV --version VERSION`。历史最终review CSV和query身份必须随数据迁入，不能用自动“accept”记录伪造审核结果。当前代码保留上述生产能力；原detector版本、实际mine参数由已接受原manifest提供，不能用旧r50v1示例补齐未知历史命令。Detector曾使用all-real配置，需记录实际数据身份；不能额外断言其训练集与source隔离。

## 3. 11个标准方法：特征、score、val选参与test

下列完整执行线消费当前审核500包；query更新时，旧query features失效，train features仅在模型及训练身份完全相同时复用。CFM与DDPM的process必须贯穿各步骤。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
for PROCESS in cfm ddpm; do
  for FEAT in das_T100 dtrak_T100 trak_T100; do
    balds-run --data-root "$BALDS_DATA_ROOT" --device cuda:0 featurize \
      --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --query-type inject \
      --feat "$FEAT" --split both --no-error-weight
  done
  for FEAT in pixel clip; do
    balds-run --data-root "$BALDS_DATA_ROOT" --device cuda:0 featurize \
      --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --query-type inject \
      --feat "$FEAT" --split both
  done
  for METHOD in dtrak_T100 das_T100 trak_T100 relative_if_T100 renorm_if_T100 grad_dot_T100 grad_cos_T100 pixel_dot pixel_cos clip_dot clip_cos; do
    balds-run --data-root "$BALDS_DATA_ROOT" --device cuda:0 score \
      --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --query-type inject --method "$METHOD"
    balds-run --data-root "$BALDS_DATA_ROOT" --device cpu inject evaluate \
      --dataset cifar10_inj4 --process "$PROCESS" --seed 42 --method "$METHOD" --split test --k 200
  done
done
```

接受CFM lambda：DTRAK2、DAS.1、TRAK/Relative/Renorm .2；DDPM对应.5/.05/.05/.05/.05。`inject_val`只看100val选参；无可调kernel的similarity不把通用占位lambda称作实质调优。不得把旧CFM .5参数搬入最终v2。

## 4. 两曲率方法：完整500新块生产与装配

FMAS、IF共享已接受曲率；两pass各125，score train/query各MC250，query_chunk100，row_chunk1000，grad_chunk125，bfloat16 query坐标，train-mode/hflip。原query ID参与RNG；不能把分块local0..99当原ID。FMAS val选rho1；IF CFM1e-9/DDPM1e-12。

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
for PROCESS in cfm ddpm; do
  BLOCKS="$BALDS_DATA_ROOT/results/paper/retrieval_blocks/$PROCESS"
  ASSEMBLED="$BALDS_DATA_ROOT/results/paper/retrieval_new/$PROCESS"
  balds-run --data-root "$BALDS_DATA_ROOT" --device cuda:0 ekfac fit \
    --dataset cifar10_inj4 --process "$PROCESS" --seed 42
  for METHOD in fmas_raw ekfac_if; do
    python Codes/tools/score_retrieval.py prepare \
      --input-data-root "$BALDS_DATA_ROOT" --output-root "$BLOCKS" \
      --process "$PROCESS" --method "$METHOD" --query-chunk 100
    python Codes/tools/score_retrieval.py run \
      --input-data-root "$BALDS_DATA_ROOT" --output-root "$BLOCKS" \
      --process "$PROCESS" --method "$METHOD" --query-chunk 100 --device cuda:0 \
      --rank 0 --world 1 --row-rank 0 --row-world 1 --cpu-threads 2
    python Codes/tools/score_retrieval.py status \
      --input-data-root "$BALDS_DATA_ROOT" --output-root "$BLOCKS" \
      --process "$PROCESS" --method "$METHOD" --query-chunk 100
    python Codes/tools/assemble_retrieval.py \
      --data-root "$BALDS_DATA_ROOT" --sources "$BLOCKS" --output "$ASSEMBLED" \
      --process "$PROCESS" --method "$METHOD"
  done
done
```

有接受曲率时跳过fit。prepare核验checkpoint/factors/query/source-meta/cache；不会隐式拟合曲率。query_chunk不是MC预算；prepare/run/status具有同一明确身份。可用rank/world分query、row-rank/row-world分训练行，每分片均要完成；此处单worker覆盖全部。省略selection即完整100val+400test；若提供selection只能是完整400，旧common200被拒绝。

新block布局保留原500ID；历史val100/test100lane各自布局保持原状，不能伪装成新块。公开装配选择val参数、仅在400test报告；历史论文结果直接用第1节replay，不用新装配器混读不同批次。新的训练/score结果通过显式archive manifest接入replay时必须保留其query/model身份，不能覆盖接受manifest后称已验收。

## 接受来源与数值边界

CFM标准11方法：`scores/<method>/cifar10_inj4_inject/seed_42/`及`results/cfm_reviewed500_20260923/verified_summary.json`；曲率：`results/cfm_test400_20260923/analysis/{fmas_raw,ekfac_if}/{result.json,test400_scores.npy}`，val100保留`results/cfm_val100_20260923/incoming/`。DDPM对应最终review-v2 native score及已验收IF/FMAs val/test产物，准确地址由`Codes/src/balds/workflows/retrieval_replay.py:archive_manifest()`以相对数据根登记。接受回执：`_Data/reports/experiment_closeout_2026-09-24/ACCEPTANCE.md`。

最终CFM FMAS AP/Recall=.25310883645654325/.2767375，IF约.23498299/.260725。fine/coarse统计中，FMAS AP为.0494322394/.4567854335、Recall为.0899/.463575；IF AP为.0460366060/.4239293689、Recall为.08825/.4332。数值必须使用最终审核query版本，不能混用先前查询包。
