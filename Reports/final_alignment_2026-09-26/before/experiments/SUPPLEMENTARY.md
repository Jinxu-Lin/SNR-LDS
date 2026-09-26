# S1–S4：附录聚合补充

这些保留现稿既有结果，不是已退役的D1–D4。论文`XY05_EvaluationExperiments.tex:300–533`。上游复用[E-DEL](E_DEL.md)原干预和[共用管线](00_SHARED_PIPELINE.md)原score/response；本实验不使用BA筛选、不新增训练。

| ID | 论文标签 | 固定设计 |
|---|---|---|
| S1 | `tab:app-selection-complete`, `tab:app-deletion-pairs` | C2seed42 gen50；三候选FMAS/D-TRAK/DAS与加IF的四候选；head300/1000，与删除预算相同即6%/20% |
| S2 | `tab:app-transform-utility-complete` | native score施加sign(s)*abs(s)^2，保持原排名、原干预、原U；DAS从已平方native出发 |
| S3 | `tab:app-aggregation-controls`, `tab:app-count-controls` | C2/C10三seed，AB2单seed；四核心native；fixed5% positive/soft/unit与DAS centered/count |
| S4 | `tab:app-extended-c2`, `tab:app-extended-c10`, `tab:app-extended-ab2` | 其余传统方法的已有full/head；C10/AB2 head来自接受E4，C2缺head不新增填补 |

## S1–S2：选法与同名单控制

每query以full或相应head LDS选候选，用同一已测U计算差；ties平均并列候选效用。配对bootstrap2000，seed20260920，原50query不因BA fit丢失而删减。两个head分别300/1000，不能改写为5%或BA。

方法对诊断对每对候选跨query比较LDS差与U差，报告Pearson/Spearman及符号一致；一侧tie半分，两侧tie按原定义；常量差导致未定义相关保留NA。S2只改变评价数值，删除集合和U完全相同。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/tools/aggregation_controls.py deletion \
  --data-root "$BALDS_DATA" \
  --utilities "$BALDS_DATA/results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv" \
  --output "$BALDS_DATA/results/paper/controls/deletion"
```

输入800行长表含native/signed-square两种读出；公开入口先取native实效用，重新计算控制，不把旧变换后的评价分数冒充原效用。输出逐query与汇总。

历史四候选native head−full效用差.000926/.001280（300/1000）；同名单signed-square使Full所选效用增.000939/.002926。有效负/正结果都保留，不能预设终端变换必然伤害选法。

## S3：聚合及均值/计数

support score降序取k=ceil(.05N)，保持原值和符号。positive-only=`max(s,0)`；soft=`max(s-max(0,s_k),0)`；unit-head=H的0/1指示。DAS输入native t²。count identity为`D*s=D*(s-mean(s))+mean(s)*D*1`，count保留score均值符号；三者LDS不是可相加的相关贡献。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/tools/aggregation_controls.py transforms \
  --data-root "$BALDS_DATA" \
  --output "$BALDS_DATA/results/paper/controls/transforms"
```

同一入口还产生M3 matched DAS一次项/square/signed-square及κ控制，共享一次计算。不得为S3重拟合BA背景；AB2 IF历史缺格保留。跨seed query-bootstrap对共同val同步抽样。

## S4：其余方法

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
python Codes/Figure/tab_4_4_main.py
```

此脚本提供现存full-sum方法覆盖；已接受E4的head表依赖`results/lds_e1e5_wave1_20260917/jinxu2/e4_c10_ab2_summary.json`及纠正后的逐query长表，不能因主表源码保留而假称所有head从该脚本重新生成。若新公共表格导出器尚未覆盖所有E4列，已有表的重排可直接读取接受JSON；不安排新的GPU评分来填旧C2空格。TracInCP/GAS在C10仅seed42，pixel/CLIP val共享输入，Journey仅gen。

## 来源

- `reports/lds_nextwave_2026-09-20/{ACCEPTANCE_J3_CPU_j3-a1.md,ACCEPTANCE_J2_CPU_j2-a1.md}`，执行1f2e6d1；`results/lds_nextwave_20260920/{j3/a1/e2,j2/a1/transforms}`。
- `reports/lds_e1e5_wave1_2026-09-17/ACCEPTANCE_JINXU2_j2-a3_20260917.md`：纠正后E4。
- `Paper/ICLR/Figures/evaluation/{paired-rows.tex,sign-rows.tex,results.json}`保留当前稿的接受汇总及来源。

S1–S4现有结果不是本次DAS BA t/t²更正的验收目标；其native/transform定义保持原样。
