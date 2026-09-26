# CFM 人审查询集迁移与旧结果适用范围更正

2026-09-23，研究者提供 `/path/to/1`，授权迁入项目并全量重算查询特征。

已将整个目录移动到 `/path/to/CFA/_Data/results/inject/cifar10_inj4/c10mine_cfm_v2_png`；原`/path/to/1`不再存在，内容并未删除，可从项目内目录恢复。500张candidates为正式选择，1396张backup仅保留原审核材料，rename_map.csv/replacements.csv原样保留。
DDPM原审核目录 `/path/to/CFA/_Data/results/inject/cifar10_inj4/c10mine_v2_png` 未动；这是DDPM，不得因其旧名称没有ddpm字符串而拿来当CFM输入。

实际迁移命令：
```bash
test ! -e /path/to/CFA/_Data/results/inject/cifar10_inj4/c10mine_cfm_v2_png && mv /path/to/1 /path/to/CFA/_Data/results/inject/cifar10_inj4/c10mine_cfm_v2_png
```

prepare_review.py按rename_map + 280条替换记录重建candidate ID，每张PNG与mined tensor逐像素相等；500个唯一候选，10类各50。相对旧查询保留220、替换280；替换旧val52/test228。研究者明确不用旧query features，故这220也全量重算，不做增量拼接。

已用e585313普通导入器在CPU构造新500：process=cfm、seed42、mine_tag=c10mine_cfm_v1、tag=None、review=results/inject/cifar10_inj4/c10mine_cfm_v2_png/review_decisions_v2.csv、version=v2、reviewer=researcher、conditional=True、force=True；config inject.split.seed=0, inject.split.val_frac=.2。
输出 `generations/cifar10_inj4/cfm_cond/seed_42/inject_queries.pt`，按原host/candidate_index排序，每概念10val/40test，Q500。
程序原有queries_sha256：`cb0f6c7240e814c17c945abaa53e9dd0633b7bfcdb99ea5a3ab75773ea6c6cd7`。
导入前后DDPM查询包和CFM五件train_features的stat未变；没有GPU运行、没有特征/评分/评测新结果。

## 更正旧验收范围

旧CFM-v1候选集身份fe3004f1...，与新审核集不同。2026-09-17特征/11方法验收和2026-09-23 FMAS/IF full400验收仅对旧输入成立，**不得把旧值称为本次最终人审集结果**。历史回执与数值保留，不改写。新集11方法ready，FMAS/IF待排期；DDPM已审核v2、BA-LDS、删除实验均不受此次CFM来源检索图片更新影响。Author须将旧CFM来源检索表/分组/排名暂退为待更新，本文不直接修改Paper。

## FMAS / IF 全量新500开销（只估算，未授权启动）

沿用原采样/逆曲率/阻尼网格和已有曲率；不复用任何旧查询特征/分数，也不能复用旧100val的选参结论。新100val重新选参，400test只评测。
旧同配方新增200查询的实际4090墙钟：FMAS10.06410582h、IF11.22448562h（overnight_acceptance_2026-09-23）。按100查询块线性估算：

| 方法 | 500查询估算 | 排期建议 |
|---|---:|---:|
| FMAS | 25.16卡时 | 26–28卡时 |
| EK-FAC IF | 28.06卡时 | 29–31卡时 |
| 合计 | 53.22卡时 | 55–60卡时 |

每方法5个100查询块，约5.0/5.6小时/块；两卡各负责一方法约28–31小时，四卡混合排约15–18小时（块粒度不均匀，另留汇总/传输）。本估计含原流式评分中重复的训练梯度计算，不承诺训练侧零计算；只是无需重新训练或重新fit约23小时的曲率。
这比单纯补旧集200查询贵，是因为现在每方法要重新算100val+400test共500，不是补280或仅重算400test。

## 执行包验证与交付状态

没有核心算法代码修改。prepare_review真实500张索引/像素验证通过；run.sh shell及内嵌Python语法、query-only/process=cfm/torch_chunked/强制覆盖参数检查通过；两个既有计算worktree干净。e752744核心CPU测试428 passed/1 skipped（103.00s），layer check通过。提取版本e585313沿用DDPM实际成功的500-query回执验证，不重复冒烟。
已交付TASK_JINXU3_CFM_REVIEWED500_V1.md与三个薄脚本；状态ready，正式GPU及11方法CPU链尚未启动、未通过会话工具发送。研究者手动转发给jinxu3 Executor后可按现成supervisor直接运行。
