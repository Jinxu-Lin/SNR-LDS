# 终稿实验与复现索引

更新：2026-09-26。以本仓平铺的 `Paper/iclr2027_conference.tex` 为准；旧 CFA/Paper 与9月22报告已不是当前科学定义。Paper不在本轮改写。[变更审计](final_alignment_2026-09-26/PAPER_SCOPE_AUDIT.md)说明所有口径变化，[有效图表图谱](final_alignment_2026-09-26/paper_include_graph.json)给出准确路径和行号；旧报告保存在[日期快照](final_alignment_2026-09-26/before/EXPERIMENT_INDEX.md)。

**当前评价是SNR-LDS：零均值Gaussian噪声拟合，硬阈值 `abs(score)/sigma>3`；没有密度比gamma、pi0或soft BA。DAS始终以signed pre-square t拟合并选择，聚合native t²且只平方一次。**

## 执行入口

1. [共用上游](experiments/00_SHARED_PIPELINE.md)：数据、模型、query、特征/曲率、原始分数、子集重训及GT。已有接受工件时不重复训练。
2. [SNR主表](experiments/E_BENCH.md)：准备明确身份的输入→拟合/四阈值评价→按有效query汇总→覆盖率/拟合图。
3. [实际删除](experiments/E_DEL.md)：按三seed benchmark共同有效集选方法，再比较既有50query干预及图像变化。
4. [来源检索](experiments/E_SOURCE.md)：CFM/DDPM各最终reviewed500，100val选参、400test评测，13方法。
5. 机制与诊断：[M1幅值](experiments/M1_MAGNITUDE.md)、[M2a完整R16](experiments/M2A_REPEATABILITY.md)、[M3变换](experiments/M3_TRANSFORMS.md)。

代码入口`balds-run`生产方法分数，`balds`执行SNR评价，`balds-repeat`执行重复估计；Figure脚本读取数值并输出独立结果目录。文件存在、可运行、正式验收和论文采用分别记录。

## 当前全部有效图表

| 资产 | 稳定LaTeX标签 | 报告 |
|---|---|---|
| Fig1(a), Fig4 | `fig:motivation`, `fig:mech-density-full` | M1 |
| Fig1(b), Table4 | `fig:motivation`, `tab:app-r16-bands` | M2a |
| Fig2, Fig5, Fig6, Table5 | `fig:mechanism`, `fig:mech-head-full`, `fig:mech-power-full`, `tab:mech-fm-square` | M3 |
| Tables8/9/10 | `tab:app-square-comparison`, `tab:app-square-gains`, `tab:mech-ddpm-matched` | M3 |
| Fig3(a) | `fig:snr-diagnostics` | E-BENCH固定SNR筛选诊断 |
| Fig3(b), Tables12/13 | `fig:snr-diagnostics`, `tab:app-delete-existing`, `tab:exp-selection` | E-DEL |
| Tables1/15 | `tab:exp-snr-cifar`（别名`tab:exp-c2`）, `tab:exp-snr-platforms`（别名`tab:exp-platforms`） | E-BENCH |
| Tables14/16, Figs9/10 | `tab:snr-coverage`, `tab:snr-sensitivity`, `fig:snr-fixed-fits`, `fig:snr-fit-distributions` | E-BENCH |
| Tables2/17/18/19 | `tab:exp-retrieval`, `tab:app-injection-pairs`, `tab:app-retrieval-by-concept`, `tab:app-retrieval-ddpm-by-concept` | E-SOURCE |

共8图、15表。目录编号存在空缺是正常退役结果；不按Paper/Figures里存在的目录或旧README恢复已删实验。

## 方法与结果边界

- 主表16方法：pixel dot/cos、CLIP dot/cos、gradient dot/cos、TracInCP、GAS、Journey-TRAK（仅gen）、Relative IF、Renormalized IF、TRAK、D-TRAK、DAS、EK-FAC IF、FMAS raw。
- C2/C10 FM各3模型seed42/123/456；AB2/C2DDPM各seed42。所有平台每轨100query；mask64/64/32/64。
- SNR a4正式逐query覆盖243格/24300物理记录：16946有效、7354失败、3876合法空选择。共享pixel/CLIP val在表中去重；它们的SNR fit全部失败，NA不是缺数据或0。
- Table1 DDPM TracInCP/GAS四格是作者9月26提供的汇总，Table14明确没有逐querycoverage；不将其虚构为24300中的新记录。Journey DDPMgen仍缺可比结果，val为n/a。
- Table1/15是各方法own-valid均值；E-DEL先用每seed四核心共同有效query（83/75/81）作公平候选选择，再比较seed42全部50干预，不逐query切换候选。
- SNR sigma只拟合一次，zeta1/2/3/4与fixed-head/readout控制共享该fit；默认3不按LDS调参。R16只用于机制诊断，不是运行SNR所必需。

## 已退役范围

旧BA密度比评价/positive-only pilot、旧逐query四候选择法、旧S1–S4聚合补充、独立R4结论、M2b采样预算、SNR独立重复校准B线、common200与旧CFM-v1 retrieval、DDPM1000扩展、13构造power及p6/8、signed-support范围图Fig7/8与Tables6/7、单独FMAS band-only Table3均不属于终稿。

PW-DTRAK、AbU+、NDA仍可出现在相关工作引用中，但不是当前实验/待补主表。Tables8/9里明确标注的DAS signed-support5%配对控制仍保留；R16 repeat0–3和独立pilot仍是当前依赖。

迁移按当前[范围审计](final_alignment_2026-09-26/PAPER_SCOPE_AUDIT.md)和本轮实际迁移清单执行。旧9月22迁移清单只提供历史线索；共享输入可复用，但不能把旧query版本或旧BA数值改名成SNR。

## 已记录但未修改的稿件差异

当前CFM细/粗组prose仍保留旧数字，与Tables2/18及正式reviewed500 JSON不一致；具体值见范围审计末节。代码和迁移使用最终reviewed500数据。Paper保持作者终稿原样。

终稿图表生产入口见[Figure说明](../Codes/Figure/README.md)，实际运行及两项数值差异见[绘图复现回执](final_alignment_2026-09-26/FIGURE_REPRODUCTION.md)。Fig2/5旧argpartition ties与终稿稳定index规则不一致；Table10投影float32 CPU复算存在少量0.01–0.02 LDS点差异。两者均保留新逐项结果和旧证据，不修改Paper。
