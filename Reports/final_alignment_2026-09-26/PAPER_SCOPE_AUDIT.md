# 2026-09-26 终稿范围与复现对齐审计

本轮权威输入是目标仓当前 `Paper/iclr2027_conference.tex` 及其有效 include graph，**不是 CFA/Paper，也不是 9 月 22 日 Reports**。Paper 保持只读。本报告记录科学范围和依赖，不宣称新实验已执行；代码修改与数据搬运由本轮对应负责人另行记录。

[有效 include graph](paper_include_graph.json)逐项记录源文件、行号、标签、图片；所有被包含文件与图片均存在。当前有 8 张图（Fig1–6、9、10）及 15 张表（Table1、2、4、5、8、9、10、12–19）。旧 `Paper/Figures/README.md` 仍列出已退役目录，不能据此确定迁移范围。

## 相对 9 月 22 日报告的变更矩阵

| 项目 | 旧报告 | 当前终稿要求 | 直接依据与实现影响 |
|---|---|---|---|
| 主评价器 | BA-LDS，背景密度比 gamma>.5 | **SNR-LDS**，硬选择 `abs(t)/sigma>3` | `Paper/Sections/04_Method.tex` 的 `sec:snr-lds`；`XY04_EvaluationPipelines.tex:13–60`。没有 soft BA、连续权重、pi0 或 gamma |
| 中心密度拟合 | median±MAD；A+B*x+C*x²；背景均值自由 | RMS归一化；Gaussian KDE，Silverman样本SD带宽；MAD仅决定区间半宽；101点 **[-1.4826MAD,+1.4826MAD]**；密度加权拟合 **A+C*x²** | 同上13–38行。零均值固定，不保留线性项；`sigma_x=sqrt(-1/(2*C))`，原单位sigma=rms*sigma_x；截距不参与选择 |
| 退化情形 | 旧pi0合法性/空支持逻辑 | 全零向量直接合法空选择；带宽/中心宽度不正、秩亏/非有限、C>=0或sigma非正失败NA；合法空集合及常数预测/响应记0且分类计数 | 同上54–60行。不能用旧BA有效率或pi0拒绝规则替代 |
| DAS | 9月22明确定义t拟合/t²聚合，尚待重聚合 | **仍对所有平台以signed pre-square t拟合、按abs(t)选择，聚合t²且仅平方一次** | 同上41–52行。9月25 SNR a3/a4是新的正式结果；旧t/t结果不再作为待完成主线 |
| 阈值分析 | BA参数/阈值扫描不在稿 | zeta=1,2,3,4均复用同一个sigma；默认3不由结果调参；报告保留率/覆盖/空集合/排名 | `XY05_EvaluationExperiments.tex:73–84`；Table16 |
| M1幅值 | concentration表、单独FMAS分带表、组合Figure1 | Fig1(a)/Fig4密度；FMAS中央50%区间约±.502；Fig2/Fig5幅值头；**Table3单独分带表退役** | `XY03_MechanismExperiment.tex:50–63,142–152`；`appendix_c4_retirement_2026-09-26/README.md` |
| R16统计 | `abs(mean−querymedian)/sampleSD`的Rep | **VarRatio=sum(sample variance)/sum(repeat mean²)**；Fig1(b)依abs(mean)升序分10等大bin，每query先算比例再等权平均16queries；Table4保留独立pilot五带与band-LDS | `04_Method.tex:eq:band-variation`；`XY03_MechanismExperiment.tex:65–91`。64完整矩阵和独立pilot仍有用，旧0–3属于R16 |
| 主幅值头/平方 | signed-support head与native head混有多种曲线 | main Fig2(a,c)及Fig5由**abs(pre-square)**排名，ceil(kappa*N)、训练行序打破并列；同名单线性/平方；DAS native平方 | `XY03_MechanismExperiment.tex:41–48,142–152`。跨平台Tables8/9仍保留明确标注的signed-support 5%控制，二者不可互换 |
| 幂变换 | 13构造、额外EKFAC readout、p6/8锚点 | 仅四核心、signed/folded、p=0,.2,…,4、gen/val、三seed | `XY03_MechanismExperiment.tex:94–109`；Fig6 |
| E-DEL选法 | 每query在四候选中取Full/BA最大，common49 | 在每seed四候选共同有效集上求均值，再等权跨3seed选**一个方法**；Full选DAS、SNR选FMAS；然后比较seed42全部50query固定干预 | `XY05_EvaluationExperiments.tex:98–112`；共同有效数83/75/81；不是逐query混选，不按干预50query重新选法 |
| E-DEL视觉 | 旧报告未覆盖最新图像比较 | Fig3(b)包括FMAS/DAS/random的loss、uint8/255像素L2及CLIP cosine；随机5集合先按query平均 | 同上116–132行；需 `fmas_das_visual_20260922` 与 `fmas_das_random_visual_20260922` |
| SNR固定筛选诊断 | 旧密度比筛选或旧同名单平方图 | Fig3(a)复用已接受zeta3 mask，与abs-score head交集；每个readout/head不重拟合；四核心C2 gen三seed | 同上134–142行；`Reports/snr_lds_2026-09-25/ACCEPTANCE_FIG3_SNR_CONTROLS.md` |
| 主表范围 | C10部分checkpoint方法缺格，AB2 IF待补；新增3方法pending | 16标准方法；C10 checkpoint方法3seed已补；AB2 IF完整、gen lambda1e-7/val1e-6 oracle Full选参；PW-DTRAK/AbU+/NDA不入稿 | Tables1/14/15；`XY05_EvaluationExperiments.tex:38–41,144–168`。新方法仅相关工作引用，不保留其生产代码/工件作为论文依赖 |
| SNR覆盖 | 旧BA 17800/3700等 | a4 243格/24300物理query，16946有效、7354失败、3876合法空选择；pixel/CLIP全NA；共享val去重展示 | 同上149–158行；Table14。DDPM TracInCP/GAS四格仅作者提供的汇总，不混入24300计数；Journey DDPMgen仍缺 |
| E-SOURCE | common200主表、旧CFM400附录、DDPM400只11方法 | **CFM与DDPM均最终审核500，100val+400test，13方法**；CFM新v2替换280图；各自原query身份固定 | `05_Experiment.tex:Controlled-Source Retrieval`；`XY05_EvaluationExperiments.tex:49–63,174–204`；Tables2/17–19 |
| E-SOURCE选参 | CFM IF1e-8的common200 | FMAS rho1两模型；IF CFM1e-9、DDPM1e-12；100val within-host AP选参、400test global AP/recall | 同上58–63行。旧同样400列仍可能是旧图片，不能按形状合并 |
| 附录旧补充 | positive-only BA pilot、S1–S4、signed head ranges、DDPM1000等 | 这些不在当前include graph；只保留Tables8–10匹配平方与DDPMmatched100 | 两份Sep26 retirement记录；旧文件存在不等于当前论文需求 |

## 当前全部有效图表

| 资产 | 标签 | 最小数值来源/依赖 |
|---|---|---|
| Fig1 | `fig:motivation` | 四核心C2密度；FMAS gen R16完整分数，16queries的10等大bin VarRatio |
| Fig2 | `fig:mechanism` | 四核心C2 gen三seed原score+M64/GT；abs head native、signed power、同abs head平方 |
| Fig3 | `fig:snr-diagnostics` | SNR a3 selection masks与原分数/GT；FMAS/DAS/random固定50query损失及图像指标 |
| Fig4 | `fig:mech-density-full` | 四核心C2 gen/val三seedRMS密度 |
| Fig5 | `fig:mech-head-full` | 四核心C2 gen/val三seedabs-head线性readout；DAS未平方 |
| Fig6 | `fig:mech-power-full` | 四核心两轨signed/folded p0..4 |
| Fig9 | `fig:snr-fixed-fits` | SNR固定C2 seed42 gen q0/q1，四核心拟合，不择优 |
| Fig10 | `fig:snr-fit-distributions` | 全有效SNR记录sigma与保留比例，失败单独计数 |
| Table1 | `tab:exp-snr-cifar`, `tab:exp-c2` | C2FM、C2DDPM16方法SNR；DDPMcheckpoint四格汇总来源单列 |
| Table2 | `tab:exp-retrieval` | 两模型13方法最终reviewed500、共同400test |
| Table4 | `tab:app-r16-bands` | 两方法×两轨R16 coordinate mean/sampleSD/pilot bands和band-LDS |
| Table5 | `tab:mech-fm-square` | C2四核心两轨full线性/ordinary square |
| Tables8/9 | `tab:app-square-comparison`, `tab:app-square-gains` | C2/C10/AB2两轨DAS线性/平方/signed square；signed-support5%与fixed head；paired bootstrap |
| Table10 | `tab:mech-ddpm-matched` | DDPM四核心共同100，两轨；原/平方/signed square和paired CI |
| Table12 | `tab:app-delete-existing` | 全四方法native top300/1000、seed42 q0–49原效用及CI |
| Table13 | `tab:exp-selection` | SNR全局选FMAS、Full全局选DAS后，原50干预的均值 |
| Table14 | `tab:snr-coverage` | a4逐query状态、按shared val去重；作者汇总不伪造coverage |
| Table15 | `tab:exp-snr-platforms`, `tab:exp-platforms` | AB2、C10FM16方法SNR |
| Table16 | `tab:snr-sensitivity` | 四核心四平台两轨zeta1/2/3/4，复用同fit |
| Table17 | `tab:app-injection-pairs` | 十注入概念、五fine/五coarse、每概念200source/10val/40test |
| Tables18/19 | `tab:app-retrieval-by-concept`, `tab:app-retrieval-ddpm-by-concept` | 两模型13方法各400test原逐query/逐concept统计 |

## 必须保留的工件族及可退役边界

### 主评价和重新生产的共用上游

- 标准方法ID：`pixel_dot,pixel_cos,clip_dot,clip_cos,grad_dot_T100,grad_cos_T100,tracincp_T100,gas_T100,journey_trak_T100,relative_if_T100,renorm_if_T100,trak_T100,dtrak_T100,das_T100,ekfac_if,fmas_raw`。DAS评价名可为`das_native_sq`，磁盘signed源仍是das_T100。
- 平台：C2FM/C10FM seeds42/123/456，各gen/val100；AB2及C2DDPMseed42各gen/val100；mask64/64/32/64。需要各方法最终scores、准确query/train IDs、对应keep masks与GT。pixel/CLIP失败仍需原scores证明失败，不能按NA删除输入。
- 重新计算分数才需要checkpoint、原数据/模型权重、generation、特征/curvature及checkpoint特征。重训子集模型是重建GT的上游；已接受GT足以复算论文所有LDS，不必为数据空间机械搬运所有历史子集checkpoint。具体分层与容量由迁移清单记录。
- SNR正式归档：`results/snr_lds_20260925/a3/{inputs,panels,tables}`与`a4/{tables,edel,figures,verify_a.json}`。a4是对a3的后处理，**只搬a4不足以恢复Fig3 masks或逐query**。
- DDPM checkpoint四格的作者汇总与任何后来可用正式score/逐query分别登记；缺coverage不等于分数为0。Journey DDPM的另一次生成图与旧GT不匹配，不能为消除缺格偷偷配旧GT。

### 机制和删除

- R16全64矩阵、独立pilot源分数、原IDs、coordinate_stats/summary、repeat0–15；旧e3c目录0–3保留。新的VarRatio decile与pilot-band CSV在`Reports/figure_split_2026-09-25/`，作小型接受证据；不需要独立R4结论或M2b新增预算。
- 四核心C2原score/GT/mask支持密度、head、power和平方；DAS原signed分数绝不能被平方覆盖。AB2 matched DAS lambda1恢复原项及必要原features/kernel用于Tables8/9；DDPMmatched100正确原query映射用于Table10。DDPM1000扩展不再是独立迁移目的，但共享原始1000列档案如果是提取matched100的唯一源，仍属于底层依赖。
- 原四候选删除效用保留Table12；FMAS/DAS/random重生成图、原图、loss、CLIP embeddings或可重算编码输入及原query identity支持Fig3b。直接接受来源为`results/fmas_das_visual_20260922/`与`results/fmas_das_random_visual_20260922/`；random每预算只有5模型，不能把250观测称250重训。

### 来源检索

- 两模型最终审核500query包、review_decisions/rename/replacements、注入meta/原训练行、各自gen model与train features/curvature（重新评分所需）。CFM最终v2身份在`_Data/reports/cfm_reviewed500_2026-09-23/MIGRATION.md`；220旧图也按该批次全量重新query featurize，不拼接旧特征。
- CFM11方法标准scores/inject_result、`results/cfm_reviewed500_20260923/verified_summary.json`；FMAS/IF `results/cfm_test400_20260923/analysis/{method}/{result.json,test400_scores.npy}`及`cfm_val100_20260923`对应val结果/块。接受依据 `_Data/reports/experiment_closeout_2026-09-24/ACCEPTANCE.md`，取代早期作者撤回状态。
- DDPM11方法review-v2正式scores/results以及FMAS/IF最终reviewed500 val/test结果、选参证据；精确source路径由迁移审计核对主库当前接收状态，不能只依据远端produced报告。
- common200、旧CFMv1 400、旧DDPM初版查询只保留历史报告，不作为终稿结果输入。共享train features/模型可依实际身份复用。

## 终稿本身发现的差异（仅记录，不改Paper）

1. **CFM细/粗组prose仍是旧值。** `XY05_EvaluationExperiments.tex:195–198`写FMAS AP .056/.486、Recall .100/.490及IF AP .050/.444、Recall .097/.453；当前Table18与接受reviewed500 JSON不一致。正式JSON的FMAS fine/coarse global AP为 .0494322394/.4567854335，Recall .089900/.463575；IF为 .0460366060/.4239293689，Recall .088250/.433200。Table2 overall与正式JSON一致。代码和迁移应使用最终reviewed500数据，不为了匹配旧prose混入旧query结果。
2. **DDPM TracInCP/GAS口径不完整。** Table1注释说明四个数由作者于9月26提供；Table14/附录明确无逐querycoverage。这四数可以保留作者来源声明，不得虚构对应记录/覆盖率，后续若有真正运行产物须单列验收。
3. **历史文档状态过期。** Figures/README列退役资产；9月22实验索引和命令仍是BA/common200；9月25早期adoption报告描述旧figure目录。当前include graph和最终接受身份优先，但不修改这些历史回执的原始事实。

## 本轮更新要求

新实验报告应在保留9月22历史快照后，改为当前SNR、reviewed500和VarRatio，移除把旧pilot/S1–S4/新增三方法作为当前任务的说明；命令只指向实际交付的便携CLI。代码功能存在、真实工件可读、科学数值验收、论文采用分别记录。本报告没有复制数据、改Paper、训练、提特征、评分或拟合。

## 本轮代码与实际重算补充

最终便携入口及真实计算结果见[Figure复现回执](FIGURE_REPRODUCTION.md)。除原先记录的CFM fine/coarse prose差异，实际复算另确认：Fig2/5历史head采用argpartition，终稿文字要求stable training-index ties；公开实现遵循文字，新旧点有小差异。Table10投影float32重建也有少量打印精度差异，精确数值与原因边界均保留在回执中。Paper保持原样。
