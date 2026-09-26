# 终稿M3：幅值头、幂变换与匹配平方

有效资产：Fig2 `fig:mechanism`、Fig6 `fig:mech-head-full`、Fig5 `fig:mech-power-full`；Table4 `tab:mech-fm-square`、Tables5/6 `tab:app-square-comparison` / `tab:app-square-gains`、Table7 `tab:mech-ddpm-matched`。

## 固定科学定义

同一score、model、query、正则化、mask与GT，仅改变选择或readout。四核心C2三seed、两轨100；main Fig2用gen。所有LDS与差值展示×100。

- **主幅值头**：按abs(pre-square)稳定降序，训练行序破tie，k=ceil(kappa*N)；kappa5/10/20/30/40/50/60/70/80/90/100%。Fig2(a)DAS为native t²，其余linear；Fig6所有方法linear，DAS未平方。
- **同名单平方**：Fig2(c)在完全相同abs-score head上比较t与t²，不按平方后的支持排名重新选名单。原稿top5%的square−linear为FMAS+.76、IF−.17、DTRAK.00、DAS+.05 LDS点；采用终稿稳定tie规则重算后的FMAS/IF为+.73643/−.09685，具体差异见下文。
- **幂变换**：每query除max(abs(t))，signed=sign(t)*abs(t)^p、folded=abs(t)^p；p0,.2,…,4。p0分别保号或unit。Fig5只四核心gen/val三seed；p6/8、13方法及额外EKFAC readout退出。
- **跨平台DAS Tables5/6**：仍使用明确标注的signed-support5%，不是主图abs head。C2/C10/AB2两轨，native配置signed t、ordinary square和signed square；既比较各自重排head，也比较固定原signed head。AB2 t在native lambda1恢复，不用旧lambda.5。2000次paired bootstrap；val各模型共用query同步抽，gen各seed内抽。
- **DDPM Table7**：独立C2档案、共同100query双轨、前64半集；gen/val正确原ID对齐。原LDS选参后固定：projected lambda10，IF1e-12，FMASrho1e-7/1e-8（gen/val）。报告original/folded/signed；1000次paired query-bootstrap。DDPM1000扩展表已退役，但原1000列档案可作为matched100的底层源。

## 执行线

[共用上游](../00_pipeline/README.md)→读取最终score/masks/GT→按既定head或幂读出→deleted-row sum→每query Spearman→seed均值/sampleSD或paired区间→终稿图表。无新的训练/MC/curvature。

基础计算入口（只运行终稿所需内容）：

```bash
export BALDS_DATA_ROOT="$PWD/_Data"
python Codes/Figure/fig2_head_profile.py
python Codes/Figure/tab_3_1_square.py
python Codes/Figure/fig1_power_diagnosis.py
python Codes/Figure/compute_abs_heads.py
python Codes/Figure/render_fig2.py
python Codes/Figure/render_fig5.py
python Codes/Figure/render_core_power.py
python Codes/Figure/tab_C3_ddpm.py
python Codes/tools/aggregation_controls.py transforms \
  --data-root "$BALDS_DATA_ROOT" --output "$BALDS_DATA_ROOT/results/paper/matched_das"
python Codes/tools/render_matched_das.py \
  --input "$BALDS_DATA_ROOT/results/paper/matched_das" \
  --output "$BALDS_DATA_ROOT/results/paper/figures"
```

四核心C2原分数是同一来源；DAS机制linear值与主benchmark native Full不是同一readout，不能互换。所有新图及metadata在`results/paper/figures/`，可用`BALDS_FIGURE_ROOT`独立改址。Table7只重算固定lambda的投影kernel，读取已保存曲率score；不采样新特征、不重选参数，不输出退役1000query表。其dataset loader可设`HF_DATASETS_OFFLINE=1`只读已迁缓存。

### 先前数值重放发现的差异

终稿`XY03_MechanismExperiment.tex:43`明确要求按训练index打破head ties。旧Figure JSON实际来自`np.argpartition`：同一迁入数组上恢复旧算法可逐seed精确复现旧top5%值；稳定排序改变FMAS三seed的55/68/49个query、IF的60/48/53个query边界名单，DTRAK/DAS为0。公开代码采用终稿的稳定规则。因而再生成Fig2/6与Paper旧点存在小差异；四核心head/readout88格最大差约0.07717 LDS×100（IF20% linear）；FMAS head曲线最大差约0.04372，不能把旧数值当作稳定规则验证结果。逐seed证据在`results/paper/figures/metadata/head_tie_audit.json`，历史PDF和JSON保留，不修改Paper。

Table7先前CPU重算的IF/FMAs四行与Paper显示值及区间一致；由float32投影features重建的DTRAK/L1四行有约0.01–0.02 LDS×100的显示差异。保留每query数值与固定lambda，明确其为CPU重算；接受`tab_C3_ddpm.json`提供原数值，不能默默替换新的计算结果。head 的数值核验见 [tie audit](evidence/head_tie_audit.json) 和 [curve comparison](evidence/head_curve_comparison.json)。

## 保留证据及退出边界

Tables5/6数据来自`results/lds_nextwave_20260920/j2/a1/transforms/`匹配DAS分析；底层score/fit/GT身份和AB2lambda1恢复方法不变。Table7使用存档匹配100查询和固定参数。Fig2与Fig6由当前head计算器生产；Fig5使用 `Codes/Figure/render_core_power.py`，只排版p0..4四核心。脚本及输出目录可能保留历史编号，以本报告的论文标签为准。

不包含已退役的范围扫描、DDPM1000-query扩展、p6/8或旧DAS胜负计数。Tables5/6保留的是匹配DAS配对控制，不是这些范围扫描。

同样本SNR-mask交集的head/square控制属于Fig3(a)，见[E-BENCH](../05_snr_lds/README.md)，不在这里重拟合sigma。上述机制结果只限定所测score和平台，不主张平方总有益或截断总消除平方优势。
