# Coder：B 的主库接入与一个统计量修正

日期2026-09-25；起点 `CodeSnapshots/snr_lds_v1_20260925_r2`；状态：待修正。A执行书已更新为V2，**不要修改A只读快照，也不要让这些修正阻塞A主表**。无需重做SNR算法、训练或评分。

## 1. 完整R16输入接入

`tools/run_snr_lds_b.sh`写死 `results/repeatability_r16`，但主库不存在此目录。主库现有package线索：

- `/path/to/CFA/_Data/results/e3c_full_repeats_20260918/package.json`
- `/path/to/CFA/_Data/results/narrative_wave_20260921/incoming/xuchang3/R-A/package.json`
- 同级`R-B/package.json`、`R-C/package.json`
- `/path/to/CFA/_Data/results/narrative_wave_20260921/packages/pilot_manifest.json`

复用原R16接收/读取逻辑，交付实际0..15完整repeat及16 gen/16 val ID的读取清单；不要重算repeat，不要把旧4次当16次。既有只读读取器若要求一个逻辑根，可在本次b1/inputs下准备轻量路径映射/链接及正确package，或适配显式来源列表；不搬整库、不改历史工件。

## 2. pilot接口对齐

`Reports/evidence/r16_pilots.json`为records[]溯源文档，当前`summarize()`读取`pilots[method][track]`，两者不兼容。交付实际pilot映射，并按R16配置的原query IDs从100列独立pilot中取16列；保留来源路径和ID，不从repeat自身拟合。可复用现有已对齐pilot，不重复生产。运行命令指向处理后的映射，不留给Executor临场写代码。

## 3. 预测差方差

`snr_repeatability_statistics`目前`prediction_variance`是单subset预测方差平均。保留它作为辅助值，另交付论文所需各group/query的subset-pair差方差：先构造每repeat的完整预测P，再算样本方差 `Var_r(P_m-P_n), ddof=1`，m<n。可逐pair保存或逐query汇总，但要保存可复算证据/明确pair覆盖，不能丢协方差或把当前值改名。

增加小数组普通测试：两subset各自随repeat同幅波动时，单subset方差>0而差方差=0；验证新字段能区分。检查pilot拟合失败的query不作为合法空选择混入校准统计，若受影响按NA列出；不新增科学条件。

## 交付

独立B修正版、定向测试、修订后的实际launch/status/resume/verify命令及只读来源对应关系；不重复A开发、不跑正式实验、不加冒烟。验收器除8/8行数外覆盖实际repeat IDs、pilot对齐与新预测差方差字段。交付后Experimenter更新B运行书。
