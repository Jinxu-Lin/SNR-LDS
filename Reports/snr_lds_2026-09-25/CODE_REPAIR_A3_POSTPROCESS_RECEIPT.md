# A3 后处理修复回执

2026-09-25；Experimenter按用户“直接修理”兼任Coder。
交付：`/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4`，无Git元数据；源Codes同步修复，旧r2/r3快照和a3工件未修改。

## 根因与修改

1. `tools/summarize_benchmarks.py` 通过非空valid seed组判断adaptive字段；全失败组valid为空，误用ba_lds，造成混合CSV。现在根据原summary schema判定；全失败SNR仍导出snr_lds=null和失败统计。真正混合BA/SNR不静默拼表。CSV临时写入后原子替换，避免半张表。
2. 同模块Full query bootstrap原先把SNR拟合失败但Full可计算的query也纳入；修为与SNR相同有效集，不改变原逐query值或seed均值。
3. `workflows/benchmark.py::verify_manifest` 原本先对float32 t平方，生产端先转float64再平方，导致q0约4.61e-10差异；修正转换次序。另DDPM q51暴露矩阵向量乘法与生产端逐行dot的舍入顺序差异（约1.36e-12）；重放改为生产端逐subset点积。保留rtol=0、atol=1e-12；未改分数、预测、选择或拟合产物。
4. 现有A shell增加 `SNR_STAGE=closeout`：只读a3、写a4，只执行汇总/E-DEL/图/verify。nohup增加setsid，采用Executor已成功使用的进程组隔离方式。不新建调度系统。

## 验证与兼容

- 新增全失败SNR组CSV测试、Full配对CI测试、float32 DAS在不同尺度下验证及故意篡改预测必须失败的测试。定向测试13 passed。
- 实际a3后处理在临时诊断目录 `/tmp/snr-closeout-qa.yJk8mV/` 跑通：119个汇总组，E-DEL status=ok，两个图文件生成。没有执行拟合、训练或评分；不是正式a4产物或论文采用。
- 最终全库CPU测试：330 passed、1 skipped、22 warnings，71.12秒。warnings为CUDA被隐藏时AMP关闭提示及既有scheduler弃用提示；layer检查92模块通过，launcher shell语法通过。
- 实际a3全量只读验证：verdict=pass，verified_queries=24300，errors=[]，missing仅原5个评分格。验证没有新拟合或修改a3；严格rtol=0/atol=1e-12保留。
- 五个DDPM缺输入仍是真实缺格，不能通过字段修复补造。当前本机标准路径及新重训方法路径未找到对应评分；本次不启动任何上游GPU任务。

运行任务书 `TASK_JINXU3_SNR_LDS_CPU_V4.md`。只交付修复和正式收尾命令，未启动a4 detached正式作业；243格全部复用，a3只读。B仍待独立修复，不宣称完成。
