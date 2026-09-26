# CFA → BA-LDS 论文产物迁移报告

## 1. 范围与当前交付

源项目为 `/path/to/CFA`，目标项目为 `/path/to/BA-LDS`。新的公共包名为 `balds`。本报告与程序不依赖源机器别名；历史目录中的机器字符串仅标识当时的分片来源，不用于 SSH 或运行调度。

**本轮不迁移 `_Data` 产物。** 当前生成的是本地只读盘点、按论文筛选的逐文件搬运清单、待完成项，以及默认不执行复制的工具。CFA 的训练、评分、文件登记和论文编辑仍可继续；完成后必须刷新清单。盘点时间和字节数以 [inventory.json](inventory.json) 为准，便于区分后续运行产生的新文件。

论文范围包括 M1/M2a/M3、E-DEL/E-BENCH/E-SOURCE，以及附录 S1–S4。范围依据是目标中直接复制的 [正文实验](../../Paper/ICLR/Sections/05_Experiment.tex)、[机制附录](../../Paper/ICLR/Sections/XY04_MechanismExperiment.tex)、[实验附录](../../Paper/ICLR/Sections/XY05_EvaluationExperiments.tex) 和 CFA 的 `Experiment/experiment_v2.md`。本次不把历史任务编号当作当前论文采用状态。

## 2. 必须处理的 DAS 语义差异

最终协议必须区分三个对象：

1. `t`：DAS 原生平方之前的有符号分数，按查询 RMS 归一后拟合密度与背景。
2. `keep(t)`：上述拟合得到的双侧硬筛选集合；有限正背景强度可以大于 1，ReLU 后严格使用阈值 `> 1/2`。
3. `t²`：预测聚合值。Full 使用全部 `t²`，BA 使用被 `keep(t)` 选中的 `t²`，不再对已经平方的分数平方，也不把筛选强度乘进聚合值。

CFA 的 `results/ba_c2_c10_das_presquare_20260922/inputs/bench_manifest.json` 和 `edel_manifest.json` 为 DAS 指定了 `ba_input_space=das_signed_presquare`，但 `score_representation=native` 且 `scores` 指向同一份平方前 `das_T100/.../scores.npy`。源 V3 实际采用的是 **fit(t) / aggregate(t)**。因此，该批历史验收、49 查询选择、均值与置信区间只能说明当时那个协议下的结果，不能成为本项目最终 DAS 协议的预期数值。

可复用：原始 `t`、同一 mask/响应、完全同身份的背景/KDE/筛选结果、原有真实删除干预及其效用。需要重算：DAS Full/BA 预测、逐查询 LDS、own-valid 聚合、跨种子汇总，以及 E-DEL 的四候选方法选择和配对 bootstrap。非 DAS 的数值输入未因这项纠正改变，但 E-DEL 的选择会受到 DAS 改动影响。

不得直接复制旧缓存中保存的 `full_prediction`、`ba_prediction`、LDS 或效用选择作为新结果。缓存兼容性须由新工作流核对拟合输入身份；路径相同或拟合成功并不证明聚合规则相同。无需因此重新训练主模型、提取全部梯度或重做真实删除实验。

本清单将该批 12 个面板的原配置/汇总放入 `provenance`，只将 DAS 的 1,200 个平方前拟合缓存放入 `optional_fit_cache`。整批约 36.6 GiB 的 V3 目录不在默认复制清单。正确新结果的源 run ID 尚未产生或确认，在清单中明确记作 `required_reaggregation`，不编造一个已完成的来源。清单的目标示例 `_Data/results/paper/ba_all_platforms/` 对齐 E-BENCH 执行报告；它是 `balds batch --output` 可配置的新生成结果目录，不是已经存在的历史搬运来源。

旧 `ba_native_gpu_wave_20260921` 在 DAS 平方分数上拟合，旧 `closeout_cpu_20260921` / `ba_soft_20260921` 又属于此前协议，也不能替代此项纠正。AB2/DDPM 的旧结果需要逐平台核对相同的拟合/读出口径，不能因为旧 V3 没改动它们就自动认证为新协议结果。

## 3. 清单结构及迁移层次

[selection.json](selection.json) 保存筛选理由、论文位置、上游验收位置和路径规则。[inventory.json](inventory.json) 保存当前命中的精确路径、文件数/字节数、存在性、源时间及仍未完成的内容。[files.tsv](files.tsv) 每行包含：

`source_relative, destination_relative, family, status, profile, role, bytes, mtime_ns`。

源路径相对 CFA 根，目标路径相对 BA-LDS 根。文件身份、实验和证据通过 `family` 引用规则中的对应记录，避免为每张原始图片重复写整份科学配置。当前被多个家族引用的同一个文件只列一次，额外关系保存在 `also_matches`。

| Profile | 用途 | 是否默认复制 |
|---|---|---|
| `analysis` | 分数、mask、响应、查询、已采用的结果与 R16 输入，支持分析复现 | 是 |
| `regeneration` | 主模型/子集/删除模型、梯度特征、曲率、所需原始数据与潜变量，支持重新测量/评分 | 是 |
| `optional_fit_cache` | 仅兼容的 DAS 平方前拟合缓存；也可以从 t 重新拟合 | 否 |
| `provenance` | 历史验收、配置和不应冒充最终协议结果的旧汇总 | 否，独立保留证据时选择 |
| `pending` | 尚在计算、缺块或未验收的 AB2 IF / 新基线 | 否，程序拒绝把该状态放进复制集 |
| `external` | SD3.5 Medium 等外部模型依赖 | 否，记录已有资产与下载/外置缓存要求 |

各层实测数量见 [SUMMARY.md](SUMMARY.md)。文件数和大小是实际 metadata 盘点，不是训练预算，也不是内容或数值正确性的证明。默认复制层并不意味着所有论文表格已经完成；pending 表列出的工作仍须完成。

## 4. 当前论文实验对应的具体来源

### E-BENCH 与共同上游

保留当前 `cifar2_5k`、`cifar10_v2`、`artbench2_256`、`cifar2_das` 平台和当前方法目录中的分数/选参元数据；匹配 `subsets/*_masks.pkl` 与 `results/gt_matrix_*`、`gt_losses_*` 的精确训练行、查询列和重训链。C2/C10 的三条重训链与三种主模型种子不混同；共享 validation 输入不伪作三次独立数据测量。

默认保留现有有效主模型和 TracInCP/GAS 所需中间模型。C2/C10 当前子集模型为各 3×64 个，AB2 为 32 个。旧 `checkpoints/subsets/fm/cifar2`、额外 `_r1` 和早期同名 DDPM 子集模型不因历史登记存在而自动迁移。

**DDPM 数据口径须单独说明：** 当前 `cifar2_das` 导入工件包含 128×5,000 mask 和 128×1,000 响应，匹配 100-query 与扩展 1,000-query 分析需按原列 ID 选取。投影分数、曲率分数和 response 各自的 query ID 数组必须同时保留并显式对齐；不得对所有矩阵直接取 `[:, :100]` 冒充共同查询。不能把论文概述中的“64 CIFAR subsets”直接用来改写这些已有工件。现有 native checkpoint、导入特征、响应和 mask 已进入相应选取家族。另保留 `das_archive_identity` 的四个精确文件（共 19,218 字节），相对 `_Data/raw/das_archive/` 为：

- `_code_DAS/CIFAR2/data/indices/5000-0.5/idx-train.pkl`（5,000 行，14,990 字节）与 `idx-val.pkl`（1,000 行，2,995 字节）。这两个有序原 CIFAR 索引定义训练行与验证 query 位置，不能排序或由 C2-5k 的另一个 split 替代。
- `CIFAR2/saved/5000-0.5/ddpm/ddpm_42/unet/config.json`（825 字节）与 `scheduler/scheduler_config.json`（408 字节）。保存模型/扩散身份，同时保证迁移后具有 `resolve_root()` 要求的 `CIFAR2/` 与 `_code_DAS/` 两个顶层子目录。

`prepare_benchmarks.py` 的 DDPM validation ID 推导和后续 curvature `_load_ds` 均依赖这些索引，并需 `regeneration` 已选的 CIFAR cache（或同版本原始数据）。已经选入的 `checkpoints/cifar2_das/ddpm_cond/seed_42/final.pt` 自含 `unet_config`、`unet_state`、`scheduler_config`；`build_das_ddpm` 从 checkpoint 重建，不读取旧绝对 archive 地址，因此不必再迁入 153,355,601 字节的原始权重副本。仍不整体迁移其余约 74 GiB archive。最小闭包支持现有 score/GT 分析与后续曲率，但不承诺 `import-das --force` 从头重导全部特征/GT 或重建原始训练链。

**ArtBench 原始图像闭包：** 现有规则完整保留 post_impressionism、ukiyo_e 两种风格各 5,000 train + 1,000 test，共 12,000 图片。必须保留原文件名和完整两类训练候选池：loader 先按文件名排序，再以 RandomState(42) 每类抽 2,500；若只复制抽样后的 5,000 训练图，会改变索引/重采样结果。loader 使用固定风格 ID `[4,9]`，不依赖其他八类目录。

当前仍有 C10 TracInCP/GAS seeds123/456 的两轨共 800 个查询输入缺项，必须保留 missing 状态。AB2 IF 现有分片/大型 query cache 在 `pending` 中，需完整合并与验收后更新规则。加权 D-TRAK 的独立 1,000 张学习查询属于训练权重的输入，不能混入正式 gen/val 评测；AbU+ 的 query 梯度 MC250 和训练损失测量 10 点是另外两个配置，NDA 也单列未完成。

### E-DEL / S1 / S2

当前效用主表为 `results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv`，包含两种读出的 800 行；其中四候选×50 查询×两预算的 400 行是原生删除效用。相应 400 个删除模型位于：

`checkpoints/counterfactual/fm/cifar2_5k/{fmas,dtrak_T100,das_native_sq,ekfac_if}_k{300,1000}/...`。

保留原始删除 mask、配置、测量及模型。最终 BA 只是选择哪个原有候选干预，不能把 BA 筛选集合说成已执行的删除集合。旧的两方法可行性 panel 仍被当前附录采用，因此 `results/ba_lds_feasibility_20260920/j3-a1/` 作为 `historical_appendix` 保留；不拿它填主实验的新表。

### M1 / M3 / S3 / S4

保留四核心方法原始分数和 GT；当前 M3 十三构造幂曲线还明确保留 C2 seeds42/123/456 双轨的 `ekfac_mean` / `ekfac_msl2` 原始分数与已有元数据，只作为机制读出诊断，不加入主表方法集合。M3/S3 的同配置 DAS 读出及中心化等控制主要位于 `results/lds_nextwave_20260920/j2/a1/transforms/`。`lds_e1e5_wave1_20260917` 只选当前需要的 corrected e3b/e4 表及原生分数符号统计，不整体搬入已退役的 E1/A2 和 E3a 扰动实验。

原 `Codes/Figure/out/json` 下仍被当前图表消费的九份科学汇总数据映射到 `_Data/results/paper_figures/legacy/`。公共 Figure 渲染器通过 `BALDS_DATA_ROOT/results/paper_figures/legacy` 读取；`Codes/Figure/out/json` 的兼容链接供未修改的 Paper 脚本读取同一数据。不能在删去旧 Figure 代码后留下悬空依赖。Paper 内现成图表和图片已直接复制，迁移分析数据是为后续重建。

### M2a 完整 R16

R0–R3 读取 `results/e3c_full_repeats_20260918/repeat_0..3/`，R4–R15 读取 `results/narrative_wave_20260921/incoming/xuchang3/R-{A,B,C}/`，最终统计位于 `.../analysis/m2a_r16_20260921/`。合计 64 个 `(5000,16)` 分数矩阵；保留配置、身份和独立 pilot 来源。

旧 R4 独立结果表虽然退役，**被 R16 复用的 R0–R3 原始分数仍必须迁移**。本清单只选分数、输入和身份，不自动搬入 R4 大型 query/row 缓存或旧 R4 结果表。反之，M2b 的 W-A/W-B 和 MC250/500/1000 区域干预已经退役，不迁移整个 `narrative_wave`。

机器消费的 `PILOT_M2A_FOUR_TRACKS_V1.json` 原在 `_Data/reports/narrative_gpu_wave_2026-09-21/`，现在映射到 `Reports/migration/source_evidence/`；运行前使用下述 relocation 工具产生引用新地址的运行副本。

### E-SOURCE

最终 CFM 共同 200-query、13 方法表位于 `results/cfm_if_common200_closeout_20260922/`，FMAS 复用的矩阵位于 `results/narrative_wave_20260921/analysis/es_c_20260921/fmas_raw/`。保留 validation-only 选参的 `e5_common200_20260920` 评分块、查询 ID 和选参曲线，而不是只保存 13 个均值。

CFM400 是仍保留的附录实验；DDPM 使用更新后的 500 查询中的 100 validation / 400 test。二者和 CFM common200 不是相同查询集合。保留当前 `cifar10_inj4` 的来源行真值、CFM/DDPM 查询 tensor、审阅 manifest、选择结果与当前候选图像。排除备份图片、旧 DDPM v1 的独立结果，以及只服务于 ES-L 学入发生率/ES-D 来源删除的 clean 模型生成池。

## 5. 不迁移的内容

M2b 区域采样、ES-L/ES-D、旧 E1/A2、D1–D4 随机性/扰动分析、完整 R4 独立表和额外 DAS R4、旧 prediction-layer/positive-square 迁移实验、soft BA 候选，以及退役 Tweedie/投影变体，均不因为拥有产物而进入新项目。

不整体复制 `_Data/raw/das_archive`、ArtBench 外来概念注入池、整个 `results/`、整个 `reports/`、机器台账、进度文件、锁文件、通用 launcher 和旧数据库备份。某个退役实验与现役实验共享的真实上游按现役依赖保留，R16 是一个已明确处理的例子。

完整 SD3.5 Medium 外部模型单列 inventory。公共仓库须给出模型标识/版本和可配置缓存位置；不能把本机已有 45 GiB 模型目录当作所有下载代码者默认拥有的文件，也不能将其默认打包进 Git。

## 6. 新地址与代码对应

训练、特征、曲率、分数、mask、响应等核心产物保持 `_Data` 下相对路径和既有 artifact kind / run identity，改的是配置的数据根。这能让新 `balds.artifacts` 寻址层读取旧数组，而不通过一轮搬目录改变科学身份。新 `models/data/attribution/workflows/evaluation` 代码使用传入的数据根和逻辑输入角色，不按历史 hostname 决定计算。

实际发生路径变化的条目已经在 TSV 的两列中展开：历史报告 → `Reports/migration/source_evidence/`，旧 Figure JSON → `_Data/results/paper_figures/legacy/`。所有其他文件也记录明确的源/目标相对地址。

绝对路径有三种处理方式：

- 当前可执行配置由新工作流按数据根生成；不保留 `/home/...` 作为运行默认值。`BALDS_DATA_ROOT` 可显式指定运行数据根。
- 历史收据、`code_version`、执行主机名和原命令保持原样，注明是历史证据。
- 会被读取的历史 JSON package/manifest 用 `relocate-json` 生成独立 runtime copy，原件不改。旧 `.pt/.npz` 内部的历史来源字符串不盲改；其数值/身份通过新读入接口使用。需要依赖其中旧路径的消费者应改为显式输入路径。

`relocate-json` 对 BA 的两份 manifest 另外生成明确的新接口字段：每个方法必须有 `score_space=native` 或 `das-presquare`。DAS 的 `scores` 指向原 t，字段强制为 `das-presquare`，由新 `balds` 评价器解释为 fit(t)/aggregate(t²)；移除有歧义的旧 `score_representation` / `ba_input_space` 字段。BA 文件路径转换为相对数据根的路径。历史原件不改，**这一步只生成正确的新运行输入，不计算、认证或改标签任何旧结果**。必须使用新 BA 工作流输出到独立的 `results/paper/...`。AB2 若只保存原生平方，必须先按同配置 λ=1 恢复 t；不得用有符号平方根猜测其原符号。转换器将已知 AB2 panel 的 `n_subsets` 明确修正为 32，替代旧 manifest 的 64（旧读入器通过 min-slicing 实际只用了 32）。当前默认 runtime 输入列表只包含已确认的 C2/C10 V3 BA 输入；旧全平台 manifest 作为来源证据保留，需完成 AB2 t 的来源确认后才加入可执行转换清单。

## 7. `manifest.db` 的处理

现有源库的 registry 既包含缺失的历史文件，也遗漏许多后来直接生成的现代结果。当前总条数、417 个缺失路径及本次选中却未登记的路径列表写在 `inventory.json.registry`。因此不能整体复制数据库来代表目的地，也不能因 registry 没有记录就丢弃正式结果。

复制后，`registry` 子命令从只读源库提取“已选中、目标实际存在、相对路径未改变、字节数符合既有登记”的行，保留原有 item key、code version、已有 blob hash 和 upstream 字段。它不重算 hash、不创造未有的科学身份，也不改造旧 upstream 成虚假的完整依赖图。未登记工件继续由本报告的完整文件清单与其实际配置/验收证据说明；新 `balds` 工作流后续写入的新产物才按新运行身份登记。

数据库筛选不是内容验收。保留已有 hash 是保留历史元数据，不能声称本次重新验证了其内容。跨迁移的普通核对采用精确路径、文件数、大小/mtime，以及选定消费者的数组维度、query/train ID、参数和结果口径检查。现有源文件无删改。

## 8. CFA 执行完成后的实际命令

下面路径是本次任务已赋值的本地位置。工具本身只使用参数，因此移到其他机器时替换 `--source` / `--target` 即可；无需机器别名。**本次只执行过 inventory/plan，没有执行 copy、registry 或 relocation 的写入模式。**

先接收剩余实验和正确 DAS 重聚合的实际产物，将 `selection.json` 中相应 pending 条目改为已确认的精确来源/目的地。保留旧口径证据，不在旧 V3 路径上覆盖新结果。然后刷新并检查：

```bash
cd /path/to/BA-LDS
python Reports/migration/migrate_artifacts.py inventory --source /path/to/CFA
python Reports/migration/migrate_artifacts.py plan --source /path/to/CFA --target /path/to/BA-LDS
```

已有完整查询映射的结果分析/图表可选择较小的 analysis 层；重新运行 DDPM 面板准备器还需 regeneration 中的 CIFAR cache 或等价公共原始数据。完整保留重新测量/评分所需模型与特征时使用默认两层。下列两行分别是**默认 dry-run** 和**未来真正复制**，本轮均未做复制：

```bash
python Reports/migration/migrate_artifacts.py copy --source /path/to/CFA --target /path/to/BA-LDS --profiles analysis,regeneration
python Reports/migration/migrate_artifacts.py copy --source /path/to/CFA --target /path/to/BA-LDS --profiles analysis,regeneration --execute
```

可选保留历史来源与 DAS 拟合缓存时，再单独明确选择：

```bash
python Reports/migration/migrate_artifacts.py copy --source /path/to/CFA --target /path/to/BA-LDS --profiles provenance,optional_fit_cache
python Reports/migration/migrate_artifacts.py copy --source /path/to/CFA --target /path/to/BA-LDS --profiles provenance,optional_fit_cache --execute
```

工具不会删除源文件或覆盖目标不同内容。如果源文件在 inventory 后又变化，或目标同名文件的大小/mtime 不同，会报告对应路径；先刷新/明确冲突内容，再继续，不通过忽略冲突伪装完成。复制使用 `copy2` 保存原始 mtime；核对只报告元数据相符，未宣称逐字节相同。

复制完成后核对已选择层，查看 registry 筛选计划，再建立目的地索引：

```bash
python Reports/migration/migrate_artifacts.py verify --source /path/to/CFA --target /path/to/BA-LDS --profiles analysis,regeneration
python Reports/migration/migrate_artifacts.py registry --source /path/to/CFA --target /path/to/BA-LDS
python Reports/migration/migrate_artifacts.py registry --source /path/to/CFA --target /path/to/BA-LDS --execute
```

索引目标已存在时不会覆盖；新项目开始写入之后不能再次把源数据库整体压回去。最后生成需要的运行 JSON 副本：

```bash
python Reports/migration/migrate_artifacts.py relocate-json --source /path/to/CFA --target /path/to/BA-LDS
python Reports/migration/migrate_artifacts.py relocate-json --source /path/to/CFA --target /path/to/BA-LDS --execute
```

副本落在 `_Data/runtime_manifests/<原源相对路径>`，命令会逐个报告地址。历史 JSON 和新副本都有清晰用途，不能把路径适配当作一次新的科学验收。运行报告使用新 `balds` 命令继续完成正确 DAS 聚合、缺失方法和论文表格重建。
