# 共用上游：从数据到原始分数与响应

这些命令是从当前CLI和正式配方整理的**新环境重新执行流程**，不是声称本次运行过的现场日志。若只重现图表，应先迁入已验收分数、mask和响应，直接运行对应分析；无需重新训练。科学参数来源为终稿附录C.1/E.1、源 `Codes/cfa/conf/defaults.yaml` 及各实验任务书。

## 环境和路径

从下载后的仓库根目录执行。所有命令均使用当前目录推导路径，无服务器别名、conda绝对路径或自动SSH依赖。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './Codes[train,latent,projection,figures]'
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

CUDA版PyTorch安装依据本机驱动选择；原正式完整R16使用torch2.6.0+cu124。当前论文评价器为SNR-LDS，完整数组评价与汇总见[E-BENCH](E_BENCH.md)；`balds-run evaluate`是原方法Full LDS评价入口。不要将更换环境后的逐位相等当作默认承诺。CIFAR数据下载缓存位于数据根内；AB2还需原始ArtBench imagefolder与合法获取的SD3.5 Medium权重。代码安装不会自动附带研究数据/权重；本轮已迁数据范围以实际迁移清单为准，重新下载代码时需另行获得输入工件。

## 配方与身份

| 平台键 | 内容 | 模型 / 模型seed | 子集/响应 |
|---|---|---|---|
| `cifar2_5k` | automobile/horse各2500；32²；固定抽样seed42 | unconditional 36M U-Net FM；42/123/456 | Bernoulli keep .5，M64；三重训链42/123/456 |
| `cifar10_v2` | 完整50000 CIFAR-10 | class-conditional同架构，无CFG；42/123/456 | Bernoulli keep .5，M64；三链 |
| `artbench2_256` | post-impressionism/ukiyo-e各2500；256² | SD3.5 Medium LoRA rank/alpha128；seed42 | M32；一链；损失三噪声 |
| `cifar2_das` | 独立DAS CIFAR-2存档 | 38.3M DDPM；seed42 | 只用存档subset0–63；固定半集；3重训×3噪声均值 |

CIFAR主模型和子集均训200epochs、AdamW、batch128、lr1e-4、wd1e-6、cosine、10%warmup、dropout.1、随机hflip。子集步数按实际训练集大小换算。中间25/50/75%检查点与final是TracInCP/GAS的必要输入。AB2为100epochs、有效batch64、microbatch8、lr3e-4、wd1e-6、warmup500、prompt `a {style} painting`，VAE后验均值及独立flip潜变量缓存。

每模型gen/val各100。gen为Euler100，噪声seed=123+模型seed，即165/246/579；val由类均衡固定索引构成，无生成命令。响应损失CIFAR使用1000个t∈[.05,.95]、噪声0/1/2；AB2使用100个t。DDPM1000扩展是单独机制分析，不替代BA共同100。

## 1. 训练主模型与生成查询

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
for SEED in 42 123 456; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 train --dataset cifar2_5k --process cfm --uncond --seed "$SEED"
  balds-run --data-root "$BALDS_DATA" --device cuda:0 generate --dataset cifar2_5k --process cfm --uncond --seed "$SEED" --Q 100 --ode-steps 100
  balds-run --data-root "$BALDS_DATA" --device cuda:0 --set train.p_uncond=0 train --dataset cifar10_v2 --process cfm --seed "$SEED"
  balds-run --data-root "$BALDS_DATA" --device cuda:0 generate --dataset cifar10_v2 --process cfm --seed "$SEED" --Q 100 --ode-steps 100
done
```

产物：`checkpoints/<dataset>/cfm_{uncond|cond}/seed_<seed>/` 和 `generations/<dataset>/cfm_{uncond|cond}/seed_<seed>/samples.pt`。已有匹配工件时CLI复用；不要对正式唯一输入使用`--force`。

## 2. 子集重训 → query损失 → 三链响应

先建立固定mask，再运行训练/测损循环：

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
balds-run --data-root "$BALDS_DATA" --device cpu subsets masks --dataset cifar2_5k --process cfm --uncond --seed 42
balds-run --data-root "$BALDS_DATA" --device cpu subsets masks --dataset cifar10_v2 --process cfm --seed 42
```


下面依次覆盖C2/C10三条重训链。C10显式p_uncond=0；条件身份及mask在全部交叉query计算中保持一致。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
for DATASET in cifar2_5k cifar10_v2; do
CONDITION=()
CFG=()
if [ "$DATASET" = cifar2_5k ]; then CONDITION=(--uncond); else CFG=(--set train.p_uncond=0); fi
for CHAIN in 42 123 456; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 "${CFG[@]}" subsets train --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$CHAIN" --rank 0 --world 1
  balds-run --data-root "$BALDS_DATA" --device cuda:0 "${CFG[@]}" subsets losses --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$CHAIN" --query-type val --e-seeds 0,1,2
  for SEED in 42 123 456; do
    balds-run --data-root "$BALDS_DATA" --device cuda:0 "${CFG[@]}" subsets losses --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --chain "$CHAIN" --query-type gen --e-seeds 0,1,2
  done
done
for SEED in 42 123 456; do
  for TRACK in gen val; do
    balds-run --data-root "$BALDS_DATA" --device cpu "${CFG[@]}" subsets gt --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --query-type "$TRACK" --chains 42,123,456 --e-seeds 0,1,2
  done
done
done
```

mask按数据集共享、固定seed42，不能每模型重抽。val不接受`--chain`；其查询跨种子相同，三份链自己的val损失已经覆盖所有交叉计算。GT输出`results/gt_matrix_fm_<dataset>[_val]_seed_<seed>.npy`，shape(M,100)；mask为`subsets/<dataset>_masks.pkl`，是keep指示，评价转换为D=1−K。

## 3. 投影梯度/图像特征 → 方法原始分数

以下C2/C10各三种子完整双轨。`both`只覆盖train和指定的一种query，必须额外执行val。投影p4096、seed0；每样本梯度逐MC先L2标准化，再平均及投影。D-TRAK读出mean(v²)，FM DAS读出mean(v)，TRAK读出MSE loss。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
for DATASET in cifar2_5k cifar10_v2; do
  CONDITION=()
  if [ "$DATASET" = cifar2_5k ]; then CONDITION=(--uncond); fi
  for SEED in 42 123 456; do
  for FEAT in das_T100 dtrak_T100 trak_T100 pixel clip; do
    balds-run --data-root "$BALDS_DATA" --device cuda:0 featurize --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --feat "$FEAT" --split both --query-type gen
    balds-run --data-root "$BALDS_DATA" --device cuda:0 featurize --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --feat "$FEAT" --split query --query-type val
  done
  for TRACK in gen val; do
    for METHOD in das_T100 dtrak_T100 trak_T100 grad_dot_T100 grad_cos_T100 relative_if_T100 renorm_if_T100 pixel_dot pixel_cos clip_dot clip_cos; do
      balds-run --data-root "$BALDS_DATA" --device cuda:0 score --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --query-type "$TRACK" --method "$METHOD"
      balds-run --data-root "$BALDS_DATA" --device cpu evaluate --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --query-type "$TRACK" --method "$METHOD"
    done
  done
done
done
```

投影kernel网格为`[.01,.02,.05,.1,.2,.5,1,2,5,10,20,30,40,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000]`；逆核除以mean(abs(Kinv))。现稿保留原LDS选参的开发集属性，不把这些数说成独立test选参。要逐数字重现历史表，优先使用原保存分数和其实际选参记录；新默认网格不自动等于每个历史版本。

DAS磁盘的`das_T100/scores.npy`可为有符号一次项；**不凭文件名判断已平方**。原生比较需要平方一次，SNR噪声拟合需要保留原符号。具体清单见E-BENCH。

TracInCP/GAS使用25/50/75%与final四checkpoint无权平均；下面从当前模型记录读取实际step。缺失中间检查点就停止，不以少于四个checkpoint继续当同一正式方法。C10 TracInCP/GAS三seed现均完成。DDPM四格汇总已入稿，但对应逐querycoverage尚未提供，不能将原生feature生产与SNR结果覆盖混为一谈。Journey为十点mean-readout adaptation，只对gen。

```bash
export BALDS_DATA="$PWD/_Data"
python - <<'PY'
import os,json,subprocess
root=os.environ['BALDS_DATA']
for ds in ['cifar2_5k','cifar10_v2']:
  for seed in [42,123,456]:
    identity=['--dataset',ds,'--process','cfm','--seed',str(seed)]
    if ds=='cifar2_5k': identity+=['--uncond']
    gpu=['balds-run','--data-root',root,'--device','cuda:0']
    cpu=['balds-run','--data-root',root,'--device','cpu']
    cp=json.loads(subprocess.check_output(cpu+['checkpoints']+identity,text=True))
    assert cp['tracin_terms']==4, (ds,seed,cp)
    for step in cp['steps']:
      for track,split in [('gen','both'),('val','query')]:
        subprocess.run(gpu+['featurize']+identity+['--feat','trak_T100','--ckpt-step',str(step),'--split',split,'--query-type',track],check=True)
    for track in ['gen','val']:
      for method in ['tracincp_T100','gas_T100']:
        subprocess.run(gpu+['score']+identity+['--method',method,'--query-type',track],check=True)
        subprocess.run(cpu+['evaluate']+identity+['--method',method,'--query-type',track],check=True)
    subprocess.run(gpu+['generate']+identity+['--trajectory','--journey-points','10'],check=True)
    subprocess.run(gpu+['featurize']+identity+['--feat','journey','--split','query','--query-type','gen'],check=True)
    subprocess.run(gpu+['score']+identity+['--method','journey_trak_T100','--query-type','gen'],check=True)
    subprocess.run(cpu+['evaluate']+identity+['--method','journey_trak_T100','--query-type','gen'],check=True)
PY
```

此前阶段3的final TRAK特征须先存在。此段按完整新生产输入运行；如果复用历史已有分数，跳过本段即可。

## 4. 曲率 → FMAS / EK-FAC IF

终稿FMAS是`fmas_raw`。公开版`fmas`别名已收束到`fmas_raw`；为消除历史歧义，命令显式使用`fmas_raw`。不能拿旧README的repeats/gamma-scan六步链代替当前方法。

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
for DATASET in cifar2_5k cifar10_v2; do
  CONDITION=()
  if [ "$DATASET" = cifar2_5k ]; then CONDITION=(--uncond); fi
  for SEED in 42 123 456; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 ekfac fit --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED"
  for TRACK in gen val; do
    for METHOD in fmas_raw ekfac_if; do
      balds-run --data-root "$BALDS_DATA" --device cuda:0 ekfac score --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --query-type "$TRACK" --method "$METHOD"
      balds-run --data-root "$BALDS_DATA" --device cpu evaluate --dataset "$DATASET" --process cfm "${CONDITION[@]}" --seed "$SEED" --query-type "$TRACK" --method "$METHOD"
    done
  done
done
done
```

两方法共享Conv2d/Linear EK-FAC GGN/MC-Fisher，含bias，不含GroupNorm/class embedding。训练梯度train mode、query eval mode；每侧MC250；因子/特征值修正各125次。FMAS stratified-antithetic抽样和逐层阻尼rho；IF iid和统一lambda。FMAS标准rho网格`1e-4,1e-3,1e-2,1e-1,1`，C2机制固定`.01`。IF网格`1e-13`至`1e-4`十档；历史部分C2数据使用旧网格，其选择必须由原工件确认。只计算原始bilinear，不附加Tweedie或EB层。

C10将dataset换为`cifar10_v2`并去掉uncond；AB2换为`artbench2_256`、只seed42，并按保留的内存配置使用query/row分块。AB2 IF已完整验收，gen/val分别以Full LDS选lambda1e-7/1e-6；不能把FMAS分数改名为IF。

## 5. AB2与DDPM的特殊准备

```bash
export BALDS_DATA="$PWD/_Data"
export BALDS_DATA_ROOT="$BALDS_DATA"
balds-run --data-root "$BALDS_DATA" --device cuda:0 latents --dataset artbench2_256
balds-run --data-root "$BALDS_DATA" --device cuda:0 train --dataset artbench2_256 --process cfm --seed 42
balds-run --data-root "$BALDS_DATA" --device cuda:0 generate --dataset artbench2_256 --process cfm --seed 42 --Q 100 --ode-steps 100
balds-run --data-root "$BALDS_DATA" --device cpu subsets masks --dataset artbench2_256 --process cfm --seed 42
balds-run --data-root "$BALDS_DATA" --device cuda:0 subsets train --dataset artbench2_256 --process cfm --seed 42 --rank 0 --world 1
for TRACK in gen val; do
  balds-run --data-root "$BALDS_DATA" --device cuda:0 subsets losses --dataset artbench2_256 --process cfm --seed 42 --query-type "$TRACK" --e-seeds 0,1,2
  balds-run --data-root "$BALDS_DATA" --device cpu subsets gt --dataset artbench2_256 --process cfm --seed 42 --query-type "$TRACK" --e-seeds 0,1,2
done
balds-run --data-root "$BALDS_DATA" --device cpu import-das --root "$BALDS_DATA/raw/das_archive" --model
```

AB2后续特征/评分调用同阶段3/4、单seed42、不带uncond，pixel由256²面积平均到64²，CLIP ViT-B/32为512维。DDPM archive的conditional路径段只是历史存储身份，**不能传uncond**；输入本身是无条件DDPM。只导入支持的DAS/D-TRAK/L1读出；TRAK存档布局未经确认，不自动用于正式表。DDPM重新训练存档上游不是本仓可一条命令复现的自产链，应保留存档来源、查询、模型及损失。

## 交付和未决

每次报告记录：数据键、模型seed、process/conditional、query IDs、mask IDs、有效配置、代码版本、模型/生成/特征/曲率/score/response路径、实际运行时间和缺项。正式模型和大型产物尚未随代码公开；公共数据下载、受限权重许可与实验计算仍需使用者准备。不能把`pip install`成功描述为论文全部数字已经复现。
