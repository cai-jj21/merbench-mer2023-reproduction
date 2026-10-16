# MERBench MER2023 多模态情感识别复现实验报告

课程：情感计算  
姓名：__________  学号：__________  
选题：MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition  
作者：Zheng Lian 等  
实验日期：2026 年 10 月  

## 1 研究背景和问题定义

多模态情感识别需要同时利用语音、文本和视觉信息，预测样本的离散情绪类别以及连续情绪维度。不同研究往往使用不同的数据划分、特征提取器、融合模型和评价指标，因此即使报告了相似名称的结果，也不一定能够直接比较。MERBench 的目标是在统一数据、特征、训练和评价流程下比较多种多模态情感识别方法，并分析特征选择、融合策略、跨语料库迁移、噪声鲁棒性和预训练等因素。

本文定义的主要任务为：给定一个中文视频片段的音频特征 $x^a$、文本特征 $x^t$ 和视觉特征 $x^v$，预测六分类情绪标签 $y^e$ 以及连续 valence 标签 $y^v$。本次作业的主测试集为 MER-MULTI，分类标签为 neutral、angry、happy、sad、worried、surprise。

本复现严格锁定 MERBench，不使用 FER2013、FER+ 或其他数据集替代 MER-MULTI，也没有更换成其他论文。

## 2 论文实验范围和本次复现范围

论文完整的 MERBench 实验族包括单模态特征比较、多模态融合、跨语料库实验、标点鲁棒性、加性噪声鲁棒性、是否需要预训练，以及是否需要下游微调等内容。本次实验优先完成老师要求的 MER2023/MER-MULTI 核心部分：单模态基线、融合基线、结构改进和模态消融。

### 2.1 MER-MULTI 论文结果行统计

以下统计只针对本项目整理的 MER-MULTI 参考结果文件，不代表论文全部实验族的总数量。

| 论文表格 | 内容 | 结果行数 | 本地直接复现 |
|---|---|---:|---:|
| Table II | high/medium 单模态 | 6 | 3 |
| Table III | 两种官方多模态特征组合 | 2 | 1 |
| Table IX | 五种融合模型的 high/medium 结果 | 10 | 5 |
| Table X | Top-k 特征选择 | 6 | 0 |
| **合计** | 论文参考表中的 MER-MULTI 结果 | **24** | **9** |

因此，本地严格对应论文结果行的覆盖率为 $9/24=37.5\%$。已经完成的 9 行包括 high 特征的三个单模态结果、一个 high 多模态特征组合结果，以及 Attention、LMF、MISA、TFN、MulT 五个 high 融合结果。Table IX 的 high 融合结果已经完成 5/5。

### 2.2 本地实验配置统计

本地结果总表中目前有 13 个命名实验配置：5 个论文融合基线、3 个单模态实验、2 个改进实验和 3 个模态缺失消融实验。Attention 另使用固定超参运行了 seed 2026、2027、2028 三次统计，因此实际保存了额外的多 seed 运行结果。每个配置还分别记录交叉验证、MER-MULTI、MER-NOISE 和 MER-SEMI 的结果，这些测试划分输出不应被重复计数为新的独立模型。

## 3 MERBench 方法和数据流程

本地复现流程如下：

1. 使用官方 `label-6way.npz` 和转写文件读取 MER2023 标签及数据划分。
2. 使用作者提供的预提取音频、文本和视觉特征，避免在 8GB 显存设备上重新提取大型模型特征。
3. 将三种模态特征输入官方融合模型或本地结构改进模型。
4. 同时优化离散情绪分类和 valence 回归任务。
5. 在 Train&Val 上进行五折交叉验证，并在 MER-MULTI、MER-NOISE、MER-SEMI 上测试。
6. 保存每次运行的配置、训练日志、每折最佳模型、预测结果、分类报告和混淆矩阵。

MER2023 的本地划分如下。该划分与论文数据统计一致：Train&Val 有 3373 个样本，MER-MULTI 有 411 个样本，MER-NOISE 有 412 个样本，MER-SEMI 有 834 个带离散标签样本；MER-SEMI 还包含大量未标注样本。

| split | 含义 | 样本数 | 本地使用方式 |
|---|---|---:|---|
| train_corpus | Train&Val | 3373 | 五折训练和验证 |
| test1_corpus | MER-MULTI | 411 | 主要对照测试集 |
| test2_corpus | MER-NOISE | 412 | 噪声鲁棒性测试输出 |
| test3_corpus | MER-SEMI | 834 | 半监督相关测试输出 |

MER-MULTI 的六类标签分布为：neutral 88、angry 105、happy 93、sad 50、worried 57、surprise 18。该分布不均衡，尤其 `surprise` 样本较少，因此报告同时使用 WAF 和 UAR 观察分类性能。

## 4 特征和模型

### 4.1 官方预提取特征

本地可验证读取的 high UTT 特征为：

| 模态 | 特征 | 本地目录 | 单样本维度 |
|---|---|---|---:|
| audio | chinese HuBERT large | `chinese-hubert-large-UTT` | 1024 |
| text | Baichuan-13B-Base | `Baichuan-13B-Base-UTT` | 5120 |
| visual | CLIP ViT-L/14 | `clip-vit-large-patch14-UTT` | 768 |

MulT 正式实验使用对应 FRA 特征：`chinese-hubert-large-FRA`、`Baichuan-13B-Base-FRA` 和 `clip-vit-large-patch14-FRA`。UTT 表示句子级特征，FRA 表示用于帧级对齐序列融合的特征。

### 4.2 官方 Attention 融合基线

对于每种模态，官方 Attention 模型先使用独立编码器得到统一隐空间表示：

$$h_a=E_a(x^a),\quad h_t=E_t(x^t),\quad h_v=E_v(x^v).$$

UTT 特征使用 MLP 编码器，FRA 特征使用 LSTM 编码器。三个模态表示拼接后输入注意力网络，得到三个模态的融合分数：

$$s=W_2\phi(W_1[h_a;h_t;h_v]),$$

然后按官方实现将三个隐向量进行加权融合，并通过两个输出头分别预测离散情绪和 valence：

$$\hat y^e=F_e(z),\quad \hat y^v=F_v(z).$$

训练目标为分类损失和回归损失之和，官方模型选择指标为：

$$M=WAF-0.25\times MSE.$$

### 4.3 本地门控残差 Attention 改进

为了实现模型结构层面的改进，本文在官方 Attention 的三路编码器和双任务输出头基础上增加门控融合模块。设原始注意力分数为 $s$，首先计算归一化注意力权重：

$$\alpha=softmax(s).$$

再由拼接特征估计每个样本的模态可靠性：

$$r=\sigma(G([h_a;h_t;h_v])).$$

将两者相乘后重新归一化：

$$w_m=\frac{\alpha_m r_m}{\sum_j\alpha_jr_j+\epsilon}.$$

最终融合表示为：

$$z=\sum_{m\in\{a,t,v\}}w_mh_m+\beta R([h_a;h_t;h_v]),$$

其中 $R$ 是轻量 MLP 残差支路，beta 为可学习缩放参数，初始化为 0.1。该设计有两个目的：一是根据样本内容动态降低不可靠模态的影响，二是保留简单加权和不能表达的跨模态交互。该改进只增加融合层参数，不重新训练特征提取器，适合当前显存条件。

### 4.4 类别加权改进

对于类别加权版本，分类损失使用基于训练集类别频数反比得到的 balanced weight：

$$w_c=\frac{N}{C\cdot n_c},$$

其中 $N$ 为训练样本数，$C$ 为类别数，$n_c$ 为第 $c$ 类样本数。回归损失仍使用普通 MSE。该实验用于检验类别不均衡修正的效果，与门控残差结构改进分开运行。

## 5 实验环境和运行设置

| 项目 | 设置 |
|---|---|
| 操作系统 | Windows PowerShell |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 约 8GB |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| NumPy | 1.26.4 |
| scikit-learn | 1.7.2 |
| pandas | 2.2.2 |
| 默认随机种子 | 2026 |
| UTT batch size | 32 |
| MulT FRA batch size | 16 |
| 训练轮数 | 100 |
| 交叉验证 | Train&Val 五折 |

本地环境与作者仓库中的旧版环境并不完全一致，因此本报告将结果称为“基于官方特征和官方代码结构的本地复现值”，不把它们写成论文原始报告值。

## 6 评价指标

- **WAF**：weighted F1，按类别样本数加权的 F1。
- **ACC**：分类准确率。
- **UAR**：macro recall，即各类别召回率的平均值，更关注少数类。
- **MSE**：valence 回归均方误差，越低越好。
- **模型选择指标**：$WAF-0.25\times MSE$。

MER-SEMI 的 valence 标签为 `-10` 占位值，因此本地结果中只报告其分类指标，不将其 MSE 与 MER-MULTI 或 MER-NOISE 的 MSE 比较。

## 7 论文结果与本地复现结果

下表主要列出论文 Table II、Table III 和 Table IX 中与当前 high 特征直接对应的结果。论文值来自本地保存的 `results/official_mer2023_mermulti.csv`，本地值来自 seed 2026 的真实运行结果。论文 WAF 和本地 WAF/ACC/UAR 均以百分数表示。

| 实验 | 论文来源 | 论文 WAF | 论文 MSE | 本地 WAF | 本地 ACC | 本地 UAR | 本地 MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Audio HUBERT-large | Table II | 69.91 | 0.97 | 68.59 | 68.61 | 63.56 | 0.9345 |
| Text Baichuan-13B | Table II | 62.54 | 1.63 | 60.68 | 60.83 | 55.36 | 1.6614 |
| Visual CLIP-large | Table II | 62.69 | 1.12 | 60.99 | 62.29 | 51.87 | 1.1297 |
| Attention high | Table IX / Table III | 83.03 | 0.62 | 80.45 | 80.78 | 75.78 | 0.6253 |
| LMF high | Table IX | 83.44 | - | 80.02 | 80.05 | 74.91 | 0.7023 |
| MISA high stable | Table IX | 82.42 | - | 80.00 | 80.05 | 75.11 | 0.6591 |
| TFN high | Table IX | 82.52 | - | 80.49 | 80.78 | 75.35 | 0.6934 |
| MulT high FRA | Table IX | 82.94 | - | 80.39 | 80.54 | 77.62 | 0.8183 |

### 7.1 结果分析

1. 三个单模态本地结果与论文值处于同一数量级，audio 结果最接近论文 WAF。
2. 多模态融合明显优于单模态，说明音频、文本和视觉之间存在互补信息。
3. 五个 high 融合基线的本地 WAF 约为 80%，比论文报告值低约 2.4 到 3.4 个百分点。
4. Attention 的本地 MSE 0.6253 与论文 Table III high 特征组合的 0.62 接近，但分类 WAF 低于论文值。
5. 可能的差异来源包括论文多次运行均值和标准差、本地使用单 seed、官方随机超参搜索没有完全复刻、MISA 使用了稳定超参，以及 PyTorch/CUDA 版本不同。

### 7.2 Attention 多随机种子稳定性

固定 `configs/attention_stable.yaml`，在 MER-MULTI 上运行三个 seed：

| seed | WAF | ACC | UAR | MSE |
|---:|---:|---:|---:|---:|
| 2026 | 80.45 | 80.78 | 75.78 | 0.6253 |
| 2027 | 83.06 | 82.97 | 80.19 | 0.6311 |
| 2028 | 82.71 | 82.73 | 78.66 | 0.6157 |
| **均值 ± 标准差** | **82.07 ± 1.15** | **82.16 ± 0.98** | **78.21 ± 1.83** | **0.6240 ± 0.0064** |

Attention 的 WAF 标准差为 1.15 个百分点，UAR 标准差为 1.83 个百分点，说明单次 seed 结果存在一定波动。报告中的主要基线对照仍使用 seed 2026，三 seed 均值作为稳定性补充，不替代论文报告值。

## 8 单模态、多模态和消融实验

以下为 MER-MULTI/test1 的本地结果：

| 类型 | 实验 | WAF | ACC | UAR | MSE |
|---|---|---:|---:|---:|---:|
| 单模态 | audio only | 68.59 | 68.61 | 63.56 | 0.9345 |
| 单模态 | text only | 60.68 | 60.83 | 55.36 | 1.6614 |
| 单模态 | visual only | 60.99 | 62.29 | 51.87 | 1.1297 |
| 多模态 | Attention high | 80.45 | 80.78 | 75.78 | 0.6253 |
| 多模态 | LMF high | 80.02 | 80.05 | 74.91 | 0.7023 |
| 多模态 | MISA high stable | 80.00 | 80.05 | 75.11 | 0.6591 |
| 多模态 | TFN high | 80.49 | 80.78 | 75.35 | 0.6934 |
| 多模态 | MulT high FRA | 80.39 | 80.54 | 77.62 | 0.8183 |
| 消融 | Drop audio | 74.54 | 74.45 | 70.15 | 0.9129 |
| 消融 | Drop text | 76.74 | 76.89 | 72.31 | 0.6763 |
| 消融 | Drop visual | 75.22 | 75.18 | 71.08 | 0.8886 |

Drop audio 后 WAF 比完整 Attention 下降 5.91 个百分点，Drop visual 后下降 5.23 个百分点，Drop text 后下降 3.71 个百分点。在当前特征和模型设置下，音频和视觉对离散分类的贡献更明显。Drop text 的 MSE 仍为 0.6763，说明 valence 回归可能仍较依赖音频和视觉信息。

## 9 算法改进和改进结果

### 9.1 类别加权改进

类别加权版本的 MER-MULTI 结果如下：

| 实验 | WAF | ACC | UAR | MSE |
|---|---:|---:|---:|---:|
| Attention high | 80.45 | 80.78 | 75.78 | 0.6253 |
| Attention + class weight | 81.56 | 81.51 | 78.45 | 0.6327 |

相对于原始 Attention，WAF 提升 1.11 个百分点，ACC 提升 0.73 个百分点，UAR 提升 2.67 个百分点，MSE 增加 0.0074。该结果说明类别加权对于不均衡六分类任务有效，尤其改善宏平均召回；但分类权重调整不保证 valence 回归误差同时下降。

### 9.2 门控残差 Attention

| 实验 | WAF | ACC | UAR | MSE |
|---|---:|---:|---:|---:|
| Attention high | 80.45 | 80.78 | 75.78 | 0.6253 |
| Gated residual Attention | 81.21 | 81.27 | 77.74 | 0.6369 |

相对于原始 Attention，门控残差结构的 WAF 提升 0.76 个百分点，ACC 提升 0.49 个百分点，UAR 提升 1.96 个百分点，MSE 增加 0.0116。它优于原始 Attention，但 WAF 和 UAR 略低于类别加权版本。因此，本实验支持“门控残差结构有正向效果”，但不支持“门控残差结构在所有指标上优于类别加权”。

两种改进的侧重点不同：类别加权修改训练损失，针对类别不均衡；门控残差 Attention 修改融合结构，针对样本级模态可靠性和跨模态交互。为区分各自贡献，本轮没有直接叠加两种改进。

## 10 误差分析

以 `Attention + class weight` 在 MER-MULTI 上的混淆矩阵为例：

| 真实类别 \\ 预测类别 | neutral | angry | happy | sad | worried | surprise |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 68 | 3 | 7 | 4 | 6 | 0 |
| angry | 8 | 88 | 2 | 4 | 2 | 1 |
| happy | 8 | 0 | 84 | 0 | 0 | 1 |
| sad | 2 | 2 | 1 | 44 | 1 | 0 |
| worried | 10 | 0 | 0 | 7 | 40 | 0 |
| surprise | 1 | 1 | 2 | 1 | 2 | 11 |

主要现象如下：

- `happy` 和 `angry` 的识别相对稳定，主要因为测试样本数量较多且情绪线索较明显。
- `surprise` 只有 18 个测试样本，类别加权后正确数为 11，召回率约为 61.11%，仍然容易受单个样本影响。
- `worried` 容易被预测为 `neutral` 或 `sad`，说明低唤醒负向情绪之间存在边界模糊。
- 类别加权后少数类表现有所改善，但整体 MSE 没有同步下降，体现了分类和 valence 回归之间的任务权衡。

## 11 复现产物和运行方法

官方代码和本地复现代码位于：

```text
references/MERTools/MERBench
reproduction/MERBench
```

本地复现代码、配置、结果表和本报告源文件已公开在 GitHub：
<https://github.com/cai-jj21/merbench-mer2023-reproduction>

重要产物如下：

| 内容 | 路径 |
|---|---|
| 最终报告 Markdown | `reproduction/MERBench/reports/merbench_report_final.md` |
| 数据准备说明 | `reproduction/MERBench/DATA_PREP.md` |
| 数据核验状态 | `reproduction/MERBench/DATA_STATUS.md` |
| 运行 README | `reproduction/MERBench/README.md` |
| 运行脚本 | `reproduction/MERBench/scripts/run_merbench_experiments.ps1` |
| 官方结果参考表 | `reproduction/MERBench/results/official_mer2023_mermulti.csv` |
| 本地结果总表 | `reproduction/MERBench/results/local_reproduction_summary.csv` |
| 多 seed 汇总 | `reproduction/MERBench/results/attention_multiseed_summary.csv` |
| 改进结果 | `reproduction/MERBench/results/gated_residual_attention_high_seed2026_formal/gated_residual_attention_high_seed2026_formal.csv` |
| 改进模型代码 | `references/MERTools/MERBench/toolkit/models/gated_attention.py` |
| 日志和 checkpoint | `reproduction/MERBench/runs/` |

从工作区根目录执行数据核验：

```powershell
python .\reproduction\MERBench\scripts\verify_mer2023_features.py `
  --data-root .\reproduction\MERBench\data\mer2023-dataset-process `
  --features chinese-hubert-large-UTT Baichuan-13B-Base-UTT clip-vit-large-patch14-UTT
```

运行官方融合基线：

```powershell
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage attention
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage lmf
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage misa
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage tfn
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage mult_resume -Epochs 100 -BatchSize 16
```

运行改进、消融和汇总：

```powershell
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage improved -Epochs 100 -BatchSize 32
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage classweight -Epochs 100 -BatchSize 32
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage ablation -Epochs 100 -BatchSize 32
python .\reproduction\MERBench\scripts\build_summary_tables.py --root .\reproduction\MERBench
```

## 12 复现限制和未完成部分

1. 当前 `feature.zip` 只包含 high UTT/FRA 特征目录，没有 medium/low 特征池，因此无法完成论文 Table II、Table III、Table IX 中对应的 medium 结果，也无法完成 Table X 的 Top-k 特征选择。
2. 本地对 Attention 完成了三个 seed 的固定超参统计，LMF、MISA、TFN、MulT 等其他正式基线主要使用 seed 2026，不能冒充论文的多次运行均值和标准差。
3. 官方默认 MISA 随机超参在本机出现 NaN，正式报告使用稳定超参版本，并保留默认运行失败日志。
4. 本地没有重新提取 CLIP、HuBERT 和 Baichuan 特征，因此复现范围是“基于作者官方预提取特征的融合训练与评测”。
5. 本轮没有完成论文的跨语料库、标点鲁棒性、原始加性噪声生成流程、预训练和下游微调等完整实验族。MER-NOISE 和 MER-SEMI 的本地测试输出已经保存，但不能据此宣称完整复现论文全部鲁棒性和半监督实验。
6. 本地环境为 Windows、Python 3.12、PyTorch 2.6，而作者 README 中的环境较旧，版本差异可能影响数值结果。

因此，本项目的准确表述是：**完成 MERBench/MER2023 核心融合实验的本地复现，以及一组结构改进和消融实验；尚未完成论文全部实验族的严格复现。**

## 13 结论

本复现使用官方 MERBench 代码结构和官方 MER2023 预提取特征，在 RTX 4060 Laptop 8GB 环境下完成了可运行、可追溯的 MER-MULTI 核心实验。单模态实验表明，单独使用音频、文本或视觉的性能明显低于多模态融合；五种融合基线均表现出多模态互补优势。模态缺失消融显示，音频和视觉缺失对分类性能的影响大于文本缺失。

改进方面，类别加权版本取得了当前最高的 WAF 81.56 和 UAR 78.45；门控残差 Attention 取得 WAF 81.21 和 UAR 77.74，相比原始 Attention 有稳定的正向提升。两种改进分别从类别不均衡和模态可靠性两个角度改善模型，结果具有可解释性，但仍需要更多随机种子和完整特征池实验验证泛化性。

本地严格对应论文 MER-MULTI 参考结果完成 9/24 项，已满足本课程作业要求的论文基线、改进实验和消融实验要求；medium/low、Top-k、跨语料库及其他论文实验族明确列为尚未完成部分。

## 14 参考文献

1. Zheng Lian, Licai Sun, Yong Ren, Hao Gu, Haiyang Sun, Lan Chen, Bin Liu, Jianhua Tao. *MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition*. arXiv:2401.03429, 2024. https://arxiv.org/html/2401.03429
2. MERBench 官方代码仓库：https://github.com/zeroQiaoba/MERTools/tree/master/MERBench
3. MERBench 官方 README 中的 MER2023 预提取特征入口：https://pan.baidu.com/s/1l2yrWG3wXHjdRljAk32fPQ，提取码：`uds2`

