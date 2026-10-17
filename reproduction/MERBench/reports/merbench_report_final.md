# MERBench MER2023 多模态情感识别复现实验报告

课程：情感计算  
姓名：蔡均炅  学号：2026210870
选题：*MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition*  
作者：Zheng Lian 等  
实验日期：2026 年 10 月

## 1 实验目标

本作业严格复现 MERBench 在 MER2023/MER-MULTI 上的核心实验。任务是根据中文视频的音频、文本和视觉特征，预测六类情绪：neutral、angry、happy、sad、worried 和 surprise，同时回归 valence。实验重点是比较单模态、三模态融合模型，并在官方 Attention 基线上加入轻量改进。

考虑到设备为 RTX 4060 Laptop GPU（约 8GB 显存），本实验直接使用作者提供的预提取特征，没有重新训练 HuBERT、CLIP 或大型语言模型。论文报告值和本地运行值始终分开列出；本地结果是使用官方特征、官方代码结构和本地随机种子得到的复现值，不冒充论文原始结果。

## 2 数据、特征与评价指标

MER2023 的本地划分与论文统计一致：Train&Val 有 3373 个样本，MER-MULTI 有 411 个样本。本报告以 MER-MULTI 为主要测试集。其六类标签数量分别为 neutral 88、angry 105、happy 93、sad 50、worried 57、surprise 18，类别明显不均衡。

| 模态 | 特征提取器 | 特征目录 | 维度 |
|---|---|---|---:|
| audio | chinese HuBERT large | `chinese-hubert-large-UTT` | 1024 |
| text | Baichuan-13B-Base | `Baichuan-13B-Base-UTT` | 5120 |
| visual | CLIP ViT-L/14 | `clip-vit-large-patch14-UTT` | 768 |

官方模型同时进行分类和 valence 回归。报告分类的 WAF（weighted F1）、ACC 和 UAR（macro recall），以及回归 MSE。综合选择指标为 `M = WAF - 0.25 x MSE`。除特别说明外，正式结果使用随机种子 2026。

## 3 论文基线与本地复现结果

本地直接对应论文 MER-MULTI 参考表的结果共有 10/24 项，覆盖率为 41.7%。其中 high 特征下的三种单模态、Table III 的 high 多任务组合、Attention、LMF、MISA、TFN、MulT 和 MMIM 六种融合结果均已完成；另外 MFN 和 GMFN 完成了全量五折 5 epoch 扩展训练，MFM/MCTN 保留了 debug 冒烟。扩展训练不等价于论文的 100 epoch 正式结果。medium/low 特征和 Top-k 特征选择没有在本轮执行。

| 实验 | 论文 WAF | 本地 WAF | 本地 ACC | 本地 UAR | 论文 MSE | 本地 MSE |
|---|---:|---:|---:|---:|---:|---:|
| Audio HUBERT-large | 69.91 | 68.59 | 68.61 | 63.56 | 0.97 | 0.9345 |
| Text Baichuan | 62.54 | 60.68 | 60.83 | 55.36 | 1.63 | 1.6614 |
| Visual CLIP | 62.69 | 60.99 | 62.29 | 51.87 | 1.12 | 1.1297 |
| High multi-task combination (Table III) | 83.02 | 80.45 | 80.78 | 75.78 | 0.62 | 0.6253 |
| Attention high | 83.03 | 80.45 | 80.78 | 75.78 | 0.62 | 0.6253 |
| LMF high | 83.44 | 80.02 | 80.05 | 74.91 | -- | 0.7023 |
| MISA high | 82.42 | 80.00 | 80.05 | 75.11 | -- | 0.6591 |
| TFN high | 82.52 | 80.49 | 80.78 | 75.35 | -- | 0.6934 |
| MulT high FRA | 82.94 | 80.39 | 80.54 | 77.62 | -- | 0.8183 |
| MMIM high stable | 81.01 | 75.96 | 76.40 | 68.72 | -- | 0.7084 |

本地结果与论文在同一数量级，但常规融合模型的 WAF 大约低 2.4--3.4 个百分点，稳定 MMIM 低 5.05 个百分点。Attention 的回归 MSE 为 0.6253，与论文的 0.62 接近；分类差距可能来自论文多次运行均值、本地只使用单一 seed、超参数搜索范围不同以及软件环境差异。因此，本地列用于复现和比较，不能直接当作论文原值。

## 4 单模态、多模态与消融

| 类型 | 实验 | WAF | ACC | UAR | MSE |
|---|---|---:|---:|---:|---:|
| 单模态 | audio | 68.59 | 68.61 | 63.56 | 0.9345 |
| 单模态 | text | 60.68 | 60.83 | 55.36 | 1.6614 |
| 单模态 | visual | 60.99 | 62.29 | 51.87 | 1.1297 |
| 融合 | Attention | 80.45 | 80.78 | 75.78 | 0.6253 |
| 融合 | LMF | 80.02 | 80.05 | 74.91 | 0.7023 |
| 融合 | MISA | 80.00 | 80.05 | 75.11 | 0.6591 |
| 融合 | TFN | 80.49 | 80.78 | 75.35 | 0.6934 |
| 融合 | MulT FRA | 80.39 | 80.54 | 77.62 | 0.8183 |
| 融合 | MMIM stable | 75.96 | 76.40 | 68.72 | 0.7084 |
| 扩展 | MFN FRA (5 epochs) | 56.52 | 61.80 | 49.05 | 0.9639 |
| 扩展 | GMFN FRA (5 epochs) | 59.35 | 62.04 | 51.25 | 0.9903 |
| 消融 | Drop audio | 74.54 | 74.45 | 70.15 | 0.9129 |
| 消融 | Drop text | 76.74 | 76.89 | 72.31 | 0.6763 |
| 消融 | Drop visual | 75.22 | 75.18 | 71.08 | 0.8886 |

三模态融合明显优于单模态：Attention 相比最好的单模态 audio，WAF 提高 11.86 个百分点，UAR 提高 12.22 个百分点，MSE 从 0.9345 降至 0.6253。去掉 audio 后 WAF 下降 5.91 个百分点，去掉 visual 后下降 5.23 个百分点，去掉 text 后下降 3.71 个百分点；在当前特征和划分下，audio 对整体识别最重要。

## 5 算法改进与结果

本次改进围绕类别不均衡和模态可靠性展开，均不增加新的特征提取开销。

### 5.1 类别加权

对分类交叉熵加入按训练集类别频率计算的逆频率权重，使少数类在损失中具有更大影响：

`L = L_reg + lambda L_cls_weighted`，其中 `w_c = N / (K N_c)`，`K=6`。

### 5.2 门控残差 Attention

在官方 Attention 融合后增加轻量门控残差模块。模型根据三种模态的隐向量计算样本级门控权重，再将门控融合结果与原 Attention 表示相加，最后送入分类和回归头。这样可以在不同样本上降低不可靠模态的影响，同时保留官方融合结构。

| 实验 | WAF | ACC | UAR | MSE |
|---|---:|---:|---:|---:|
| Attention baseline | 80.45 | 80.78 | 75.78 | 0.6253 |
| Attention + class weight | 81.56 | 81.51 | 78.45 | 0.6327 |
| Gated residual Attention | 81.21 | 81.27 | 77.74 | 0.6369 |

类别加权是本轮最有效的改进：相对 Attention，WAF 提升 1.11 个百分点，ACC 提升 0.73 个百分点，UAR 提升 2.67 个百分点；MSE 仅增加 0.0074。门控残差 Attention 的 WAF 提升 0.76 个百分点，UAR 提升 1.96 个百分点，也优于原始基线，但略低于类别加权版本。

为检查随机性，Attention 固定超参使用 seed 2026、2027、2028 重复运行，WAF 均值为 82.07，标准差为 1.15；UAR 均值为 78.21，标准差为 1.83。这组均值用于稳定性补充，不替代单 seed 对照。

## 6 简要误差分析、限制与结论

从混淆矩阵和分类报告看，audio 已经包含较强的情绪线索；融合后整体性能提高，但少数类 surprise 仍然容易被识别为 neutral 或 happy。worried 与 neutral、sad 之间也存在混淆，说明句子级特征对细粒度负面情绪的区分仍然有限。类别加权提高了 UAR，说明它主要改善了少数类召回。

分类指标和回归指标并不完全同步。类别加权和门控残差都提高了分类表现，但 MSE 略有上升，说明模型在优化离散类别边界时牺牲了一部分连续 valence 拟合。

本轮完成了 MER-MULTI 的 high 特征单模态、六种正式融合基线、MFN/GMFN 全量五折 5 epoch 扩展、三种模态缺失消融、类别加权和门控残差改进，论文参考结果覆盖 10/24。MFM/MCTN 只完成了 1 epoch debug 冒烟，尚未完成正式 100 epoch 结果；medium/low 特征、Top-k、跨语料库、完整噪声生成、预训练与下游微调等实验族也尚未完成，因此不能声称完整复现论文全部结果。

结论是：三模态融合显著优于单模态；六种 high 官方融合基线中五种常规模型在本地达到约 80% WAF，稳定 MMIM 对超参数更敏感；针对 MER-MULTI 类别不均衡的类别加权取得最高本地结果 WAF 81.56、UAR 78.45，是本次最值得保留的改进。

代码、配置、日志索引和数据准备说明：

<https://github.com/cai-jj21/merbench-mer2023-reproduction>

仓库不上传受限数据、特征压缩包或模型权重；README 说明了从本地数据目录运行的步骤，官方 MERTools 作为外部依赖保留。

## 参考文献

1. Zheng Lian et al. *MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition*. arXiv:2401.03429, 2024.
2. Zheng Lian et al. MER2023 数据集与多模态情感识别基准，见 MERBench 官方代码仓库。
3. 本地复现代码：<https://github.com/cai-jj21/merbench-mer2023-reproduction>。
