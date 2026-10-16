# MERBench 论文复现实验报告初稿

## 1. 论文背景和问题定义

论文 **MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition** 面向多模态情感识别评测。以往方法常因数据划分、特征提取器、融合模型、训练协议和评价指标不同而难以公平比较。MERBench 将音频、文本、视觉特征和多种融合模型放在统一流程下评估，核心任务包括离散情绪分类和连续 valence 回归。

本课程作业锁定 MERBench，不更换为 EfficientFace、AffectGPT、Emotion-LLaMA、S2D 或其他论文。本地优先复现 MER2023/MER-MULTI，使用作者提供的预提取特征，避免在 RTX 4060 Laptop 8GB 显存上重新训练或重新提取 CLIP、HuBERT、Baichuan 等大型模型特征。

## 2. MERBench 方法和数据流程

MERBench 的基本流程为：

1. 对原始视频样本进行统一预处理，得到标签、转写和数据划分。
2. 使用冻结的预训练模型提取音频、文本、视觉特征。
3. 使用统一融合模型训练离散情绪分类和 valence 回归。
4. 在固定测试集上报告 WAF、ACC、UAR、MSE 等指标。

本地使用官方仓库：

```text
references/MERTools/MERBench
```

本地数据目录：

```text
reproduction/MERBench/data/mer2023-dataset-process
```

MER2023 划分如下：

| split | 含义 | 样本数 |
|---|---|---:|
| train_corpus | Train&Val | 3373 |
| test1_corpus | MER-MULTI | 411 |
| test2_corpus | MER-NOISE | 412 |
| test3_corpus | MER-SEMI | 834 |

本报告主要对照 `test1_corpus`，即 MER-MULTI。

## 3. 特征与融合模型说明

本地使用作者提供的 high UTT 特征：

| modality | paper feature | local feature dir | shape |
|---|---|---|---:|
| audio | HUBERT-large | `chinese-hubert-large-UTT` | `[1, 1024]` |
| text | Baichuan-13B | `Baichuan-13B-Base-UTT` | `[1, 5120]` |
| visual | CLIP-large | `clip-vit-large-patch14-UTT` | `[1, 768]` |

完成的实验类型：

| 类型 | 实验 |
|---|---|
| 单模态 | audio、text、visual |
| 论文融合基线 | Attention、LMF、MISA、TFN |
| 论文融合基线 | MulT（FRA 序列融合，正式五折已完成） |
| 改进实验 | Attention + balanced class weight |
| 结构改进实验 | Gated residual Attention |
| 消融实验 | drop audio、drop text、drop visual |

MISA 需要特别说明：官方默认随机超参在本机运行时出现 NaN，失败日志保留在 `runs/misa_high_seed2026.log`。因此正式报告采用 `configs/hyper_stable.yaml` 中的稳定超参版本，记为 `MISA high stable`，不把它伪装成论文默认设置。

## 4. 实验环境

| 项目 | 本地环境 |
|---|---|
| OS | Windows PowerShell |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 约 8GB |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| numpy | 1.26.4 |
| scikit-learn | 1.7.2 |
| pandas | 2.2.2 |
| seed | 2026 |
| epoch | 100 |
| batch size | UTT 通常为 32；MulT FRA 正式运行使用 16 |

官方环境文件为 `references/MERTools/MERBench/environment.yml`。本地没有完全重建官方旧环境，但所有训练、测试、结果保存均已实际运行。

## 5. 数据划分和评价指标

MER-MULTI/test1 标签分布：

| label | count |
|---|---:|
| neutral | 88 |
| angry | 105 |
| happy | 93 |
| sad | 50 |
| worried | 57 |
| surprise | 18 |

评价指标：

- WAF：weighted F1，是论文表格中主要离散情绪指标。
- ACC：accuracy。
- UAR：macro recall，用于观察类别不均衡下的平均召回。
- MSE：valence mean squared error。
- 官方模型选择指标：`WAF - 0.25 * MSE`。

## 6. 论文结果与本地复现结果对照

下表只列 MER-MULTI/test1。论文值来自 `results/official_mer2023_mermulti.csv`，本地值来自 seed 2026 实际运行结果。论文 WAF 是百分数，本地 WAF/ACC/UAR 也换算为百分数。

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
| Gated residual Attention | 本地结构改进 | - | - | 81.21 | 81.27 | 77.74 | 0.6369 |

观察：

- 三组单模态结果与论文同量级，audio 最接近论文 WAF。
- Attention high 的 MSE 与论文 Table III high 特征组 MSE 接近，但 WAF 低约 2.58 个百分点。
- LMF、MISA stable、TFN 的 WAF 均在 80% 左右，低于论文 Table IX 报告值约 2.4 到 3.4 个百分点。
- 主要差异来源可能包括：单 seed 而非多次均值、官方随机搜索超参未完全复刻、MISA 使用稳定超参、PyTorch/CUDA 版本不同、论文可能使用多次重复后择优或平均。

## 7. 单模态、多模态和消融实验

本地 MER-MULTI/test1 完整结果：

| 类型 | 实验 | WAF | ACC | UAR | MSE |
|---|---|---:|---:|---:|---:|
| 单模态 | audio only | 68.59 | 68.61 | 63.56 | 0.9345 |
| 单模态 | text only | 60.68 | 60.83 | 55.36 | 1.6614 |
| 单模态 | visual only | 60.99 | 62.29 | 51.87 | 1.1297 |
| 多模态 | Attention high | 80.45 | 80.78 | 75.78 | 0.6253 |
| 多模态 | LMF high | 80.02 | 80.05 | 74.91 | 0.7023 |
| 多模态 | MISA high stable | 80.00 | 80.05 | 75.11 | 0.6591 |
| 多模态 | TFN high | 80.49 | 80.78 | 75.35 | 0.6934 |
| 消融 | Drop audio | 74.54 | 74.45 | 70.15 | 0.9129 |
| 消融 | Drop text | 76.74 | 76.89 | 72.31 | 0.6763 |
| 消融 | Drop visual | 75.22 | 75.18 | 71.08 | 0.8886 |

结论：

- 多模态融合显著优于任一单模态，说明 MER-MULTI 中音频、文本、视觉存在互补信息。
- Drop audio 后 WAF 降至 74.54，Drop visual 后 WAF 降至 75.22，Drop text 后 WAF 降至 76.74。在当前特征组合和 Attention 模型下，音频和视觉对分类贡献更明显。
- Drop text 的 MSE 仍接近完整 Attention，说明 valence 回归对 audio/visual 仍有较强依赖。

## 8. 改进方法和改进结果

本作业实现了两种互相独立的改进。第一种是在官方 Attention 模型上加入 balanced cross entropy 类别权重，不改变模型结构。权重由 `train_corpus` 类别频数反比计算，重点提升 `sad`、`worried`、`surprise` 等低频类别的损失贡献。

第二种是门控残差 Attention 结构改进，模型文件为 `references/MERTools/MERBench/toolkit/models/gated_attention.py`。它保留官方三路 MLP 编码器和双任务输出头，在融合层加入三步：

1. 使用 Softmax 将原始 Attention 融合分数归一化为模态权重；
2. 使用样本级可靠性门控估计 audio、text、visual 的可信程度，并与 Attention 权重相乘后重新归一化；
3. 使用一个小型跨模态残差支路补充简单加权和无法表达的交互信息，残差初始缩放为 0.1，降低训练初期扰动。

该改进不重新提取 CLIP、HuBERT 或 Baichuan 特征，只增加轻量融合层，适合 8GB 显存环境。训练仍采用与官方 Attention 相同的 MER2023 划分、100 epoch、五折交叉验证和 `WAF - 0.25 * MSE` 模型选择准则。

MER-MULTI/test1 对比：

| 实验 | WAF | ACC | UAR | MSE |
|---|---:|---:|---:|---:|
| Attention high | 80.45 | 80.78 | 75.78 | 0.6253 |
| Attention + class weight | 81.56 | 81.51 | 78.45 | 0.6327 |
| Gated residual Attention | 81.21 | 81.27 | 77.74 | 0.6369 |

改进结果：

- WAF 提升 1.11 个百分点。
- ACC 提升 0.73 个百分点。
- UAR 提升 2.67 个百分点。
- MSE 从 0.6253 轻微变差到 0.6327。

因此，类别加权对离散情绪分类有效，尤其改善宏平均召回；但由于训练目标同时包含 valence 回归，分类权重调整不保证 MSE 同时改善。门控残差结构相对于原始 Attention 提升了 WAF 0.76 个百分点、ACC 0.49 个百分点、UAR 1.96 个百分点，MSE 增加 0.0116。它优于原始 Attention，但略低于类别加权版本的 WAF/UAR，因此当前证据支持“结构改进有效”，不支持“结构改进全面优于所有改进”。

从改进对比看，类别加权版本的 WAF/UAR 最高，说明 MER-MULTI 的类别不均衡是重要问题；门控残差版本的优势在于不修改损失函数，通过样本级模态可靠性建模提升融合适应性。两种策略可以作为后续组合实验，但本轮未将它们叠加，避免无法区分各自贡献。

## 9. 误差分析

以 `Attention + class weight` 在 MER-MULTI/test1 上的混淆矩阵为例：

| true \ pred | neutral | angry | happy | sad | worried | surprise |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 68 | 3 | 7 | 4 | 6 | 0 |
| angry | 8 | 88 | 2 | 4 | 2 | 1 |
| happy | 8 | 0 | 84 | 0 | 0 | 1 |
| sad | 2 | 2 | 1 | 44 | 1 | 0 |
| worried | 10 | 0 | 0 | 7 | 40 | 0 |
| surprise | 1 | 1 | 2 | 1 | 2 | 11 |

主要现象：

- `happy` 和 `angry` 的 F1 较高，分别约 88.89 和 88.44。
- `surprise` 只有 18 个测试样本，召回为 61.11%，仍是最不稳定类别之一。
- `worried` 容易被预测为 `neutral` 或 `sad`，说明该类别与低唤醒负向情绪边界模糊。
- 类别加权使 `surprise` 正确数从未加权 Attention 的 9/18 提升到 11/18，同时整体 UAR 提升，符合类别不均衡修正的预期。

## 10. 复现产物

| 类型 | 路径 |
|---|---|
| 数据准备说明 | `reproduction/MERBench/DATA_PREP.md` |
| 数据状态 | `reproduction/MERBench/DATA_STATUS.md` |
| 运行 README | `reproduction/MERBench/README.md` |
| 分阶段运行脚本 | `reproduction/MERBench/scripts/run_merbench_experiments.ps1` |
| 稳定超参 | `reproduction/MERBench/configs/hyper_stable.yaml` |
| 实验配置 | `reproduction/MERBench/configs/planned_experiments.json` |
| 官方结果表 | `reproduction/MERBench/results/official_mer2023_mermulti.csv` |
| 本地结果总表 | `reproduction/MERBench/results/local_reproduction_summary.csv` |
| 分类报告/混淆矩阵 | `reproduction/MERBench/results/<experiment>/` |
| 训练日志 | `reproduction/MERBench/runs/*.log` |
| 最优模型 | `reproduction/MERBench/runs/*/model/*.pt` |
| 每次运行配置 | `reproduction/MERBench/runs/*/run_config.json` |
| 多 seed 汇总 | `reproduction/MERBench/results/attention_multiseed_summary.*` |
| 有效 valence 审计 | `val_missing_count`、`val_mse_comparable` 字段 |

说明：LMF 和三组单模态是在加入 checkpoint 保存补丁前完成的，因此保存了日志、结果和预测，但没有最佳模型 `.pt`。Attention high、class weight、drop audio、drop text、drop visual、TFN、MISA stable 均保存了 5 折最佳 checkpoint。

## 11. 复现限制

本地结果不能表述为“严格完整复现论文全部内容”，原因如下：

1. 当前下载的 `feature.zip` 只包含 high UTT/FRA 特征目录，未包含论文中 medium/low/top-k 所需的完整特征池。
2. Attention 已完成固定超参的多 seed 统计；LMF、MISA、TFN、MulT 等其他正式基线仍以 seed 2026 本地复现值为主，不能冒充论文均值。
3. MISA 默认官方随机超参在本机出现 NaN，正式结果使用稳定超参版。
4. MulT FRA 100 epoch 五折正式运行已经完成，最终数值来自实际生成的 `.npz` 和五折最佳 checkpoint。
5. 固定 Attention 的 seed 2026/2027/2028 多 seed 均值和标准差已生成，文件位于 `results/attention_multiseed_summary.*`。
6. 本地没有重新执行 CLIP、HuBERT、Baichuan 特征提取，因此复现范围是“基于官方预提取特征的融合训练与评测”。

## 12. 结论

本地已经完成 MERBench/MER2023 的大部分核心可运行复现子集：官方预提取特征已核验，单模态 audio/text/visual 已运行，Attention/LMF/MISA/TFN/MulT 融合基线已运行，类别加权改进、门控残差 Attention 结构改进和三类模态缺失消融已完成，并生成日志、结果表、分类报告、混淆矩阵和历史实验的配置文件/最佳 checkpoint。MulT FRA 正式五折运行和结构改进的正式五折运行均已完成，结果已纳入本地总表并与论文 Table IX 分开列出。

从 MER-MULTI/test1 结果看，多模态融合显著优于单模态；类别加权使 Attention 的 WAF 从 80.45 提升到 81.56，UAR 从 75.78 提升到 78.45，但 MSE 略有变差。门控残差 Attention 使 WAF 达到 81.21、UAR 达到 77.74，证明轻量结构改进有收益，但当前不如类别加权版本。消融实验显示 audio、visual 对分类性能影响较大，text 缺失对 WAF 的影响相对较小但仍不可忽略。

后续若要更接近论文完整复现，主要还需要在获得官方完整特征池后复现 Table X 的 Top-k、medium/low 特征组合；当前硬件与下载包已支持的核心实验、MulT 五折和 Attention 三 seed 统计均已完成。

## 13. 参考文献

Zheng Lian, Licai Sun, Yong Ren, Hao Gu, Haiyang Sun, Lan Chen, Bin Liu, Jianhua Tao. **MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition**. arXiv:2401.03429, 2024.

官方代码：https://github.com/zeroQiaoba/MERTools/tree/master/MERBench

论文链接：https://arxiv.org/html/2401.03429
