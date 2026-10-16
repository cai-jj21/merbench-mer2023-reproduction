# MERBench / MER2023 复现实验

本目录用于复现论文 **MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition** 中 MER2023 / MER-MULTI 相关实验。当前选题严格锁定 MERBench，不使用 FER2013、FER+ 或其他数据集冒充 MER-MULTI。

GitHub 仓库：<https://github.com/cai-jj21/merbench-mer2023-reproduction>

当前状态：已使用官方代码和官方百度网盘提供的 `mer2023-dataset-process` 预提取特征，完成单模态、Attention/LMF/MISA/TFN/MulT 融合基线、类别加权改进、门控残差 Attention 结构改进、三类模态缺失消融，以及 Attention 固定超参三 seed 统计。MulT FRA 100 epoch 五折结果和结构改进的 100 epoch 五折结果已生成并纳入汇总表。

## 目录

- `data/mer2023-dataset-process/`: MER2023 标签、转写和预提取特征。
- `scripts/verify_mer2023_features.py`: 核验标签、划分、标签分布和特征维度。
- `scripts/summarize_merbench_results.py`: 从官方 `.npz` 结果生成 CSV/JSON/Markdown、分类报告和混淆矩阵。
- `scripts/build_summary_tables.py`: 汇总所有本地实验到 `results/local_reproduction_summary.*`。
- `scripts/aggregate_attention_multiseed.py`: 汇总固定 Attention 超参的多 seed MER-MULTI 结果。
- `scripts/backfill_run_configs.py`: 从旧 `.npz` 结果为历史运行补生成 `run_config.json`。
- `scripts/run_merbench_experiments.ps1`: Windows PowerShell 分阶段运行脚本。
- `configs/hyper_stable.yaml`: MISA/TFN/MulT 稳定超参配置。
- `configs/attention_stable.yaml`: 多 seed Attention 固定超参配置。
- `scripts/run_attention_multiseed.ps1`: 顺序运行 seed 2027/2028 的 Attention 多 seed 实验。
- `configs/planned_experiments.json`: 已完成和待完成实验清单。
- `patches/`: 本地对官方 MERBench 代码的修改文件和补丁。
- `references/MERTools/MERBench/toolkit/models/gated_attention.py`: 门控残差 Attention 结构改进。
- `results/official_mer2023_mermulti.csv`: 论文表格中的官方 MER-MULTI 对照值。
- `results/local_reproduction_summary.csv`: 本地复现实验总表。
- `reports/merbench_report_draft.md`: 中文实验报告初稿。
- `reports/merbench_report_final.md`: 中文实验报告最终版。
- `reports/merbench_report.tex`: LaTeX 实验报告源文件。
- `reports/merbench_report.pdf`: 编译后的 LaTeX 实验报告，供验收阅读。

为控制仓库体积并避免上传原始特征，GitHub 仓库不包含 `data/`、`downloads/`、`runs/`、`tmp/`、checkpoint、日志和 `.npz` 预测文件。数据准备和本地运行方式见 `DATA_PREP.md`、`DATA_STATUS.md` 与本 README。

官方代码仍需从作者仓库获取：<https://github.com/zeroQiaoba/MERTools/tree/master/MERBench>。本地修改的文件已放在 `patches/`，包括类别加权、模态缺失处理和门控残差 Attention。

## 数据准备

用户下载的官方特征包：

```text
D:\BaiduNetdiskDownload\feature.zip
```

已解压并整理到：

```text
reproduction/MERBench/data/mer2023-dataset-process
```

目录应包含：

```text
label-6way.npz
transcription-engchi-polish.csv
features/
```

如果需要从头重新解压，PowerShell 命令如下：

```powershell
New-Item -ItemType Directory -Force `
  .\reproduction\MERBench\data\mer2023-dataset-process | Out-Null

Expand-Archive `
  -LiteralPath 'D:\BaiduNetdiskDownload\feature.zip' `
  -DestinationPath .\reproduction\MERBench\data\mer2023-dataset-process `
  -Force
```

最终特征目录至少应包含：

```text
features/chinese-hubert-large-UTT
features/Baichuan-13B-Base-UTT
features/clip-vit-large-patch14-UTT
features/chinese-hubert-large-FRA
features/Baichuan-13B-Base-FRA
features/clip-vit-large-patch14-FRA
```

## 实验环境

本机实际完成实验的环境：

```text
Windows PowerShell
Python 3.12.3
torch 2.6.0+cu124
CUDA 12.4
GPU NVIDIA GeForce RTX 4060 Laptop GPU
numpy 1.26.4
scikit-learn 1.7.2
pandas 2.2.2
```

额外安装：

```powershell
python -m pip install thop
```

## 数据核验

```powershell
python .\reproduction\MERBench\scripts\verify_mer2023_features.py `
  --data-root .\reproduction\MERBench\data\mer2023-dataset-process `
  --features chinese-hubert-large-UTT Baichuan-13B-Base-UTT clip-vit-large-patch14-UTT
```

已核验划分：

| split | 含义 | 样本数 |
|---|---|---:|
| train_corpus | Train&Val | 3373 |
| test1_corpus | MER-MULTI | 411 |
| test2_corpus | MER-NOISE | 412 |
| test3_corpus | MER-SEMI | 834 |

高性能 UTT 特征维度：

| modality | feature dir | shape |
|---|---|---:|
| audio | `chinese-hubert-large-UTT` | `[1, 1024]` |
| text | `Baichuan-13B-Base-UTT` | `[1, 5120]` |
| visual | `clip-vit-large-patch14-UTT` | `[1, 768]` |

## 运行命令

从工作区根目录执行：

```powershell
cd 'C:\Users\88911\Desktop\异构智能体'
```

分阶段运行：

```powershell
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage verify
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage attention
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage lmf
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage tfn
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage misa
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage unimodal
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage classweight
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage improved
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage ablation
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 -Stage summary
```

MulT 正式实验如果中断，可从旧 run 的已完成折继续：

```powershell
.\reproduction\MERBench\scripts\run_merbench_experiments.ps1 `
  -Stage mult_resume -Epochs 100 -BatchSize 16 -Seed 2026
```

固定超参的多 seed Attention：

```powershell
.\reproduction\MERBench\scripts\run_attention_multiseed.ps1 `
  -Seeds 2027,2028 -Epochs 100 -BatchSize 32

python .\reproduction\MERBench\scripts\aggregate_attention_multiseed.py `
  --root .\reproduction\MERBench
```

说明：`-Stage misa` 使用 `configs/hyper_stable.yaml` 中的稳定超参。官方默认随机超参在本机曾出现 NaN，失败日志保留在 `runs/misa_high_seed2026.log`。

重建总表：

```powershell
python .\reproduction\MERBench\scripts\build_summary_tables.py `
  --root .\reproduction\MERBench
```

## MER-MULTI/test1 结果

论文值和本地值分开列出。论文 WAF 来自 `results/official_mer2023_mermulti.csv`，本地值来自实际日志和 `.npz` 汇总。除特别说明外，本地实验均为 seed 2026。

| 实验 | 论文 WAF | 论文 MSE | 本地 WAF | 本地 ACC | 本地 UAR | 本地 MSE |
|---|---:|---:|---:|---:|---:|---:|
| Audio HUBERT-large | 69.91 | 0.97 | 68.59 | 68.61 | 63.56 | 0.9345 |
| Text Baichuan-13B | 62.54 | 1.63 | 60.68 | 60.83 | 55.36 | 1.6614 |
| Visual CLIP-large | 62.69 | 1.12 | 60.99 | 62.29 | 51.87 | 1.1297 |
| Attention high | 83.03 | 0.62 | 80.45 | 80.78 | 75.78 | 0.6253 |
| LMF high | 83.44 |  | 80.02 | 80.05 | 74.91 | 0.7023 |
| MISA high stable | 82.42 |  | 80.00 | 80.05 | 75.11 | 0.6591 |
| TFN high | 82.52 |  | 80.49 | 80.78 | 75.35 | 0.6934 |
| Attention + class weight |  |  | 81.56 | 81.51 | 78.45 | 0.6327 |
| Gated residual Attention |  |  | 81.21 | 81.27 | 77.74 | 0.6369 |
| Drop audio |  |  | 74.54 | 74.45 | 70.15 | 0.9129 |
| Drop text |  |  | 76.74 | 76.89 | 72.31 | 0.6763 |
| Drop visual |  |  | 75.22 | 75.18 | 71.08 | 0.8886 |

主结果文件：

```text
reproduction/MERBench/results/local_reproduction_summary.csv
reproduction/MERBench/results/local_reproduction_summary.json
reproduction/MERBench/results/local_reproduction_summary.md
```

每个实验的分类报告和混淆矩阵在：

```text
reproduction/MERBench/results/<experiment>/
```

训练日志和 checkpoint 在：

```text
reproduction/MERBench/runs/
```

## 已修改官方代码

修改位置均在 `references/MERTools/MERBench`：

- `config.py`: 增加 `MERBENCH_MER2023_ROOT` 环境变量，默认指向本工作区数据目录。
- `main-release.py`: 增加固定随机种子、保存预测、保存每折最佳 checkpoint、class weight、drop modality 参数。
- `main-release.py`: 增加按折续跑、已完成折检查点恢复和每个 run 的 `run_config.json`。
- `toolkit/utils/loss.py`: `CELoss` 支持类别权重。
- `toolkit/data/feat_data.py`: 支持将指定模态特征置零，用于模态缺失消融。
- `toolkit/models/gated_attention.py`: 新增轻量门控残差 Attention，使用归一化模态可靠性门控和跨模态残差支路。

## 复现限制

当前结果不能表述为“严格完整复现论文全部内容”，原因如下：

- 当前 `feature.zip` 只包含 high UTT/FRA 特征，未包含论文 Table X 中更完整的 medium/low/top-k 特征池。
- 论文报告多次运行的均值和标准差，本地已对 Attention 固定超参完成 seed 2026/2027/2028 统计，其他正式基线仍以 seed 2026 为主。
- MISA 默认官方随机超参在本机出现 NaN，因此报告中使用稳定超参版并明确标注。
- MulT FRA 100 epoch 五折正式结果已生成；本地结果仍与论文报告值分开列出。
- Table X Top-k/medium/low 的本地结果需要官方对应特征池，不能用论文均值替代。
- 本地未重新提取 CLIP、HuBERT、Baichuan 特征，复现范围是“基于官方预提取特征的融合训练与评测”。
