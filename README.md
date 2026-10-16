# MERBench / MER2023 Reproduction

本仓库用于复现课程指定论文 **MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition** 的 MER2023 / MER-MULTI 核心实验。

## 内容

- `reproduction/MERBench/`: 本地复现实验代码、配置、结果表和中文 LaTeX 实验报告。
- `reproduction/MERBench/patches/`: 本地对官方 MERBench 代码所做的必要修改和补丁。
- 官方代码仍通过作者仓库获取：<https://github.com/zeroQiaoba/MERTools/tree/master/MERBench>。

详细运行说明见 [`reproduction/MERBench/README.md`](reproduction/MERBench/README.md)。报告文件如下：

- [`merbench_report.tex`](reproduction/MERBench/reports/merbench_report.tex)
- [`merbench_report.pdf`](reproduction/MERBench/reports/merbench_report.pdf)

## 数据与大文件

仓库没有上传官方预提取特征、`feature.zip`、训练 checkpoint、运行日志、临时文件和 `.npz` 预测文件。请按 [`DATA_PREP.md`](reproduction/MERBench/DATA_PREP.md) 准备作者提供的 MER2023 特征后再运行。

本地复现范围、论文结果与本地结果的区别，以及尚未完成的论文实验族，均在中文报告中明确说明。仓库不包含官方预提取特征和训练产物。
