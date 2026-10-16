# Data Preparation

本复现优先使用作者提供的预提取特征，避免在 8GB 显存 RTX 4060 Laptop GPU 上重新提取 CLIP、HuBERT、Baichuan 等大模型特征。

## 下载来源

官方 MERBench README 给出的 MER2023 预处理特征：

```text
link: https://pan.baidu.com/s/1l2yrWG3wXHjdRljAk32fPQ
password: uds2
store path: ./dataset/mer2023-dataset-process
```

用户已下载到：

```text
D:\BaiduNetdiskDownload\feature.zip
```

## 目标目录

最终目录应为：

```text
reproduction/MERBench/data/mer2023-dataset-process/
  label-6way.npz
  transcription-engchi-polish.csv
  features/
    chinese-hubert-large-UTT/
    Baichuan-13B-Base-UTT/
    clip-vit-large-patch14-UTT/
    chinese-hubert-large-FRA/
    Baichuan-13B-Base-FRA/
    clip-vit-large-patch14-FRA/
```

## 解压命令

从工作区根目录执行：

```powershell
cd 'C:\Users\88911\Desktop\异构智能体'

New-Item -ItemType Directory -Force `
  .\reproduction\MERBench\data\mer2023-dataset-process | Out-Null

Expand-Archive `
  -LiteralPath 'D:\BaiduNetdiskDownload\feature.zip' `
  -DestinationPath .\reproduction\MERBench\data\mer2023-dataset-process `
  -Force
```

如果解压后目录层级多了一层，例如：

```text
reproduction/MERBench/data/mer2023-dataset-process/feature/...
```

或：

```text
reproduction/MERBench/data/mer2023-dataset-process/features/features/...
```

需要整理为上文目标目录中的 `features/<feature-name>` 结构，再运行核验脚本。

## 标签文件

如果目标目录中缺少标签和转写文件，可从官方仓库复制：

```powershell
Copy-Item .\references\MERTools\MERBench\dataset\mer2023-dataset-process\label-6way.npz `
  .\reproduction\MERBench\data\mer2023-dataset-process\

Copy-Item .\references\MERTools\MERBench\dataset\mer2023-dataset-process\transcription-engchi-polish.csv `
  .\reproduction\MERBench\data\mer2023-dataset-process\
```

## 核验命令

```powershell
python .\reproduction\MERBench\scripts\verify_mer2023_features.py `
  --data-root .\reproduction\MERBench\data\mer2023-dataset-process `
  --features chinese-hubert-large-UTT Baichuan-13B-Base-UTT clip-vit-large-patch14-UTT
```

核验通过时应看到：

```text
train_corpus: 3373
test1_corpus: 411
test2_corpus: 412
test3_corpus: 834
chinese-hubert-large-UTT: [1, 1024]
Baichuan-13B-Base-UTT: [1, 5120]
clip-vit-large-patch14-UTT: [1, 768]
```

## 不可替代项

- 不使用 FER2013、FER+ 或其他图像分类数据集替代 MER-MULTI。
- 不把 Hugging Face gated 原始数据入口当作本作业唯一方案，除非已接受 EULA 且能复现 MERBench 同等特征流程。
- 不在未实际运行训练/测试脚本时填写本地复现值。

