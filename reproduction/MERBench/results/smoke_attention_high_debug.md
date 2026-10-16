# Smoke Run: Attention High Feature Set

日期：2026-10-13

目的：验证官方 MERBench 训练链路能在 Windows + RTX 4060 Laptop GPU 上读取官方预提取特征、完成 5 折 debug 训练并保存结果。

这不是正式复现实验结果，不能写入论文对照表。本次运行使用 `--debug`，每个 split 只取前 100 条样本，且只训练 1 epoch。

## Command

```powershell
$env:MERBENCH_MER2023_ROOT='C:\Users\88911\Desktop\异构智能体\reproduction\MERBench\data\mer2023-dataset-process'

python -u main-release.py `
  --model='attention' `
  --feat_type='utt' `
  --dataset='MER2023' `
  --audio_feature='chinese-hubert-large-UTT' `
  --text_feature='Baichuan-13B-Base-UTT' `
  --video_feature='clip-vit-large-patch14-UTT' `
  --gpu=0 `
  --debug `
  --epochs=1 `
  --batch_size=16 `
  --num_workers=0 `
  --save_root='C:\Users\88911\Desktop\异构智能体\reproduction\MERBench\runs\smoke_attention_high_winfix'
```

## Feature Dimensions

| feature | shape |
|---|---:|
| `chinese-hubert-large-UTT` | `(1, 1024)` |
| `Baichuan-13B-Base-UTT` | `(1, 5120)` |
| `clip-vit-large-patch14-UTT` | `(1, 768)` |

## Debug Metrics From Saved Filenames

| split | weighted F1 | accuracy | valence MSE |
|---|---:|---:|---:|
| cv | 0.0728 | 0.1200 | 3.8427 |
| test1 / MER-MULTI | 0.0395 | 0.1500 | 4.1422 |
| test2 / MER-NOISE | 0.0379 | 0.1100 | 2.7701 |
| test3 / MER-SEMI | 0.1789 | 0.3400 | 96.7735 |

## Notes

- 官方脚本原始保存文件名包含 `:`，Windows 不允许该字符出现在文件名中，因此已在 `references/MERTools/MERBench/main-release.py` 增加 `sanitize_filename()`，只影响保存路径，不改变训练和评价逻辑。
- 控制台中的中文路径显示为乱码，但 Python 实际文件访问正常。
- 下一步应取消 `--debug`，固定随机种子和超参数，保存完整训练日志，再跑正式 Attention / LMF 基线。

