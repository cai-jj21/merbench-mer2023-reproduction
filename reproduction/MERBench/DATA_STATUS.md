# MERBench 数据与实验状态

日期：2026-10-14

## 官方数据入口

论文官方代码：

```text
https://github.com/zeroQiaoba/MERTools/tree/master/MERBench
```

官方 README 给出的 MER2023 预处理特征入口：

```text
store path: ./dataset/mer2023-dataset-process
link: https://pan.baidu.com/s/1l2yrWG3wXHjdRljAk32fPQ
password: uds2
```

已确认百度网盘提取码有效。未登录状态下直接调用百度下载 API 曾返回 `errno=2`，因此最终由用户使用百度网盘客户端下载。

## 本地数据

用户下载位置：

```text
D:\BaiduNetdiskDownload\feature.zip
```

本地复现数据目录：

```text
reproduction/MERBench/data/mer2023-dataset-process
```

已整理出的文件：

```text
label-6way.npz
transcription-engchi-polish.csv
features/
```

已解压出的官方特征目录，每个目录约 5030 个 `.npy` 文件：

```text
Baichuan-13B-Base-FRA
Baichuan-13B-Base-UTT
chinese-hubert-large-FRA
chinese-hubert-large-UTT
clip-vit-large-patch14-FRA
clip-vit-large-patch14-UTT
```

## 划分与标签

`label-6way.npz` 读取结果：

| split | 含义 | neutral | angry | happy | sad | worried | surprise | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train_corpus | Train&Val | 871 | 822 | 689 | 383 | 462 | 146 | 3373 |
| test1_corpus | MER-MULTI | 88 | 105 | 93 | 50 | 57 | 18 | 411 |
| test2_corpus | MER-NOISE | 123 | 98 | 87 | 40 | 52 | 12 | 412 |
| test3_corpus | MER-SEMI | 166 | 183 | 169 | 257 | 45 | 14 | 834 |

注意：`test3_corpus` 的 valence 标签包含 `-10` 占位值，因此 MER-SEMI 的 MSE 不能和 MER-MULTI/MER-NOISE 按同一含义解读。

## 特征核验

高性能 UTT 特征已成功读取：

| feature | example shape |
|---|---:|
| `chinese-hubert-large-UTT` | `[1, 1024]` |
| `Baichuan-13B-Base-UTT` | `[1, 5120]` |
| `clip-vit-large-patch14-UTT` | `[1, 768]` |

FRA 特征也已抽样读取成功，可用于 MulT 等序列融合模型。

核验命令：

```powershell
python .\reproduction\MERBench\scripts\verify_mer2023_features.py `
  --data-root .\reproduction\MERBench\data\mer2023-dataset-process `
  --features chinese-hubert-large-UTT Baichuan-13B-Base-UTT clip-vit-large-patch14-UTT
```

## 已完成实验

本地正式结果均使用 seed 2026，除 LMF 和三组单模态外，其余已保存 5 个 fold 的最佳 checkpoint。

| 类型 | 实验 | 状态 |
|---|---|---|
| 论文融合基线 | Attention high | 完成 |
| 论文融合基线 | LMF high | 完成，早期运行未保存 checkpoint |
| 论文融合基线 | TFN high | 完成 |
| 论文融合基线 | MISA high stable | 完成，默认 MISA 曾 NaN，稳定超参版完成 |
| 论文融合基线 | MulT high FRA | 完成，正式五折，FRA 特征 |
| 单模态 | audio / text / visual | 完成 |
| 改进 | balanced class weight | 完成 |
| 改进 | gated residual Attention 结构改进 | 完成，正式五折 |
| 消融 | drop audio / drop text / drop visual | 完成 |
| smoke test | MulT FRA 1 epoch | 完成，仅证明可运行，不作为正式论文结果 |
| 多 seed 统计 | Attention seeds 2026/2027/2028 | 完成，均值和标准差已生成 |

MER-MULTI/test1 关键结果：

| 实验 | WAF | ACC | UAR | MSE |
|---|---:|---:|---:|---:|
| Attention high | 80.45 | 80.78 | 75.78 | 0.6253 |
| LMF high | 80.02 | 80.05 | 74.91 | 0.7023 |
| MISA high stable | 80.00 | 80.05 | 75.11 | 0.6591 |
| TFN high | 80.49 | 80.78 | 75.35 | 0.6934 |
| Attention + class weight | 81.56 | 81.51 | 78.45 | 0.6327 |
| Drop audio | 74.54 | 74.45 | 70.15 | 0.9129 |
| Drop text | 76.74 | 76.89 | 72.31 | 0.6763 |
| Drop visual | 75.22 | 75.18 | 71.08 | 0.8886 |

汇总结果：

```text
reproduction/MERBench/results/local_reproduction_summary.csv
reproduction/MERBench/results/local_reproduction_summary.md
```

## 当前结论

官方预提取特征入口可用，本机已经完成下载、解压、特征读取核验和一组覆盖较完整的 MER2023/MER-MULTI 本地复现实验。由于当前特征包只包含 high UTT/FRA 特征，本项目应表述为“完成 MERBench/MER2023 的核心本地复现”，不能表述为“严格完整复现论文全部实验”。Attention 已完成三 seed 统计，其他正式基线以 seed 2026 为主。

尚未完成：

- Table X top-k/medium/low 特征组合复现，原因是下载包没有对应官方特征池。
- 重新提取 CLIP、HuBERT、Baichuan 特征。
