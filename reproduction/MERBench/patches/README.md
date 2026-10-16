# Official Code Changes

本目录保存本地复现实验对作者官方 MERBench 代码所做的必要修改。运行完整实验时，仍需先获取作者官方代码：

<https://github.com/zeroQiaoba/MERTools/tree/master/MERBench>

然后将本目录中的文件覆盖到官方代码对应位置。修改包括：

- `config.py`、`main-release.py`: 训练参数、类别加权和复现实验入口适配。
- `toolkit/data/feat_data.py`: 模态缺失消融支持。
- `toolkit/utils/loss.py`: 类别加权交叉熵支持。
- `toolkit/models/gated_attention.py`: 门控残差 Attention 改进模型。
- 其他 `__init__.py` 和 `model-tune.yaml`: 模型注册及配置支持。
