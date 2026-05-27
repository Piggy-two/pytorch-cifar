# PyTorch 图像分类学习项目

这是一个基于开源项目 `kuangliu/pytorch-cifar` 改造的 PyTorch 学习项目，适合用来理解图像分类训练的完整流程。

- 上游项目：https://github.com/kuangliu/pytorch-cifar
- 开源协议：MIT License，见 `LICENSE`
- 本地适配：Windows 友好、默认可离线运行、代码加入教学型注释
- 默认任务：使用 scikit-learn 内置 `Digits` 小型手写数字图像数据集训练 `LeNet`

## 1. 快速运行

在 PowerShell 或 Anaconda Prompt 中运行：

```powershell
cd "D:\Anaconda Prompt\pytorch-cifar"
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --epochs 10
```

这个命令会自动使用 `dlgpu` conda 环境。如果 CUDA 可用，程序会自动使用 GPU。

已经验证过的本地结果：

```text
best test acc: 93.89%
checkpoint: checkpoint\digits_lenet_best.pth
```

## 2. 程序运行时发生了什么

运行 `python main.py --epochs 10` 后，程序会按下面的顺序执行：

1. `parse_args()` 读取命令行参数，例如数据集、模型、学习率、训练轮数。
2. `seed_everything()` 固定随机种子，让小实验更容易复现。
3. 自动选择设备：优先使用 `cuda`，否则使用 `cpu`。
4. `build_dataloaders()` 准备数据集、图像预处理和 batch 加载器。
5. `build_model()` 创建模型，例如默认的 `LeNet`。
6. 创建损失函数 `CrossEntropyLoss`，用于衡量分类错误。
7. 创建优化器，例如默认的 `Adam`。
8. 每个 epoch 调用 `train_one_epoch()` 训练模型。
9. 每个 epoch 调用 `evaluate()` 在测试集上评估准确率。
10. 如果测试准确率刷新最好结果，保存 checkpoint。

最核心的训练动作在 `train_one_epoch()`：

```python
optimizer.zero_grad(set_to_none=True)
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

这五步就是 PyTorch 训练的骨架：清空梯度、前向传播、计算损失、反向传播、更新参数。

## 3. 常用命令

只跑 1 轮，检查环境是否正常：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --epochs 1
```

查看所有参数：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --help
```

使用 ResNet18：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset Digits --model ResNet18 --epochs 5 --lr 0.001
```

使用 SGD 优化器：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --optimizer sgd --lr 0.01 --epochs 20
```

只使用一部分数据快速调试：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --train-subset 500 --test-subset 200 --epochs 3
```

网络稳定后尝试原项目的 CIFAR-10：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset CIFAR10 --model ResNet18 --optimizer sgd --lr 0.1 --epochs 20
```

## 4. 重要参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--dataset` | `Digits` | 选择数据集：`Digits`、`MNIST`、`FashionMNIST`、`CIFAR10` |
| `--model` | `LeNet` | 选择模型：`LeNet`、`ResNet18` 等 |
| `--epochs` | `10` | 训练轮数 |
| `--batch-size` | `128` | 每次训练使用多少张图片 |
| `--lr` | `0.001` | 学习率 |
| `--optimizer` | `adam` | 优化器：`adam` 或 `sgd` |
| `--workers` | Windows 下 `0` | DataLoader 使用的加载进程数 |
| `--amp` | 关闭 | 开启 CUDA mixed precision |
| `--resume` | 关闭 | 从对应 checkpoint 恢复训练 |

## 5. 推荐阅读顺序

1. `PYTORCH_TRAINING_GUIDE.md`
   - 完整解释 PyTorch 图像分类训练原理和流程。
2. `main.py`
   - 训练入口，已经加入教学型注释。
3. `models/lenet.py`
   - 最简单、最适合初学者读懂的 CNN。
4. `models/resnet.py`
   - 更深入理解残差连接和深层 CNN。
5. `utils.py`
   - 进度条与辅助函数。

## 6. 输出文件

训练过程中会生成：

```text
checkpoint/
  digits_lenet_best.pth
```

这是默认完整 Digits 训练的最佳模型权重。如果使用 `--train-subset` 或 `--test-subset` 做快速实验，checkpoint 文件名会自动带上子集大小，避免覆盖默认训练结果。`.gitignore` 已经忽略 `checkpoint/` 和 `data/`，避免把训练产物和数据集误提交。

## 7. 数据集说明

默认 `Digits` 数据集不需要下载，适合优先学习 PyTorch 训练流程。

`CIFAR10`、`MNIST`、`FashionMNIST` 仍然保留在程序里，但需要从网络下载。之前当前环境下载 CIFAR-10/MNIST 时出现过压缩包校验失败，所以默认没有依赖它们。

## 8. 原项目 CIFAR-10 参考精度

下面是上游 `kuangliu/pytorch-cifar` README 中的 CIFAR-10 参考结果，用于了解不同模型在 CIFAR-10 上的大致能力。它们不是本地 `Digits` 默认任务的结果。

| Model | Acc. |
| --- | --- |
| VGG16 | 92.64% |
| ResNet18 | 93.02% |
| ResNet50 | 93.62% |
| ResNet101 | 93.75% |
| RegNetX_200MF | 94.24% |
| RegNetY_400MF | 94.29% |
| MobileNetV2 | 94.43% |
| ResNeXt29(32x4d) | 94.73% |
| ResNeXt29(2x64d) | 94.82% |
| SimpleDLA | 94.89% |
| DenseNet121 | 95.04% |
| PreActResNet18 | 95.11% |
| DPN92 | 95.16% |
| DLA | 95.47% |
