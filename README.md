# PyTorch 图像分类学习项目

这是一个基于开源项目 [`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar) 改造的 PyTorch 图像分类学习项目。项目保留了经典 CIFAR-10 模型结构，同时增加了更适合学习和本地离线运行的入口、注释和数据集支持。

- 上游项目：[`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar)
- 开源协议：MIT License，见 `LICENSE`
- 本地适配：Windows 友好、支持离线数据、支持 CUDA 自动选择
- 已支持数据集：`Digits`、`MNIST`、`FashionMNIST`、`CIFAR10`
- 已支持模型入口：`LeNet`、`ResNet18`、`VGG16`、`DenseNet121`、`MobileNetV2`、`PreActResNet18`、`SimpleDLA`

## 1. 环境与快速运行

在 PowerShell 或 Anaconda Prompt 中运行：

```powershell
cd "D:\Anaconda Prompt\pytorch-cifar"
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --epochs 10
```

默认任务会使用 scikit-learn 内置 `Digits` 小型手写数字图像数据集训练 `LeNet`。这个数据集不需要额外下载，适合先确认环境和训练流程。

已经验证过的默认训练结果：

```text
dataset: Digits
model: LeNet
best test acc: 93.89%
checkpoint: checkpoint\digits_lenet_best.pth
```

程序会自动选择设备。如果 CUDA 可用，会使用 GPU；本地正式训练时已验证设备为：

```text
device: cuda
gpu: NVIDIA GeForce RTX 4060
```

## 2. 正式训练命令

CIFAR-10 使用 `ResNet18 + SGD`：

```powershell
cd "D:\Anaconda Prompt\pytorch-cifar"
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset CIFAR10 --model ResNet18 --optimizer sgd --lr 0.1 --epochs 20 --batch-size 128 --workers 0 --no-download
```

MNIST 使用 `LeNet + Adam`：

```powershell
cd "D:\Anaconda Prompt\pytorch-cifar"
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset MNIST --model LeNet --optimizer adam --lr 0.001 --epochs 10 --batch-size 128 --workers 0 --no-download
```

FashionMNIST 使用 `LeNet + Adam`：

```powershell
cd "D:\Anaconda Prompt\pytorch-cifar"
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset FashionMNIST --model LeNet --optimizer adam --lr 0.001 --epochs 10 --batch-size 128 --workers 0 --no-download
```

`--no-download` 表示使用本地已经准备好的数据集，不在训练时联网下载。

## 3. 已验证训练结果

以下结果在 2026-05-28 本地 CUDA 环境中完成训练。

| 数据集 | 模型 | 优化器 | 训练轮数 | best test acc | checkpoint |
| --- | --- | --- | ---: | ---: | --- |
| Digits | LeNet | Adam | 10 | 93.89% | `checkpoint\digits_lenet_best.pth` |
| CIFAR10 | ResNet18 | SGD, lr=0.1 | 20 | 92.21% | `checkpoint\cifar10_resnet18_best.pth` |
| MNIST | LeNet | Adam, lr=0.001 | 10 | 99.01% | `checkpoint\mnist_lenet_best.pth` |
| FashionMNIST | LeNet | Adam, lr=0.001 | 10 | 87.04% | `checkpoint\fashionmnist_lenet_best.pth` |

说明：`checkpoint/` 和 `data/` 已经在 `.gitignore` 中忽略，避免把模型权重和数据集直接提交到 GitHub。GitHub 仓库中保留代码、文档和可复现命令；如需公开权重文件，建议后续使用 GitHub Releases 或 Git LFS。

## 4. 程序运行时发生了什么

运行训练命令后，程序会按下面的顺序执行：

1. `parse_args()` 读取命令行参数，例如数据集、模型、学习率、训练轮数。
2. `seed_everything()` 固定随机种子，让小实验更容易复现。
3. 自动选择设备：优先使用 `cuda`，否则使用 `cpu`。
4. `build_dataloaders()` 准备数据集、图像预处理和 batch 加载器。
5. `build_model()` 创建模型，例如 `LeNet` 或 `ResNet18`。
6. 创建损失函数 `CrossEntropyLoss`，用于衡量分类错误。
7. 创建优化器，例如 `Adam` 或 `SGD`。
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

## 5. 常用命令

只跑 1 轮，检查环境是否正常：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --epochs 1
```

查看所有参数：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --help
```

使用 `ResNet18` 训练 `Digits`：

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

## 6. 重要参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--dataset` | `Digits` | 选择数据集：`Digits`、`MNIST`、`FashionMNIST`、`CIFAR10` |
| `--model` | `LeNet` | 选择模型：`LeNet`、`ResNet18` 等 |
| `--epochs` | `10` | 训练轮数 |
| `--batch-size` | `128` | 每次训练使用多少张图片 |
| `--lr` | `0.001` | 学习率 |
| `--optimizer` | `adam` | 优化器：`adam` 或 `sgd` |
| `--workers` | Windows 下建议 `0` | DataLoader 使用的加载进程数 |
| `--data-dir` | `./data` | 数据集目录 |
| `--checkpoint-dir` | `./checkpoint` | 权重保存目录 |
| `--train-subset` | 关闭 | 限制训练样本数，用于快速调试 |
| `--test-subset` | 关闭 | 限制测试样本数，用于快速调试 |
| `--amp` | 关闭 | 开启 CUDA mixed precision |
| `--resume` | 关闭 | 从对应 checkpoint 恢复训练 |
| `--no-download` | 关闭 | 不下载数据，只读取本地数据集 |

## 7. 数据集说明

本地数据默认放在 `data/` 下。当前已经验证通过的离线数据目录是：

```text
data/
  cifar-10-batches-py/
    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    test_batch
    batches.meta
  MNIST/raw/
    train-images-idx3-ubyte 或 train-images.idx3-ubyte
    train-labels-idx1-ubyte 或 train-labels.idx1-ubyte
    t10k-images-idx3-ubyte 或 t10k-images.idx3-ubyte
    t10k-labels-idx1-ubyte 或 t10k-labels.idx1-ubyte
  FashionMNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
```

如果 CIFAR-10 解压后多出一层 `cifar-10-python/`，程序也兼容：

```text
data/cifar-10-python/cifar-10-batches-py/
```

2026-05-28 已经验证过的离线小子集命令：

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset CIFAR10 --model LeNet --epochs 1 --train-subset 256 --test-subset 128 --batch-size 64 --workers 0 --no-download
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset MNIST --model LeNet --epochs 1 --train-subset 256 --test-subset 128 --batch-size 64 --workers 0 --no-download
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset FashionMNIST --model LeNet --epochs 1 --train-subset 256 --test-subset 128 --batch-size 64 --workers 0 --no-download
```

这些小子集命令只用于确认数据读取、训练、评估和 checkpoint 保存流程都能跑通，不代表正式训练精度。

| 数据集 | 训练子集 | 测试子集 | best test acc | checkpoint |
| --- | ---: | ---: | ---: | --- |
| CIFAR10 | 256 | 128 | 6.25% | `checkpoint\cifar10_lenet_train256_test128_best.pth` |
| MNIST | 256 | 128 | 12.50% | `checkpoint\mnist_lenet_train256_test128_best.pth` |
| FashionMNIST | 256 | 128 | 33.59% | `checkpoint\fashionmnist_lenet_train256_test128_best.pth` |

## 8. 推荐阅读顺序

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

## 9. 原项目 CIFAR-10 参考精度

下面是上游 `kuangliu/pytorch-cifar` README 中的 CIFAR-10 参考结果，用于了解不同模型在 CIFAR-10 上的大致能力。它们不是本地训练结果。

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
