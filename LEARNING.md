# PyTorch Learning Notes

This folder contains a local learning version of the open-source
`kuangliu/pytorch-cifar` project.

- Upstream project: https://github.com/kuangliu/pytorch-cifar
- License: MIT, kept in `LICENSE`
- Original focus: CIFAR-10 image classification with many CNN architectures
- Local change: the training entrypoint is Windows-friendly and can run without a
  network download by using the tiny `Digits` image dataset from scikit-learn.

## Quick Start

From `D:\Anaconda Prompt\pytorch-cifar`:

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --epochs 10
```

This default command uses:

- dataset: `Digits`
- model: `LeNet`
- optimizer: `Adam`
- GPU: automatically uses CUDA when available

Verified local result:

```text
best test acc: 93.89%
checkpoint: checkpoint\digits_lenet_best.pth
```

## Useful Commands

Run a very short smoke test:

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --epochs 1
```

Try a deeper model on the small local dataset:

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset Digits --model ResNet18 --epochs 5 --lr 0.001
```

Try the original CIFAR-10 workflow when the network is stable:

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --dataset CIFAR10 --model ResNet18 --optimizer sgd --lr 0.1 --epochs 20
```

Show all options:

```powershell
& "D:\anaconda3\Scripts\conda.exe" run -n dlgpu python main.py --help
```

## What To Read First

Read these files in this order:

1. `PYTORCH_TRAINING_GUIDE.md`
   - A full Chinese explanation of the PyTorch training workflow used here.
2. `main.py`
   - `parse_args`: command-line parameters
   - `build_dataloaders`: dataset, transform, DataLoader
   - `build_model`: model construction and CUDA placement
   - `train_one_epoch`: forward, loss, backward, optimizer step
   - `evaluate`: inference with `torch.no_grad`
   - `save_checkpoint`: saving trained weights
3. `models/lenet.py`
   - A small CNN that is easy to understand end to end.
4. `models/resnet.py`
   - A deeper CNN showing residual blocks and shortcut connections.
5. `utils.py`
   - Progress display and small helper functions.

## Suggested Experiments

Change one thing at a time:

- Increase `--epochs` from `10` to `20`.
- Compare `--optimizer adam` with `--optimizer sgd --lr 0.01`.
- Compare `--model LeNet` with `--model ResNet18`.
- Try `--train-subset 500` to see what happens with less data.
- Turn on mixed precision for larger CUDA runs with `--amp`.

## Notes

The first attempt to download CIFAR-10 and MNIST failed in this environment
because the downloaded archive did not pass checksum verification. The code keeps
those datasets available, but the default `Digits` dataset avoids network
problems so the PyTorch training loop is always runnable.
