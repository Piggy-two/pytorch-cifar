"""Train image classification models with PyTorch.

This is a Windows-friendly learning entrypoint adapted from the original
kuangliu/pytorch-cifar training script.

The file is intentionally written as a complete training pipeline:
arguments -> data -> model -> loss/optimizer -> train -> evaluate -> checkpoint.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from models import *
from utils import progress_bar


# The original project exposes many model constructors through models/__init__.py.
# A dictionary keeps command-line names explicit and avoids unsafe string eval.
MODEL_BUILDERS = {
    "LeNet": LeNet,
    "ResNet18": ResNet18,
    "PreActResNet18": PreActResNet18,
    "MobileNetV2": MobileNetV2,
    "SimpleDLA": SimpleDLA,
    "DenseNet121": DenseNet121,
    "VGG16": lambda: VGG("VGG16"),
}


def parse_args() -> argparse.Namespace:
    """Read command-line options and return them as a structured object."""
    parser = argparse.ArgumentParser(description="PyTorch image classification training")
    parser.add_argument(
        "--dataset",
        default="Digits",
        choices=["CIFAR10", "Digits", "FashionMNIST", "MNIST"],
        help="image dataset to train on",
    )
    parser.add_argument("--model", default="LeNet", choices=sorted(MODEL_BUILDERS))
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train")
    parser.add_argument("--batch-size", default=128, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], help="optimizer")
    parser.add_argument("--workers", default=0 if os.name == "nt" else 2, type=int)
    parser.add_argument("--data-dir", default="./data", help="dataset directory")
    parser.add_argument("--checkpoint-dir", default="./checkpoint", help="checkpoint directory")
    parser.add_argument("--train-subset", default=None, type=int, help="limit training samples")
    parser.add_argument("--test-subset", default=None, type=int, help="limit test samples")
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--amp", action="store_true", help="use CUDA mixed precision")
    parser.add_argument("--no-download", action="store_true", help="do not download the dataset")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Make random choices repeatable enough for small experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, limit: int | None, seed: int):
    """Optionally keep only part of a dataset for fast smoke tests."""
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


class SklearnDigits(Dataset):
    """Small local image dataset from sklearn.datasets.load_digits.

    PyTorch only requires a dataset to implement __len__ and __getitem__.
    This wrapper turns scikit-learn arrays into tensors so they can flow through
    the same DataLoader/training loop as CIFAR-10 or MNIST.
    """

    def __init__(self, train: bool, transform, seed: int):
        from sklearn.datasets import load_digits

        digits = load_digits()
        # digits.images shape: [N, 8, 8]. Add a channel dimension so each sample
        # becomes [1, 8, 8], then scale pixel values from 0..16 into 0..1.
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
        targets = torch.tensor(digits.target, dtype=torch.long)

        # Build a deterministic 80/20 split. A fixed seed makes the tutorial
        # results stable when you rerun the command.
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(images), generator=generator)
        split = int(0.8 * len(indices))
        selected = indices[:split] if train else indices[split:]

        self.images = images[selected]
        self.targets = targets[selected]
        self.transform = transform

    def __len__(self):
        # DataLoader calls this to know how many samples exist.
        return len(self.targets)

    def __getitem__(self, index):
        # DataLoader calls this repeatedly to fetch one (image, label) pair.
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def build_dataloaders(args: argparse.Namespace):
    """Create training and test DataLoaders.

    Dataset stores samples. Transform converts/normalizes samples. DataLoader
    batches samples and optionally shuffles them for stochastic gradient descent.
    """
    print(f"==> Preparing {args.dataset} data..")
    if args.dataset == "CIFAR10":
        dataset_cls = torchvision.datasets.CIFAR10
        # Training transforms may include random augmentation so the model sees
        # slightly different images each epoch and learns more robust features.
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        # Test transforms must be deterministic; otherwise evaluation would
        # measure both model quality and random preprocessing noise.
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif args.dataset == "FashionMNIST":
        dataset_cls = torchvision.datasets.FashionMNIST
        # FashionMNIST and MNIST are grayscale 28x28. Resize and repeat channels
        # so they match the CIFAR-style models expecting [3, 32, 32] images.
        transform_train = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
            ]
        )
    elif args.dataset == "MNIST":
        dataset_cls = torchvision.datasets.MNIST
        # Digits is already a tensor dataset in this local wrapper, so there is
        # no ToTensor step here. Resize accepts tensors directly in torchvision.
        transform_train = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            ]
        )
    else:
        dataset_cls = SklearnDigits
        transform_train = transforms.Compose(
            [
                transforms.Resize(32, antialias=True),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(32, antialias=True),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    download = not args.no_download
    if args.dataset == "Digits":
        trainset = dataset_cls(train=True, transform=transform_train, seed=args.seed)
        testset = dataset_cls(train=False, transform=transform_test, seed=args.seed)
    else:
        trainset = dataset_cls(
            root=args.data_dir, train=True, download=download, transform=transform_train
        )
        testset = dataset_cls(
            root=args.data_dir, train=False, download=download, transform=transform_test
        )

    trainset = maybe_subset(trainset, args.train_subset, args.seed)
    testset = maybe_subset(testset, args.test_subset, args.seed + 1)

    # pin_memory can speed up CPU -> GPU copies when CUDA is used.
    pin_memory = torch.cuda.is_available()
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    print(f"train samples: {len(trainset)} | test samples: {len(testset)}")
    return trainloader, testloader


def build_model(model_name: str, device: torch.device) -> nn.Module:
    """Instantiate the selected neural network and move it to CPU or GPU."""
    print(f"==> Building model: {model_name}")
    model = MODEL_BUILDERS[model_name]().to(device)
    if device.type == "cuda":
        cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    return model


def checkpoint_path(args: argparse.Namespace) -> Path:
    """Use dataset and model names so checkpoints from experiments do not clash."""
    name = f"{args.dataset.lower()}_{args.model.lower()}"
    if args.train_subset is not None or args.test_subset is not None:
        train_tag = args.train_subset if args.train_subset is not None else "all"
        test_tag = args.test_subset if args.test_subset is not None else "all"
        name = f"{name}_train{train_tag}_test{test_tag}"
    return Path(args.checkpoint_dir) / f"{name}_best.pth"


def load_checkpoint(model, args: argparse.Namespace, device: torch.device):
    """Load saved model weights when --resume is passed."""
    path = checkpoint_path(args)
    if not args.resume:
        return 0, 0.0
    print(f"==> Resuming from checkpoint: {path}")
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["net"])
    return checkpoint["epoch"] + 1, checkpoint["acc"]


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, use_amp: bool):
    """Run one full pass over the training set and update model parameters."""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move both input images and integer class labels to the same device as
        # the model. CUDA tensors and CPU tensors cannot be mixed in operations.
        inputs, targets = inputs.to(device), targets.to(device)

        # Gradients accumulate by default in PyTorch, so every optimization step
        # starts by clearing the previous batch's gradients.
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Forward pass: model converts images into raw class scores [B, 10].
            outputs = model(inputs)

            # CrossEntropyLoss combines LogSoftmax + NLLLoss. Targets are class
            # indices, not one-hot vectors.
            loss = criterion(outputs, targets)

        # Backward pass computes gradients, optimizer.step applies them, and
        # GradScaler keeps the same code path usable with or without --amp.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accuracy is computed from argmax class predictions for logging only.
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    return train_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Measure loss and accuracy without updating model parameters."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    return test_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def save_checkpoint(model, acc: float, epoch: int, args: argparse.Namespace) -> None:
    """Save the best model weights plus a little metadata for resuming."""
    path = checkpoint_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": model.state_dict(), "acc": acc, "epoch": epoch}, path)
    print(f"Saved best checkpoint to {path} (acc={acc:.2f}%)")


def main() -> None:
    """Wire the whole training program together."""
    args = parse_args()
    seed_everything(args.seed)

    # Prefer GPU automatically when PyTorch can see CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    trainloader, testloader = build_dataloaders(args)
    model = build_model(args.model, device)

    criterion = nn.CrossEntropyLoss()

    # Adam is friendly for tiny tutorial datasets. SGD is kept because it is the
    # classic choice in the original CIFAR training recipe.
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # The scheduler changes the learning rate after each epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch, best_acc = load_checkpoint(model, args, device)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch: {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, scaler, device, epoch, use_amp
        )
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        print(
            "summary | train loss: %.3f | train acc: %.2f%% | test loss: %.3f | test acc: %.2f%%"
            % (train_loss, train_acc, test_loss, test_acc)
        )
        if test_acc > best_acc:
            # Save only when the validation/test accuracy improves.
            best_acc = test_acc
            save_checkpoint(model, best_acc, epoch, args)

    print(f"done | best test acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
