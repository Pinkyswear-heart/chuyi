# -*- coding: utf-8 -*-
"""
CNN 图像分类训练脚本（云端 / 本地通用，洛天依 vs 初音未来 二分类，也可用于任意 ImageFolder 数据集）

一份脚本，两种环境都能直接训练（无需任何改动）：

  本地 Windows / Mac / Linux:
    python train.py                                   # 自动探测 dataset_augmented_v2/ 增强集
    python train.py --arch resnet18 --pretrained --epochs 40
    python train.py --smoke                            # 冒烟测试(每集 32 张 × 2 epochs)

  AutoDL / 云服务器(Linux + GPU):
    python train.py --data-dir /root/autodl-tmp/dataset --epochs 60 --batch-size 64
    python train.py --data-dir /root/autodl-tmp/dataset --arch resnet18 --pretrained --epochs 40

环境自动适配(无需手动判断):
  - 平台: Windows 下 --workers 默认 0(DataLoader 多进程最稳), Linux 下默认 8(云端建议 8~16)
  - 数据目录: 未指定 --data-dir 时自动探测, 顺序: 项目根 dataset_augmented_v2(增强集, 训练实际使用)
    → dataset(高清原图) → /root/autodl-tmp/dataset_augmented_v2 → /root/autodl-tmp/dataset
  - 设备: 有 CUDA 自动用 GPU, 无 GPU 自动 CPU(可加 --amp 开启 CPU bfloat16)
  - 路径: 所有输出路径基于脚本位置解析, 不依赖当前工作目录; 结果输出到 项目根/outputs/<时间戳>_<arch>/
  - 自检: 未安装 torch/torchvision 时给出对应平台的安装命令

功能:
  - 自定义 CNN(ResNet 风格残差网络)或 torchvision 预训练模型(--arch resnet18 等)
  - 数据增强、加权采样 / 加权损失(类别不平衡时自动生效)、标签平滑
  - AMP 混合精度、余弦退火学习率、早停、断点续训(--resume)、NaN 防护
  - TensorBoard 埋点(可选)、训练/验证曲线、混淆矩阵、AUC/精确率/召回率/F1 指标

详细说明见同目录 README.md
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent          # cnn_classifier/train/
PROJ = BASE.parent.parent                        # 项目根(含 dataset/ 等)

# ---------------------------------------------------------------- 环境自检

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from torchvision import datasets, models, transforms
    from tqdm import tqdm
except ImportError as e:
    if sys.platform.startswith("win"):
        hint = (
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  纯 CPU 版: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    else:
        hint = (
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  (AutoDL PyTorch 镜像已预装, 直接 pip install -r requirements.txt 即可)"
        )
    sys.exit(f"[错误] 缺少依赖: {e}\n请先安装 PyTorch:\n{hint}\n再安装其余依赖: pip install -r requirements.txt")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # TensorBoard 可视化(可选; 未安装 tensorboard 时自动跳过)
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

# ---------------------------------------------------------------- 模型定义


class BasicBlock(nn.Module):
    """残差基本块(3x3 conv + BN + ReLU + 3x3 conv + BN, 可选 1x1 shortcut)"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        return F.relu(out + identity, inplace=True)


class CNN(nn.Module):
    """自定义卷积网络: stem + 4 个残差阶段(2/2/2/2), 参数量约 11M(width_mult=1, 224px)"""

    def __init__(self, num_classes=2, width_mult=1.0, dropout=0.3):
        super().__init__()
        w = lambda c: int(round(c * width_mult))

        self.stem = nn.Sequential(
            nn.Conv2d(3, w(64), 7, 2, 3, bias=False),
            nn.BatchNorm2d(w(64)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self._make_layer(w(64), w(64), 2, 1)
        self.layer2 = self._make_layer(w(64), w(128), 2, 2)
        self.layer3 = self._make_layer(w(128), w(256), 2, 2)
        self.layer4 = self._make_layer(w(256), w(512), 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(w(512), num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


def build_model(arch, num_classes, pretrained=False, width_mult=1.0):
    """按 --arch 构建模型。arch == 'cnn' 使用自定义网络, 否则用 torchvision 模型。"""
    if arch != "cnn":
        if not hasattr(models, arch):
            raise ValueError(f"未知模型 {arch!r}, 可用: cnn, " + ", ".join(
                ["resnet18", "resnet34", "resnet50", "mobilenet_v3_small",
                 "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny"]))
        weights = None
        if pretrained:
            wcls = getattr(models, arch + "_Weights", None)
            if wcls is not None:
                weights = getattr(wcls, "IMAGENET1K_V1",
                                  wcls.DEFAULT if hasattr(wcls, "DEFAULT") else None)
        try:
            model = models.__dict__[arch](weights=weights)
        except TypeError:  # 旧版本 torchvision
            model = models.__dict__[arch](pretrained=pretrained)
        if arch.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch.startswith("convnext"):
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        else:  # mobilenet / efficientnet
            n_in = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(n_in, num_classes)
        return model, pretrained

    model = CNN(num_classes=num_classes, width_mult=width_mult)
    return model, None


# ---------------------------------------------------------------- 工具函数


def set_seed(seed=42, benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if benchmark:  # 充分利用 GPU: 允许 cuDNN 自动选择最快卷积算法
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def roc_auc(y_true, y_score):
    """用秩和(Mann-Whitney U)实现 AUC, 避免依赖 sklearn。y_score 为类别 1 的概率。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_true) + 1)
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def all_targets(dataset):
    """取 ImageFolder 或 Subset(ImageFolder) 的标签数组(与数据集长度一致)。"""
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    return np.asarray(dataset.dataset.targets)[dataset.indices]


def make_transforms(img_size, train=True):
    """训练 / 评估数据变换。ImageNet 均值方差, 兼容预训练模型。"""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def make_loader(dataset, batch_size, workers, train=False):
    pf = 2 if workers > 0 else None  # prefetch 过高会放大 worker 崩溃/内存压力
    if train:
        targets = all_targets(dataset)
        counts = np.bincount(targets, minlength=int(targets.max()) + 1)
        sample_weights = 1.0 / np.maximum(counts[targets].astype(np.float64), 1e-9)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=workers, pin_memory=True, drop_last=True,
                          persistent_workers=workers > 0, prefetch_factor=pf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=pf)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, amp_dtype=None):
    """在 loader 上评估: 返回 (loss, acc, precision, recall, f1, auc, 标签, 概率)。"""
    model.eval()
    total_loss = 0.0
    all_targets_l, all_probs = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)
        all_targets_l.append(y.cpu().numpy())
        all_probs.append(probs[:, 1].float().cpu().numpy())  # 类别 1 = miku(bf16 需转 float32)

    y = np.concatenate(all_targets_l)
    p1 = np.concatenate(all_probs)
    pred = (p1 >= 0.5).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    auc = roc_auc(y, p1)

    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return (total_loss / max(len(y), 1), acc, precision, recall, f1, auc, y, p1, cm)


def plot_curves(history, out_dir, cm, class_names):
    """保存训练曲线与混淆矩阵 PNG。"""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "acc", "Accuracy"),
        (axes[2], "auc", "Val AUC"),
    ]:
        if key == "loss":
            ax.plot(epochs, history["train_loss"], "o-", label="train")
            ax.plot(epochs, history["val_loss"], "s-", label="val")
        elif key == "acc":
            ax.plot(epochs, history["train_acc"], "o-", label="train")
            ax.plot(epochs, history["val_acc"], "s-", label="val")
        else:
            ax.plot(epochs, history["val_auc"], "s-", label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel(key); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 参数

def _default_workers():
    """按平台给默认 DataLoader worker 数: Windows 0(最稳), Linux 8(云端建议 8~16)。"""
    return 0 if sys.platform.startswith("win") else 8


def find_data_dir():
    """自动探测数据目录: 本地增强集 v2 优先, 回退原始 dataset/, 再尝试 AutoDL 常见挂载点。"""
    candidates = [
        PROJ / "dataset_augmented_v2",   # 本地: 增强集(训练实际使用)
        PROJ / "dataset",                # 本地: 高清原图
        PROJ / "data",
        Path("/root/autodl-tmp/dataset_augmented_v2"),  # AutoDL 数据盘
        Path("/root/autodl-tmp/dataset"),
    ]
    for d in candidates:
        if (d / "train").is_dir() and (d / "val").is_dir():
            return d
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="CNN 图像分类训练(云端/本地通用; 自动探测数据目录与环境)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="ImageFolder 数据根目录(含 train/val[/test]); 默认自动探测"
                        "(增强集 v2 → dataset → /root/autodl-tmp/...)")
    p.add_argument("--arch", default="cnn",
                   help="cnn=自定义残差网络; 或 resnet18/resnet34/resnet50/"
                        "mobilenet_v3_small/mobilenet_v3_large/efficientnet_b0/convnext_tiny")
    p.add_argument("--pretrained", action="store_true", help="预训练权重(仅 torchvision 模型)")
    p.add_argument("--width-mult", type=float, default=1.0, help="自定义 CNN 通道缩放(0.5/1.0/1.5)")
    p.add_argument("--epochs", type=int, default=60, help="最大训练轮数")
    p.add_argument("--batch-size", type=int, default=32, help="本地建议 32; 云端显存足可 64")
    p.add_argument("--lr", type=float, default=None, help="默认 1e-3(自定义 CNN)/ 3e-4(预训练)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=None,
                   help="DataLoader 进程数; 默认自动(Windows 0 / Linux 8)")
    p.add_argument("--num-workers", dest="workers", type=int, default=None,
                   help="--workers 的别名(兼容旧云端脚本), 命令行中后出现的参数生效")
    p.add_argument("--patience", type=int, default=12, help="验证损失连续 N 轮不改善则早停")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(PROJ / "outputs"),
                   help="结果输出根目录(默认 项目根/outputs, 按时间戳建子目录)")
    p.add_argument("--resume", type=str, default=None, help="从 last.pth 断点续训")
    p.add_argument("--weighted-loss", action=argparse.BooleanOptionalAction, default=True,
                   help="按类别频率加权交叉熵(默认开)")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="标签平滑系数(0=关闭, 如 0.1)")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="混合精度(默认: GPU 自动开; CPU 关, 可 --amp 开 bf16)")
    p.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True,
                   help="cuDNN benchmark 加速(默认开; 关闭可保证完全可复现)")
    p.add_argument("--limit", type=int, default=0, help="每集仅用前 N 个样本做冒烟测试(0=全部)")
    p.add_argument("--smoke", action="store_true", help="快捷冒烟: 等价 --limit 32 --epochs 2")
    return p.parse_args()


# ---------------------------------------------------------------- 主流程


def main():
    args = parse_args()
    if args.smoke:
        args.limit, args.epochs = 32, min(args.epochs, 2)
    if args.workers is None:
        args.workers = _default_workers()
    set_seed(args.seed, args.benchmark)

    # ---- 设备 / 环境自检 ---------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    amp = use_cuda if args.amp is None else args.amp
    amp_dtype = torch.bfloat16 if amp else None  # CPU/GPU 统一 bf16(比 fp16 更稳)
    print(f"平台: {sys.platform} | PyTorch {torch.__version__} | 项目根: {PROJ}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)} | 显存 "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    else:
        print("[提示] 未检测到 CUDA, 使用 CPU 训练(速度较慢, 可考虑减少 epoch 或加 --amp)")
    print(f"设备: {device} | AMP: {amp}"
          + (f" ({amp_dtype})" if amp_dtype else "") + f" | workers: {args.workers}")

    # ---- 数据 ----------------------------------------------------------
    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    if data_dir is None or not ((data_dir / "train").is_dir() and (data_dir / "val").is_dir()):
        sys.exit(f"[错误] 找不到有效数据目录(train/ + val/)。请用 --data-dir 指定, "
                 f"或在项目根放置 dataset_augmented_v2/ 或 dataset/")
    print(f"数据目录: {data_dir.resolve()}")

    train_ds = datasets.ImageFolder(data_dir / "train",
                                    transform=make_transforms(args.img_size, train=True))
    val_ds = datasets.ImageFolder(data_dir / "val",
                                  transform=make_transforms(args.img_size, train=False))
    test_ds = (datasets.ImageFolder(data_dir / "test",
                                    transform=make_transforms(args.img_size, train=False))
               if (data_dir / "test").is_dir() else None)

    class_names = train_ds.classes
    class_to_idx = dict(train_ds.class_to_idx)
    num_classes = len(class_names)
    print(f"类别: {class_to_idx} | "
          f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds) if test_ds else 0}")
    if num_classes != 2:
        print("[警告] 指标逻辑针对二分类(F1/AUC 为 1-vs-1), 多分类时 AUC 不适用")

    def cap(ds, name):
        if args.limit > 0:
            g = torch.Generator().manual_seed(args.seed)
            idx = torch.randperm(len(ds), generator=g)[: args.limit].tolist()
            ds = Subset(ds, idx)
            print(f"[{name}] 冒烟模式 limit={args.limit} -> {len(ds)} 张")
        return ds

    train_ds, val_ds = cap(train_ds, "train"), cap(val_ds, "val")
    if test_ds is not None:
        test_ds = cap(test_ds, "test")

    train_loader = make_loader(train_ds, args.batch_size, args.workers, train=True)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, train=False)

    # ---- 模型 ----------------------------------------------------------
    pretrained = args.pretrained and args.arch != "cnn"
    model, _ = build_model(args.arch, num_classes, pretrained=pretrained,
                           width_mult=args.width_mult)
    if args.arch == "cnn" and args.pretrained:
        print("[提示] --pretrained 仅对 torchvision 模型生效, 自定义 CNN 忽略")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {args.arch} | 可训练参数: {n_params / 1e6:.2f}M")

    # ---- 损失 ----------------------------------------------------------
    if args.weighted_loss and num_classes > 1:
        targets = all_targets(train_ds)
        counts = np.bincount(targets, minlength=num_classes)
        cls_weights = (len(targets) / (num_classes * np.maximum(counts, 1))).astype(np.float32)
        cls_weights = torch.from_numpy(cls_weights).to(device)
        print(f"损失权重: {dict(zip(class_names, [round(float(w), 3)
                                                  for w in cls_weights.cpu().numpy()]))}")
    else:
        cls_weights = None
    criterion = nn.CrossEntropyLoss(weight=cls_weights,
                                    label_smoothing=max(args.label_smoothing, 0.0))

    # ---- 优化器 / 调度 --------------------------------------------------
    lr = args.lr if args.lr is not None else (3e-4 if pretrained else 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 无需损失缩放

    # ---- 输出目录 ------------------------------------------------------
    out_dir = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.arch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir.resolve()}")
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # ---- TensorBoard(可选) ---------------------------------------------
    tb_writer = None
    if _HAS_TB:
        try:
            tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
            print("TensorBoard 日志: " + str(out_dir / "tensorboard"))
        except Exception as e:
            print(f"[提示] TensorBoard 初始化失败({e}), 跳过")

    # ---- 断点续训 ------------------------------------------------------
    start_epoch, best_val_loss = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch, best_val_loss = ckpt["epoch"] + 1, ckpt["best_val_loss"]
        print(f"续训: 从 epoch {start_epoch} 继续, 历史最佳验证损失 {best_val_loss:.4f}")

    # ---- 训练循环 ------------------------------------------------------
    best_path, last_path = out_dir / "best.pth", out_dir / "last.pth"
    log_path = out_dir / "metrics.csv"
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_f1": [], "val_auc": []}
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss",
                                "val_acc", "val_precision", "val_recall", "val_f1",
                                "val_auc", "lr", "epoch_time_s", "best"])

    bad_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, correct, n = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False,
                   dynamic_ncols=True)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                # NaN 防护: 跳过该步(多为极端增强样本在低精度下的数值尖峰)
                optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(loss="nan(skip)")
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / n:.3f}")
        train_loss, train_acc = running / max(n, 1), correct / max(n, 1)

        v_loss, v_acc, v_prec, v_rec, v_f1, v_auc, _, _, _ = evaluate(
            model, val_loader, criterion, device, amp, amp_dtype)
        scheduler.step()

        is_best = v_loss < best_val_loss
        best_val_loss = min(best_val_loss, v_loss)
        bad_epochs = 0 if is_best else bad_epochs + 1

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "class_to_idx": class_to_idx,
            "class_names": class_names,
            "arch": args.arch,
            "width_mult": args.width_mult,
            "img_size": args.img_size,
        }
        torch.save(ckpt, last_path)
        if is_best:
            torch.save(ckpt, best_path)

        for key, val in [("epoch", epoch), ("train_loss", train_loss),
                         ("train_acc", train_acc), ("val_loss", v_loss),
                         ("val_acc", v_acc), ("val_f1", v_f1), ("val_auc", v_auc)]:
            history[key].append(val)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                    f"{v_loss:.4f}", f"{v_acc:.4f}", f"{v_prec:.4f}",
                                    f"{v_rec:.4f}", f"{v_f1:.4f}", f"{v_auc:.4f}",
                                    f"{scheduler.get_last_lr()[0]:.2e}",
                                    f"{time.time() - t0:.1f}", "best" if is_best else ""])
        if tb_writer is not None:
            lr_now = scheduler.get_last_lr()[0]
            for tag, val in [("train/loss", train_loss), ("train/acc", train_acc),
                             ("val/loss", v_loss), ("val/acc", v_acc),
                             ("val/precision", v_prec), ("val/recall", v_rec),
                             ("val/f1", v_f1), ("val/auc", v_auc), ("lr", lr_now)]:
                tb_writer.add_scalar(tag, val, epoch)
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} auc {v_auc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.1f}s"
              + ("  *best*" if is_best else ""))

        if bad_epochs >= args.patience:
            print(f"[早停] 验证损失 {args.patience} 轮未改善(最佳 {best_val_loss:.4f}), 停止训练")
            break

    # ---- 测试集评估 + 报告 ---------------------------------------------
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"已加载最优模型(epoch {ckpt['epoch']}, val loss {ckpt['best_val_loss']:.4f})")

    if test_ds is not None:
        test_loader = make_loader(test_ds, args.batch_size, args.workers, train=False)
        t_loss, t_acc, t_prec, t_rec, t_f1, t_auc, _, _, cm = evaluate(
            model, test_loader, criterion, device, amp, amp_dtype)
        print(f"[测试集] loss {t_loss:.4f} | acc {t_acc:.4f} | precision {t_prec:.4f} | "
              f"recall {t_rec:.4f} | f1 {t_f1:.4f} | auc {t_auc:.4f}")
        with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
            f.write(f"test_loss: {t_loss:.4f}\ntest_acc: {t_acc:.4f}\n"
                    f"test_precision: {t_prec:.4f}\ntest_recall: {t_rec:.4f}\n"
                    f"test_f1: {t_f1:.4f}\ntest_auc: {t_auc:.4f}\n")
        plot_curves(history, out_dir, cm, class_names)
    else:
        print("[提示] 数据目录无 test/, 跳过测试集评估")
        plot_curves(history, out_dir, np.zeros((2, 2), dtype=int), class_names)

    if tb_writer is not None:
        tb_writer.close()
    print(f"全部完成。结果目录: {out_dir.resolve()}"
          f"(best.pth / last.pth / metrics.csv / curves.png / confusion_matrix.png)")
    print(f"推理: python inference/predict.py --checkpoint {out_dir / 'best.pth'} --input <图片>")


if __name__ == "__main__":
    main()
