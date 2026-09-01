# -*- coding: utf-8 -*-
"""单进程（num_workers=0）评估已训练模型，补齐 test_report/curves/confusion_matrix。

用法: python _eval_best.py --ckpt outputs/.../best.pth [--data-dir ../dataset_augmented_v2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, evaluate, make_loader, make_transforms, plot_curves  # noqa: E402
from torchvision import datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=r"..\..\dataset_augmented_v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)
    out_dir = Path(args.ckpt).parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()
    print(f"设备: {device} | AMP: {amp} | 类别: {class_names}")

    model, _ = build_model(ckpt["arch"], len(class_names), pretrained=False,
                           width_mult=ckpt.get("width_mult", 1.0))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"模型: {ckpt['arch']} | 最优 epoch {ckpt['epoch']} | val loss {ckpt['best_val_loss']:.4f}")

    data_dir = Path(args.data_dir)
    test_ds = datasets.ImageFolder(data_dir / "test",
                                   transform=make_transforms(img_size, train=False))
    loader = make_loader(test_ds, args.batch_size, 0, train=False)  # workers=0 单进程
    print(f"test 集: {len(test_ds)} 张 (workers=0 单进程评估)")

    criterion = nn.CrossEntropyLoss()
    t_loss, t_acc, t_prec, t_rec, t_f1, t_auc, _, _, cm = evaluate(
        model, loader, criterion, device, amp)
    print(f"[测试集] loss {t_loss:.4f} | acc {t_acc:.4f} | precision {t_prec:.4f} | "
          f"recall {t_rec:.4f} | f1 {t_f1:.4f} | auc {t_auc:.4f}")
    print(f"混淆矩阵: {cm.tolist()}")

    (out_dir / "test_report.txt").write_text(
        f"test_loss: {t_loss:.4f}\ntest_acc: {t_acc:.4f}\n"
        f"test_precision: {t_prec:.4f}\ntest_recall: {t_rec:.4f}\n"
        f"test_f1: {t_f1:.4f}\ntest_auc: {t_auc:.4f}\n", encoding="utf-8")

    # 从 metrics.csv 恢复 history，补齐 curves.png / confusion_matrix.png
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [],
            "val_acc": [], "val_auc": []}
    for line in (out_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[1:]:
        v = line.split(",")
        hist["epoch"].append(int(v[0]))
        hist["train_loss"].append(float(v[1]))
        hist["val_loss"].append(float(v[3]))
        hist["train_acc"].append(float(v[2]))
        hist["val_acc"].append(float(v[5]))
        hist["val_auc"].append(float(v[8]))
    plot_curves(hist, out_dir, cm, class_names)
    print(f"已写出: test_report.txt / curves.png / confusion_matrix.png -> {out_dir}")


if __name__ == "__main__":
    main()
