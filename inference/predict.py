# -*- coding: utf-8 -*-
"""
推理脚本：加载 best.pth，对单张图片或文件夹批量分类，输出 CSV。

用法：
  python predict.py --checkpoint outputs/.../best.pth --input test.jpg
  python predict.py --checkpoint outputs/.../best.pth --input /path/to/images --output result.csv

--input 为文件夹时遍历其中所有图片（不递归子文件夹）；输出 CSV 列：
  file, 预测类别, 各类别概率
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/train")
from train import build_model, make_transforms  # 复用模型定义与数据变换  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="best.pth / last.pth 路径")
    ap.add_argument("--input", required=True, help="图片路径或文件夹")
    ap.add_argument("--output", default=None, help="结果 CSV 路径（默认 input 旁边）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names") or list(ckpt.get("class_to_idx", {}).keys())
    num_classes = len(class_names)

    model, _ = build_model(ckpt["arch"], num_classes, pretrained=False,
                           width_mult=ckpt.get("width_mult", 1.0))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"模型 {ckpt['arch']} | 类别 {class_names} | img_size {ckpt.get('img_size', 224)}")

    img_size = ckpt.get("img_size", 224)
    tf = make_transforms(img_size, train=False)

    input_path = Path(args.input)
    files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not files:
        sys.exit(f"[错误] {input_path} 下没有图片文件")

    print(f"待推理 {len(files)} 张")
    rows, batch = [], []
    for i, f in enumerate(files, 1):
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"[跳过] {f.name}: {e}")
            continue
        batch.append((f, tf(img)))
        if len(batch) == args.batch_size or i == len(files):
            xs = torch.stack([b[1] for b in batch]).to(device)
            probs = torch.softmax(model(xs), dim=1).cpu().numpy()
            for (f2, _), p in zip(batch, probs):
                pred = int(p.argmax())
                row = [str(f2), class_names[pred]] + [f"{v:.6f}" for v in p]
                rows.append(row)
                print(f"  {f2.name}: {class_names[pred]}"
                      f" ({', '.join(f'{n}={v:.4f}' for n, v in zip(class_names, p))})")
            batch = []

    out = Path(args.output) if args.output else input_path.parent / "predictions.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "predicted"] + [f"prob_{n}" for n in class_names])
        w.writerows(rows)
    print(f"结果已保存: {out.resolve()}")


if __name__ == "__main__":
    main()
