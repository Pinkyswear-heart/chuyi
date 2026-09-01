# -*- coding: utf-8 -*-
"""把训练好的 best.pth 导出为 ONNX，并用 onnxruntime 与 PyTorch 输出做数值比对。

用法:
  python export_onnx.py --checkpoint best.pth --output best.onnx
  python export_onnx.py --checkpoint best.pth --output best.onnx --verify-image test.jpg

依赖: 无额外依赖导出（torch）；--verify 需 pip install onnxruntime
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, make_transforms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="best.pth 路径")
    ap.add_argument("--output", required=True, help="输出 .onnx 路径")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--verify-image", default=None,
                    help="可选：用 onnxruntime 对该图片做 PyTorch vs ONNX 输出比对")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    img_size = ckpt.get("img_size", 224)

    model, _ = build_model(ckpt["arch"], num_classes, pretrained=False,
                           width_mult=ckpt.get("width_mult", 1.0))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"ONNX 已导出: {args.output} | 类别 {class_names} | img_size {img_size}")

    if args.verify_image:
        try:
            import numpy as np
            import onnxruntime as ort
            from PIL import Image
        except ImportError:
            print("[提示] 缺 onnxruntime 或 PIL，跳过验证")
            return

        tf = make_transforms(img_size, train=False)
        img = tf(Image.open(args.verify_image).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            torch_out = torch.softmax(model(img), dim=1).numpy()

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": img.numpy()})[0]
        onnx_out = np.exp(onnx_out) / np.exp(onnx_out).sum(axis=1, keepdims=True)

        diff = np.abs(onnx_out - torch_out).max()
        print(f"验证图片: {args.verify_image}")
        print(f"PyTorch 概率: {dict(zip(class_names, torch_out[0].round(4)))}")
        print(f"ONNX    概率: {dict(zip(class_names, onnx_out[0].round(4)))}")
        print(f"最大概率差异: {diff:.2e}  {'OK' if diff < 1e-3 else '差异过大!'}")


if __name__ == "__main__":
    main()
