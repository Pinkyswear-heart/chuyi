# -*- coding: utf-8 -*-
"""导出特征提取版 ONNX（输出分类层前的 512 维特征），用于 OOD 拒识。

用法: python export_features_onnx.py --checkpoint best.pth --output best_features.onnx
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = len(ckpt["class_names"])
    img_size = ckpt.get("img_size", 224)

    model, _ = build_model(ckpt["arch"], num_classes, pretrained=False,
                           width_mult=ckpt.get("width_mult", 1.0))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class Feat(nn.Module):
        """去掉分类头的特征提取器（fc -> Identity）"""

        def __init__(self, m):
            super().__init__()
            m.fc = nn.Identity()
            self.m = m

        def forward(self, x):
            return self.m(x)

    feat = Feat(model)
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(feat, dummy, args.output,
                      input_names=["input"], output_names=["features"],
                      dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=args.opset)
    print(f"特征 ONNX 已导出: {args.output} (输出 features [N,512])")


if __name__ == "__main__":
    main()
