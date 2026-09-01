# -*- coding: utf-8 -*-
"""图编辑：把分类 ONNX 改为输出 penultimate 特征（去掉最后 Gemm 层）。

用法: python make_features_onnx.py --input best.onnx --output best_features.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = onnx.load(args.input)
    g = model.graph
    logits_name = g.output[0].name
    gemms = [n for n in g.node if n.op_type == "Gemm" and n.output[0] == logits_name]
    if not gemms:
        raise SystemExit(f"未找到输出 {logits_name} 的 Gemm 节点")
    gemm = gemms[0]
    feat_name = gemm.input[0]
    print(f"找到 Gemm: {gemm.name or '(unnamed)'} | 特征张量: {feat_name}")

    g.node.remove(gemm)
    # 新输出 = 特征张量
    vi = helper.make_tensor_value_info(feat_name, TensorProto.FLOAT,
                                       ["N", 512])
    del g.output[:]
    g.output.append(vi)

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"特征 ONNX 已保存: {args.output}")

    # 冒烟验证
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, 224, 224).astype(np.float32)
    y = sess.run(None, {"input": x})[0]
    print(f"验证: 输入 {x.shape} -> 输出 {y.shape} OK")


if __name__ == "__main__":
    main()
