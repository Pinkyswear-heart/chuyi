# 模型说明（当前保留模型）

## 文件

| 文件 | 说明 |
| --- | --- |
| `chuyi.onnx` | **当前唯一保留的推理模型**（单文件，44.7MB）：自定义 CNN 分类器，输入 `input` [N,3,224,224] → 输出 `logits` [N,2]（lty=洛天依 / miku=初音未来），opset 14，动态 batch |

> 历史模型（best_cnn / best_cnn_augmented / best_cnn_augmented_v2 / best_cnn_amp / best_cnn_local
> 及其特征版）均已删除，仅保留 `chuyi.onnx`。训练指标见 `../docs/RESULTS.md` 及各结果目录。

## 推理（ONNX，无需 PyTorch）

```bash
python inference/server.py   # 网页界面 http://127.0.0.1:7860（默认使用本模型）
```

```python
import sys
sys.path.insert(0, "inference")
import numpy as np, onnxruntime as ort
from PIL import Image
from app import preprocess, predict_batch
sess = ort.InferenceSession("models/chuyi.onnx", providers=["CPUExecutionProvider"])
img = Image.open("test.jpg").convert("RGB")
prob = predict_batch(sess, [img], ["lty", "miku"])[0]
print(dict(zip(["lty", "miku"], prob)))
```

## 拒识规则

当前模型未启用三信号 OOD 拒识（无配套特征模型），仅使用**置信度阈值**：
`max_prob < 0.60` 判定为「未知（识别失败）」。

若需启用 OOD 拒识：导出配套特征模型（`export_features_onnx.py`）并运行 `python _ood_setup.py --feat-model <特征模型>.onnx`，
然后在 `server.py` 的 `_OOD_MODELS` 中加入模型名。

## 模型管理

- 目录：`cnn_classifier/models/`，界面自动列出其中所有 `*.onnx`（`*_features` 除外）
- 训练脚本：`cnn_classifier/train/train.py`（云端/本地通用，AMP 自动；重新训练导出 ONNX 后模型名与界面列出的 `*.onnx` 对应）
- 对比评估：`python _model_compare.py`、`python _eval_compare.py`
