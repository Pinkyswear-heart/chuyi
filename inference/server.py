# -*- coding: utf-8 -*-
"""
CNN 图像分类 - 网页推理服务（Flask + ONNX，纯 HTML/JS 前端）

功能:
  - GET  /api/health    服务状态 + 可用模型列表
  - POST /api/predict   上传多张图片 -> 预测类别 + 概率 + OOD 拒识
  - POST /api/evaluate  输入数据集目录 -> Accuracy / P/R/F1 / 混淆矩阵 / 错误样本
  - 静态页面: webui/ (index.html / style.css / app.js)

运行:
  python server.py
  浏览器打开 http://127.0.0.1:7860
"""
import base64
import io
import os
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory

# 复用 app.py 的预处理 / 批量推理 / OOD 拒识 / 目录评估逻辑（无 gradio 依赖）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import (  # noqa: E402
    CONF_THRESHOLD,
    evaluate_folder,
    list_models,
    load_session,
    ood_reject,
    predict_batch,
    preprocess,
)

BASE = Path(__file__).resolve().parent
WEBUI = BASE / "webui"

# 中文显示名（与训练类别顺序一致）
CLASS_CN = {"lty": "洛天依", "miku": "初音未来"}
CLASS_ORDER = ["lty", "miku"]

# 会话缓存: 按模型路径缓存 onnxruntime session，避免每次请求重新加载
_session_cache = {}
# OOD 会话缓存（基于增强版模型，与 app.py 的 _ood_cache 分开，避免重复加载）
_ood_cache = None

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB，多图上传

_thumbs = {}


def _get_session(model_path):
    key = str(model_path)
    if key not in _session_cache:
        _session_cache[key] = load_session(key)
    return _session_cache[key]


# 启用三信号 OOD 拒识的模型名（需配套 *_features.onnx 与 ood_stats.npz）
# 当前唯一模型 chuyi.onnx 无配套特征模型，OOD 拒识未启用（仅置信度阈值兜底）；
# 后续导出特征模型后可将模型名加入此集合。
_OOD_MODELS = set()


def _load_ood():
    """惰性加载 OOD 拒识所需会话与特征统计；依赖文件缺失时返回 None（OOD 禁用）。"""
    global _ood_cache
    if _ood_cache is None:
        cls_path = BASE.parent / "models" / "chuyi.onnx"
        feat_path = BASE.parent / "models" / "chuyi_features.onnx"
        stats_path = BASE.parent / "ood_stats.npz"
        if not (cls_path.is_file() and feat_path.is_file() and stats_path.is_file()):
            print("[server] OOD 依赖缺失，OOD 拒识禁用（仅置信度阈值）", flush=True)
            _ood_cache = {}
            return _ood_cache
        stats = np.load(stats_path)
        _ood_cache = {
            "mean": stats["mean"],
            "cls": load_session(str(cls_path)),
            "feat": load_session(str(feat_path)),
        }
        print("[server] OOD 拒识就绪（特征统计 + 双会话）", flush=True)
    return _ood_cache


def _ood_reject_shared(img):
    """与界面同管线的 OOD 拒识（三信号：能量 / 置信度 / 特征L2距离）。"""
    ood = _load_ood()
    if not ood:  # OOD 不可用时不拒识（由置信度阈值兜底）
        return False, ""
    x = preprocess(img)[None].astype(np.float32)
    logits = ood["cls"].run(None, {"input": x})[0][0]

    m = logits.max()
    energy = float(m + np.log(np.exp(logits - m).sum()))
    if energy < 0.60:  # OOD_ENERGY_THRESHOLD
        return True, f"能量异常({energy:.2f})"

    prob = np.exp(logits - m)
    prob = prob / prob.sum()
    if float(prob.max()) < CONF_THRESHOLD:
        return True, f"置信度过低({float(prob.max()):.3f})"

    feat = ood["feat"].run(None, {"input": x})[0][0].astype(np.float64)
    feat = feat / (np.linalg.norm(feat) + 1e-9)
    dist = min(float(np.linalg.norm(feat - ood["mean"][c]))
               for c in range(len(ood["mean"])))
    if dist > 0.75:  # OOD_L2_THRESHOLD（AMP 特征统计 val p99，误拒≈1%）
        return True, f"特征距离过远({dist:.2f})"
    return False, ""


def _thumb_b64(im, size=256, quality=82):
    """PIL 图 -> base64 data URL（JPEG 缩略图）。"""
    t = im.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- 页面

@app.get("/")
def index():
    return send_from_directory(WEBUI, "index.html")


# ---------------------------------------------------------------- API

@app.get("/api/health")
def health():
    models = [m.stem for m in list_models()]
    return jsonify({
        "status": "ok",
        "models": models,
        "onnxruntime": _import_ort_version(),
        "conf_threshold": CONF_THRESHOLD,
        "class_names": CLASS_ORDER,
        "class_cn": CLASS_CN,
        "python": sys.version.split()[0],
        "default_folder": str(BASE.parent.parent / "dataset" / "test"),
    })


def _import_ort_version():
    try:
        import onnxruntime as ort
        return ort.__version__
    except Exception:
        return "?"


@app.post("/api/predict")
def predict():
    """多图推理。表单: files=图片列表, model=模型名（不带 .onnx）。"""
    files = request.files.getlist("files")
    model_name = (request.form.get("model") or "").strip()
    if not files:
        return jsonify({"ok": False, "error": "未收到图片文件"}), 400

    model_path = BASE.parent / "models" / f"{model_name}.onnx" if model_name else None
    if model_path is None or not model_path.is_file():
        available = [m.stem for m in list_models()]
        return jsonify({"ok": False, "error": f"模型不存在: {model_name}，可用: {available}"}), 400

    # 三信号 OOD 仅对当前最优 AMP 模型启用（特征统计基于其特征空间）
    use_ood = model_path.stem in _OOD_MODELS
    sess = _get_session(model_path)
    print(f"[server] predict: {len(files)} 张, model={model_path.name}, ood={use_ood}",
          flush=True)

    results = []
    for f in files:
        name = f.filename or "image"
        try:
            im = Image.open(f.stream).convert("RGB")
            prob = predict_batch(sess, [im], CLASS_ORDER)[0]
        except Exception as e:
            results.append({"name": name, "ok": False,
                            "error": f"无法解析({type(e).__name__})"})
            continue

        idx = int(prob.argmax())
        conf = float(prob.max())
        if use_ood:
            rejected, reason = _ood_reject_shared(im)
        else:
            rejected = conf < CONF_THRESHOLD
            reason = f"置信度过低({conf:.3f})" if rejected else ""
        pred = "unknown" if rejected else CLASS_ORDER[idx]
        results.append({
            "name": name,
            "ok": True,
            "pred": pred,
            "pred_cn": "未知(识别失败)" if rejected else CLASS_CN.get(pred, pred),
            "conf": round(conf, 4),
            "probs": {c: round(float(prob[i]), 4) for i, c in enumerate(CLASS_ORDER)},
            "rejected": bool(rejected),
            "reason": reason,
            "size": list(im.size),
        })

    summary = {"total": len(results),
               "lty": sum(1 for r in results if r.get("pred") == "lty"),
               "miku": sum(1 for r in results if r.get("pred") == "miku"),
               "unknown": sum(1 for r in results if r.get("rejected")),
               "failed": sum(1 for r in results if not r.get("ok"))}
    return jsonify({"ok": True, "results": results, "summary": summary,
                    "model": model_path.name, "ood": use_ood})


@app.post("/api/evaluate")
def evaluate():
    """目录评估。表单: folder=数据集目录, model=模型名。"""
    folder = (request.form.get("folder") or "").strip().strip('"')
    model_name = (request.form.get("model") or "").strip()
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return jsonify({"ok": False, "error": f"目录不存在: {folder}"}), 400

    model_path = BASE.parent / "models" / f"{model_name}.onnx" if model_name else None
    if model_path is None or not model_path.is_file():
        return jsonify({"ok": False, "error": f"模型不存在: {model_name}"}), 400

    print(f"[server] evaluate: folder={folder}, model={model_path.name}", flush=True)
    try:
        summary_md, cm_img, gallery, acc = evaluate_folder(
            folder, str(model_path), max_wrong=12)
        # gallery: [(PIL缩略图, "真: X | 预测: Y | 置信度 Z"), ...]
        import re
        wrong = []
        for im, cap in gallery:
            m = re.match(r"真: (\S+) \| 预测: (\S+) \| 置信度 ([\d.]+)", cap)
            if m:
                wrong.append({"img": _thumb_b64(im), "true": m.group(1),
                              "pred": m.group(2), "conf": float(m.group(3))})
        return jsonify({
            "ok": True,
            "folder": str(folder_path),
            "n": _extract_n(summary_md),
            "accuracy": round(float(acc), 4),
            "classes": _extract_classes(summary_md),
            "confusion": _thumb_b64(cm_img, size=560, quality=88),
            "wrong": wrong,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


def _extract_n(summary_md):
    """从 evaluate_folder 的 Markdown 摘要中提取图片总数。"""
    import re
    m = re.search(r"共 (\d+) 张", summary_md)
    return int(m.group(1)) if m else 0


def _extract_classes(summary_md):
    """解析摘要中的每类 precision/recall/f1/support。"""
    import re
    out = {}
    for line in summary_md.splitlines():
        m = re.match(r"- \*\*(\w+)\*\*: precision ([\d.]+) \| recall ([\d.]+) "
                     r"\| f1 ([\d.]+) \(support (\d+)\)", line.strip())
        if m:
            name, p, r, f1, sup = m.groups()
            out[name] = {"precision": float(p), "recall": float(r),
                         "f1": float(f1), "support": int(sup)}
    return out


# ---------------------------------------------------------------- 静态文件（兜底，须在 API 路由之后注册）

@app.get("/<path:name>")
def static_files(name):
    return send_from_directory(WEBUI, name)


# ---------------------------------------------------------------- 启动

def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if any(ip.startswith(p) for p in ("192.168.", "10.", "172.", "100.")):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


if __name__ == "__main__":
    models = [m.stem for m in list_models()]
    if not models:
        print(f"[server] 警告: {BASE.parent / 'models'} 下未找到 ONNX 模型！", flush=True)
    else:
        print(f"[server] 可用模型: {', '.join(models)}", flush=True)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    bound = None
    for p in [port, port + 1, port + 2]:
        try:
            # 先探测端口是否可绑定，再正式启动
            import socket as _s
            probe = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
            probe.bind((host, p))
            probe.close()
            bound = p
            break
        except OSError as e:
            print(f"端口 {p} 被占用({e})，尝试 {p + 1} ...", flush=True)
    if bound is None:
        raise SystemExit("无法绑定端口，请关闭占用 7860-7862 的程序后重试")

    print("\n===== 访问地址 =====")
    print(f"  本机访问:   http://127.0.0.1:{bound}")
    for ip in lan_ips():
        print(f"  局域网访问: http://{ip}:{bound}")
    print("  提示: 局域网设备打不开时，请以管理员身份放行防火墙：")
    print("    netsh advfirewall firewall add rule name=\"CNN-Web-7860\" "
          "dir=in action=allow protocol=TCP localport=7860")
    print("===================\n")
    app.run(host=host, port=bound, threaded=True, debug=False)
