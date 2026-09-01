# -*- coding: utf-8 -*-
"""
CNN 图像分类 - 推理验证界面（Gradio + ONNX，无需 PyTorch）

功能:
  1) 单图推理：上传图片 -> 预测类别 + 各类概率
  2) 准确率验证：选择 test/val 数据目录 -> 批量推理，输出
     accuracy / precision / recall / F1 / 混淆矩阵 / 错误样本图集

运行:
  pip install gradio onnxruntime numpy pillow matplotlib
  python app.py
  浏览器打开 http://127.0.0.1:7860
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 注意: onnxruntime 改为惰性导入（load_session 内），避免启动时卡顿

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE.parent / "models"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CONF_THRESHOLD = 0.60  # 低于该置信度判定为"未知/识别失败"（非 lty/miku 的图）
OOD_L2_THRESHOLD = 0.75   # 特征 L2 距离（AMP 特征统计 val p99，误拒≈1%）超过则判未知
OOD_ENERGY_THRESHOLD = 0.60  # logsumexp(logits) 能量低于该值判未知
_ood_cache = {}  # OOD 相关会话/统计缓存（基于增强版模型）


def list_models(folder=None):
    """列出可推理的分类 ONNX 模型（排除 *_features 特征提取器）。"""
    folder = Path(folder or MODEL_DIR)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.onnx") if "_features" not in p.stem)


def load_session(path):
    import onnxruntime as ort  # 惰性导入，避免启动卡顿
    path = str(path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess


def preprocess(img: Image.Image) -> np.ndarray:
    """与训练一致的预处理: Resize(255) -> CenterCrop(224) -> 归一化。"""
    img = img.convert("RGB")
    w, h = img.size
    scale = (IMG_SIZE * 1.14) / min(w, h)
    if abs(scale - 1.0) > 1e-6:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (img.size[0] - IMG_SIZE) // 2
    top = (img.size[1] - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))  # CHW


def predict_batch(sess, images, class_names, batch_size=8):
    """images: list[PIL] -> probs [N, C]；batch_size 默认 8（CPU 友好）。"""
    probs = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        x = np.stack([preprocess(im) for im in chunk]).astype(np.float32)
        logits = sess.run(None, {"input": x})[0]
        e = np.exp(logits.astype(np.float64))
        probs.append(e / e.sum(axis=1, keepdims=True))
    return np.concatenate(probs) if probs else np.zeros((0, len(class_names)))


def _cm_image(cm, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


_OOD_PAIRS = {
    # 模型名关键字 -> (分类器, 特征提取器, 特征统计)
    # 历史模型（best_cnn_augmented_v2 / best_cnn_amp / best_cnn_augmented 及其特征版）
    # 均已删除，当前唯一模型 chuyi.onnx 无配套特征模型，OOD 三信号拒识未启用，
    # 仅使用置信度阈值（CONF_THRESHOLD）。如后续导出特征模型可在此恢复条目。
}


def _load_ood(model_name):
    """按模型名惰性加载对应 OOD 统计（分类器/特征会话 + 均值）。无匹配返回 None。"""
    key = next((k for k in _OOD_PAIRS if k in model_name), None)
    if key is None:
        return None
    if key not in _ood_cache:
        import numpy as _np
        cls_name, feat_name, stats_name = _OOD_PAIRS[key]
        cls_p = BASE.parent / "models" / cls_name
        feat_p = BASE.parent / "models" / feat_name
        stats_p = BASE.parent / stats_name
        if not (cls_p.exists() and feat_p.exists() and stats_p.exists()):
            print(f"[debug] OOD 文件缺失（{cls_name}/{stats_name}），该模型仅置信度规则",
                  flush=True)
            _ood_cache[key] = None
            return None
        stats = _np.load(stats_p)
        _ood_cache[key] = {
            "mean": stats["mean"],
            "cls_sess": load_session(str(cls_p)),
            "feat_sess": load_session(str(feat_p)),
        }
        print(f"[debug] OOD 检测就绪（{key} 特征统计 + 双会话）", flush=True)
    return _ood_cache[key]


def ood_reject(img, model_name="chuyi.onnx"):
    """三信号拒识：能量 / 置信度 / 特征L2距离。返回 (是否拒绝, 原因)。"""
    import numpy as _np
    ood = _load_ood(model_name)
    if ood is None:  # 该模型无 OOD 统计：仅置信度规则
        x = preprocess(img)[None].astype(_np.float32)
        logits = load_session(str(BASE.parent / "models" /
                                  Path(model_name).name)).run(None, {"input": x})[0][0]
        m = logits.max()
        prob = _np.exp(logits - m)
        prob = prob / prob.sum()
        if float(prob.max()) < CONF_THRESHOLD:
            return True, f"置信度过低({float(prob.max()):.3f})"
        return False, ""

    x = preprocess(img)[None].astype(_np.float32)
    logits = ood["cls_sess"].run(None, {"input": x})[0][0]

    m = logits.max()
    energy = float(m + _np.log(_np.exp(logits - m).sum()))
    if energy < OOD_ENERGY_THRESHOLD:
        return True, f"能量异常({energy:.2f})"

    prob = _np.exp(logits - m)
    prob = prob / prob.sum()
    if float(prob.max()) < CONF_THRESHOLD:
        return True, f"置信度过低({float(prob.max()):.3f})"

    feat = ood["feat_sess"].run(None, {"input": x})[0][0].astype(_np.float64)
    feat = feat / (_np.linalg.norm(feat) + 1e-9)
    dist = min(float(_np.linalg.norm(feat - ood["mean"][c]))
               for c in range(len(ood["mean"])))
    if dist > OOD_L2_THRESHOLD:
        return True, f"特征距离过远({dist:.2f})"
    return False, ""


def predict_files(paths, model_path):
    """任意多张图片（拖入即用）: 返回 (摘要, 图集, 表格行)。坏文件自动跳过。"""
    print(f"[debug] predict_files: {len(paths)} 张, model={model_path}", flush=True)
    sess = load_session(model_path)
    print("[debug] session 就绪", flush=True)
    model_name = Path(model_path).name
    use_ood = _load_ood(model_name) is not None  # 该模型是否有对应 OOD 统计
    class_names = ["lty", "miku"]  # 训练固定顺序
    rows, gallery, bad = [], [], 0
    for i, p in enumerate(paths):
        p = Path(p)
        try:
            im = Image.open(p).convert("RGB")
            prob = predict_batch(sess, [im], class_names)[0]
        except Exception as e:
            bad += 1
            rows.append([p.name, f"无法解析({type(e).__name__})", "", ""])
            continue
        idx = int(prob.argmax())
        conf = float(prob.max())
        rejected, reason = (ood_reject(im, model_name) if use_ood
                            else (conf < CONF_THRESHOLD,
                                  f"置信度过低({conf:.3f})"))
        label = "未知(识别失败)" if rejected else class_names[idx]
        rows.append([p.name, label,
                     f"{float(prob[0]):.4f}", f"{float(prob[1]):.4f}"])
        t = im.copy()
        t.thumbnail((200, 200))
        cap = f"{p.name} → {label} ({conf:.3f})" + (f" [{reason}]" if rejected else "")
        gallery.append((t, cap))
        if (i + 1) % 10 == 0:
            print(f"[debug] 已处理 {i+1}/{len(paths)}", flush=True)
    print("[debug] predict_files 完成", flush=True)

    n_lty = sum(1 for r in rows if r[1] == "lty")
    n_miku = sum(1 for r in rows if r[1] == "miku")
    n_unk = sum(1 for r in rows if r[1].startswith("未知"))
    summary = f"共拖入 **{len(paths)}** 张图片"
    if len(rows) > bad:
        summary += f"，预测结果: **lty {n_lty} 张 / miku {n_miku} 张**"
    if n_unk:
        summary += f"；**{n_unk} 张判定识别失败（未知）**（详见表格/图集标注）"
    if bad:
        summary += f"；**{bad} 张无法解析**（已在表格中标注）"
    return summary, gallery, rows


def evaluate_folder(folder, model_path, max_wrong=8, batch_size=8, progress=None):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"目录不存在: {folder}")
    class_names = [d.name for d in sorted(folder.iterdir())
                   if d.is_dir() and list(d.glob("*.*"))]
    if not class_names:
        raise ValueError("请选择包含类别子目录的数据集目录（如 dataset/test）")

    items = [(p, i) for i, c in enumerate(class_names)
             for p in sorted((folder / c).iterdir())
             if p.suffix.lower() in IMG_EXTS]
    if progress:
        progress(0.05, desc=f"加载 {len(items)} 张图片...")
    imgs = [Image.open(p).convert("RGB") for p, _ in items]
    sess = load_session(model_path)
    probs = predict_batch(sess, imgs, class_names, batch_size=batch_size)
    preds = probs.argmax(axis=1)
    true = np.array([i for _, i in items])

    n = len(items)
    acc = float((preds == true).mean())
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(true, preds):
        cm[t, p] += 1
    detail = {}
    for i, c in enumerate(class_names):
        tp = cm[i, i]
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        detail[c] = {"precision": prec, "recall": rec, "f1": f1,
                     "support": int(cm[i, :].sum())}

    wrong = []
    for (p, t), pr, cnf in zip(items, preds, probs.max(axis=1)):
        if pr != t:
            wrong.append((p, class_names[t], class_names[int(pr)], float(cnf)))
        if len(wrong) >= max_wrong:
            break

    gallery = []
    for p, t, pr, cnf in wrong:
        im = Image.open(p).convert("RGB")
        im.thumbnail((256, 256))
        gallery.append((im, f"真: {t} | 预测: {pr} | 置信度 {cnf:.3f}"))

    summary = (
        f"## 验证结果（{folder.name}，共 {n} 张）\n"
        f"- **Accuracy: {acc:.4f}**\n"
        f"- 混淆矩阵列: “预测”类别，行: “真实”类别\n"
        + "\n".join(f"- **{c}**: precision {d['precision']:.3f} | "
                    f"recall {d['recall']:.3f} | f1 {d['f1']:.3f} "
                    f"(support {d['support']})" for c, d in detail.items())
    )
    return summary, _cm_image(cm, class_names), gallery, acc


def build_ui():
    import gradio as gr

    models = list_models()
    if not models:
        raise SystemExit(f"未找到 ONNX 模型: {MODEL_DIR}（请先运行 export_onnx.py）")
    model_names = {m.stem: str(m) for m in models}

    with gr.Blocks(title="CNN 图像分类 - 推理验证") as demo:
        gr.Markdown("# CNN 图像分类界面（洛天依 lty vs 初音未来 miku）\n"
                    "模型: `chuyi.onnx`（自定义CNN，当前唯一推理模型）")

        with gr.Tab("拖入图片推理"):
            gr.Markdown("**把任意数量、任意格式的图片直接拖进来（可多选），"
                        "点“开始推理”** 无需关心路径、尺寸、格式；坏文件会自动跳过并标注。")
            files_in = gr.Files(file_types=["image"], type="filepath",
                                label="拖入图片（支持多张 / 混合格式）")
            with gr.Row():
                btn1 = gr.Button("开始推理", variant="primary")
                model_dd1 = gr.Dropdown(list(model_names.keys()),
                                        value=list(model_names.keys())[0], label="模型")
            summary_out = gr.Markdown()
            gallery_out = gr.Gallery(label="预测结果")
            table_out = gr.Dataframe(headers=["文件", "预测类别", "P(lty)", "P(miku)"],
                                     datatype=["str", "str", "str", "str"],
                                     interactive=False, label="详情表格")

            def wrapped_pred(paths, model):
                try:
                    if not paths:
                        return "**请先拖入图片**", [], []
                    return predict_files(paths, model_names[model])
                except Exception as e:
                    return (f"**错误**: {e}", [],
                            [["", f"{type(e).__name__}: {e}", "", ""]])

            btn1.click(wrapped_pred, inputs=[files_in, model_dd1],
                       outputs=[summary_out, gallery_out, table_out])

        with gr.Tab("准确率验证"):
            with gr.Row():
                folder_in = gr.Textbox(
                    value=str(BASE.parent.parent / "dataset" / "test"), label="数据目录（test/val）")
                model_dd2 = gr.Dropdown(list(model_names.keys()),
                                        value=list(model_names.keys())[0], label="模型")
            btn2 = gr.Button("运行验证", variant="primary")

            def wrapped_eval(folder, model):
                try:
                    def prog(x, desc=""):
                        pass
                    return evaluate_folder(folder, model_names[model], progress=prog)
                except Exception as e:
                    return f"**错误**: {e}", None, [], 0.0

            btn2.click(wrapped_eval, inputs=[folder_in, model_dd2],
                       outputs=[gr.Markdown(), gr.Image(label="混淆矩阵"),
                                gr.Gallery(label="错误样本"), gr.Number(label="Accuracy")])
        gr.Markdown("提示: 数据目录需为 ImageFolder 结构（子文件夹=类别名），如 dataset/test。")
    return demo


if __name__ == "__main__":
    import socket

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

    demo = build_ui()
    demo.queue()
    host = os.environ.get("HOST", "0.0.0.0")          # 默认监听全部网卡（局域网可访问）
    share = os.environ.get("GRADIO_SHARE", "0") == "1"  # 1=生成公网临时链接
    port = int(os.environ.get("PORT", 7860))
    for p in [port, port + 1, port + 2]:
        try:
            demo.launch(server_name=host, server_port=p, share=share,
                        inbrowser=False, show_error=True)
            break
        except OSError as e:
            print(f"端口 {p} 被占用({e})，尝试 {p + 1} ...")
    else:
        raise SystemExit("无法绑定端口，请关闭占用 7860-7862 的程序后重试")
    print("\n===== 访问地址 =====")
    print(f"  本机访问:   http://127.0.0.1:{p}")
    for ip in lan_ips():
        print(f"  局域网访问: http://{ip}:{p}")
    if share:
        print("  公网链接:   见上方 gradio.live 地址（临时、过期失效）")
    print("  提示: 局域网设备打不开时，请以管理员身份放行防火墙：")
    print("    netsh advfirewall firewall add rule name=\"CNN-Gradio-7860\" "
          "dir=in action=allow protocol=TCP localport=7860")
    print("===================\n")
