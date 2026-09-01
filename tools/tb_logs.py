# -*- coding: utf-8 -*-
"""把训练 metrics.csv 转成 TensorBoard tfevents 日志（无需 PyTorch）。

用法: python tb_logs.py
输出: cnn_classifier/tb_logs/<run>/ 下的 tfevents 文件
启动: tensorboard --logdir cnn_classifier/tb_logs --port 6006
"""
from pathlib import Path

from tensorboard.summary import Writer

BASE = Path(__file__).resolve().parent.parent  # cnn_classifier/
RUNS = {
    "custom_cnn": BASE / "results_cnn" / "metrics.csv",
    "custom_cnn_augmented": BASE / "results_cnn_augmented" / "metrics.csv",
    "custom_cnn_augmented_v2": BASE / "results_cnn_augmented_v2" / "metrics.csv",
}
TAGS = {
    "train_loss": "train/loss",
    "train_acc": "train/acc",
    "val_loss": "val/loss",
    "val_acc": "val/acc",
    "val_precision": "val/precision",
    "val_recall": "val/recall",
    "val_f1": "val/f1",
    "val_auc": "val/auc",
    "lr": "lr",
}


def convert(run_name, csv_path, log_root):
    if not csv_path.exists():
        print(f"[跳过] {csv_path} 不存在")
        return False
    logdir = log_root / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    writer = Writer(str(logdir))
    with open(csv_path, encoding="utf-8") as f:
        header = next(f).strip().split(",")
        cols = {name: header.index(name) for name in TAGS if name in header}
        epoch_idx = header.index("epoch") if "epoch" in header else None
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header) or not parts[0]:
                continue
            step = int(float(parts[epoch_idx])) if epoch_idx is not None else 0
            for tag in TAGS:
                if tag in cols and cols[tag] < len(parts) and parts[cols[tag]]:
                    writer.add_scalar(TAGS[tag], float(parts[cols[tag]]), step)
    writer.close()
    n = len(list(logdir.glob("*.tfevents.*")))
    print(f"{run_name}: 已写入 {logdir} ({n} 个事件文件)")
    return True


def main():
    root = BASE / "tb_logs"
    done = [convert(name, path, root) for name, path in RUNS.items()]
    if not any(done):
        raise SystemExit("没有任何 metrics.csv 可转换")
    print(f"完成。启动 TensorBoard: tensorboard --logdir {root} --port 6006")


if __name__ == "__main__":
    main()
