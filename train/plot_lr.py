# -*- coding: utf-8 -*-
"""绘制训练学习率曲线（余弦退火），数据来自各轮 metrics.csv 的 lr 列。

用法: python plot_lr.py
输出: lr_curve.png（两模型并排，对数坐标）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent  # cnn_classifier/
RUNS = {
    "custom CNN baseline (54 ep)": BASE / "results_cnn" / "metrics.csv",
    "custom CNN augmented (41 ep)": BASE / "results_cnn_augmented" / "metrics.csv",
}


def load_lr(csv_path):
    rows = []
    best_epoch = None
    with open(csv_path, encoding="utf-8") as f:
        header = next(f).strip().split(",")
        i_lr = header.index("lr")
        i_ep = header.index("epoch")
        i_best = header.index("best") if "best" in header else None
        for line in f:
            cols = line.strip().split(",")
            if len(cols) > i_lr and cols[i_lr]:
                rows.append((int(float(cols[i_ep])), float(cols[i_lr])))
                if i_best is not None and len(cols) > i_best and cols[i_best].strip() == "best":
                    best_epoch = int(float(cols[i_ep]))
    return np.array(rows), best_epoch


def _draw(ax, data, best_epoch, title):
    ax.plot(data[:, 0], data[:, 1], "o-", color="tab:blue", markersize=4,
            label="learning rate")
    if best_epoch:
        y_at = data[data[:, 0] == best_epoch, 1]
        if len(y_at):
            ax.axvline(best_epoch, color="tab:red", ls="--", lw=1)
            ax.annotate(f"best epoch {best_epoch}\nlr={y_at[0]:.2e}",
                        xy=(best_epoch, y_at[0]), xytext=(8, 8),
                        textcoords="offset points", color="tab:red",
                        fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log)")
    ax.set_title(f"{title}\nlr: {data[0,1]:.1e} -> {data[-1,1]:.1e}")
    ax.grid(alpha=0.3, which="both")


def main():
    loaded = {}
    for name, path in RUNS.items():
        if path.exists():
            data, best_ep = load_lr(path)
            loaded[name] = (data, best_ep)
            print(f"{name}: {len(data)} epochs, lr {data[0,1]:.2e} -> {data[-1,1]:.2e}"
                  f"{' | best epoch ' + str(best_ep) if best_ep else ''}")

    # 每模型单独一张
    for name, (data, best_ep) in loaded.items():
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        _draw(ax, data, best_ep, name)
        fig.tight_layout()
        short = name.split(" (")[0]
        fig.savefig(BASE / f"lr_curve_{short.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    # 合并对比图
    fig, axes = plt.subplots(1, len(loaded), figsize=(6.8 * len(loaded), 4.4),
                             squeeze=False)
    for ax, (name, (data, best_ep)) in zip(axes[0], loaded.items()):
        _draw(ax, data, best_ep, name)
    fig.suptitle("Learning Rate Schedule (Cosine Annealing)")
    fig.tight_layout()
    out = BASE / "lr_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存: {out.resolve()} 及每模型单独曲线")


if __name__ == "__main__":
    main()
