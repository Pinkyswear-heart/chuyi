# -*- coding: utf-8 -*-
"""针对 lty（洛天依/luo）的增强版离线数据增强（v2，更强变换池）。

- 源：_resized/dataset（512px JPEG 训练副本，不改原始高清图）
- 变换池（7 种，按文件名+序号做确定性种子，可复现）：
    0) 水平翻转
    1) 随机旋转 ±12°（expand 后中心裁剪回原尺寸）
    2) 强颜色抖动: 亮度 0.6~1.4 / 对比度 0.7~1.3 / 饱和度 0.7~1.3 / 色相 ±0.05
    3) 缩放 0.88~1.12 + 平移 ±6%
    4) 翻转 + 亮度极端（0.55~0.75 或 1.3~1.45）——针对识别测试暴露的亮度短板
    5) 随机裁剪放大（scale 0.7~1.0, aspect 0.9~1.1）
    6) 随机矩形遮挡（面积 8%~20%）
- 目标：train/lty = 原图 3205 + 增强 = 7291（与 miku 1:1）
- val/test/miku 原样复制（不增强，评估口径不变）

用法: python augment_data.py --src <resized> --dst <out> [--target 7291] [--threads 8]
"""
import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

QUALITY = 85
POOL_SIZE = 7


def _crop_to(im, w, h):
    iw, ih = im.size
    left = max(0, (iw - w) // 2)
    top = max(0, (ih - h) // 2)
    return im.crop((left, top, min(left + w, iw), min(top + h, ih)))


def _hue_shift(im, delta):
    """HSV 色相偏移 ±delta（0~1）。"""
    a = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    diff = mx - mn
    h = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = diff > 1e-6
    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = mask & (mx == g)
    h[idx] = (b[idx] - r[idx]) / diff[idx] + 2
    idx = mask & (mx == b)
    h[idx] = (r[idx] - g[idx]) / diff[idx] + 4
    h = (h / 6.0 + delta) % 1.0
    s = np.where(diff > 1e-6, diff / np.maximum(mx, 1e-6), 0)
    v = mx
    hi = (h * 6).astype(int) % 6
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    out = np.empty_like(a)
    for i, (r_, g_, b_) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                      (p, q, v), (t, p, v), (v, p, q)]):
        m2 = hi == i
        out[..., 0][m2], out[..., 1][m2], out[..., 2][m2] = r_[m2], g_[m2], b_[m2]
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def _cutout(im, rng):
    w, h = im.size
    cw = int(w * rng.uniform(0.15, 0.35))
    ch = int(h * rng.uniform(0.15, 0.35))
    x = rng.randint(0, max(1, w - cw))
    y = rng.randint(0, max(1, h - ch))
    im = im.copy()
    d = ImageDraw.Draw(im)
    d.rectangle([x, y, x + cw, y + ch], fill=(128, 128, 128))
    return im


def augment_variant(im, kind, rng):
    """返回同尺寸增强版本。kind: 0..6"""
    w, h = im.size
    if kind == 0:
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    if kind == 1:
        ang = rng.uniform(-12, 12)
        return _crop_to(im.rotate(ang, resample=Image.BICUBIC, expand=True), w, h)
    if kind == 2:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
        im = _hue_shift(im, rng.uniform(-0.05, 0.05))
        return im
    if kind == 3:
        s = rng.uniform(0.88, 1.12)
        cw, ch = max(int(w / s), 8), max(int(h / s), 8)
        cx = w / 2 + rng.uniform(-0.06, 0.06) * w
        cy = h / 2 + rng.uniform(-0.06, 0.06) * h
        box = (int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))
        left, top = max(0, box[0]), max(0, box[1])
        right, bottom = min(w, box[2]), min(h, box[3])
        if right - left < 8 or bottom - top < 8:
            return im
        return im.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)
    if kind == 4:  # 翻转 + 亮度极端
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        f = rng.choice([rng.uniform(0.55, 0.75), rng.uniform(1.3, 1.45)])
        return ImageEnhance.Brightness(im).enhance(f)
    if kind == 5:  # 随机裁剪放大
        scale = rng.uniform(0.7, 1.0)
        aspect = rng.uniform(0.9, 1.1)
        cw = min(max(int(w * scale), 8), w)
        ch = min(max(int(h * scale * aspect), 8), h)
        x = rng.randint(0, max(0, w - cw))
        y = rng.randint(0, max(0, h - ch))
        return im.crop((x, y, x + cw, y + ch)).resize((w, h), Image.LANCZOS)
    # kind == 6: 随机矩形遮挡
    return _cutout(im, rng)


def process_one(args):
    src_file, dst_dir, idx, n_variants = args
    try:
        with Image.open(src_file) as im:
            im.load()
            im = im.convert("RGB")
        stem = src_file.stem
        for k in range(n_variants):
            rng = random.Random(f"{src_file.name}#{k}")
            out = augment_variant(im, (idx + k) % POOL_SIZE, rng)
            dst = dst_dir / f"{stem}__aug{idx}_{k}.jpg"
            out.save(dst, "JPEG", quality=QUALITY, optimize=True)
        return True
    except Exception as e:
        print(f"[跳过] {src_file}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"_resized\dataset")
    ap.add_argument("--dst", default=r"dataset_augmented")
    ap.add_argument("--target", type=int, default=7291)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    lty_src = src / "train" / "lty"
    assert lty_src.is_dir(), f"源目录无效: {src}"

    files = sorted(p for p in lty_src.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
    n0 = len(files)
    need = max(args.target - n0, 0)
    n_per = need // n0
    extra = need % n0
    print(f"lty 源 {n0} -> 目标 {args.target}（变体 {need}，每图 {n_per}+{extra}，"
          f"变换池 {POOL_SIZE} 种）")

    for split in ("train", "val", "test"):
        for cls in ("lty", "miku"):
            s = src / split / cls
            d = dst / split / cls
            if s.is_dir():
                d.mkdir(parents=True, exist_ok=True)
                for f in s.iterdir():
                    if f.is_file() and not (d / f.name).exists():
                        shutil.copy2(f, d / f.name)

    lty_dst = dst / "train" / "lty"
    lty_dst.mkdir(parents=True, exist_ok=True)
    jobs = [(f, lty_dst, i, n_per + (1 if i < extra else 0)) for i, f in enumerate(files)]
    ok = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            ok += fut.result()
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} 完成")
    n_final = len(list(lty_dst.glob("*.jpg")))
    print(f"完成: {ok}/{len(jobs)} | train/lty {n_final} "
          f"(miku {len(list((dst/'train'/'miku').glob('*.*')))})")
    print(f"输出: {dst.resolve()}")


if __name__ == "__main__":
    main()
