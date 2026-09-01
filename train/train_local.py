# -*- coding: utf-8 -*-
"""兼容入口: 已统一为 train.py(云端/本地通用), 本文件仅转发, 保持旧命令可用。

旧用法(如 启动训练.bat / 旧文档)仍有效:
  python train_local.py [--smoke | --arch resnet18 --pretrained ...]

所有功能与环境自适应(数据目录探测 / workers / 路径解析)见 train.py。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
