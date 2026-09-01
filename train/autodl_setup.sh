#!/usr/bin/env bash
# AutoDL 环境自检 + 依赖安装
set -e
PY=/root/miniconda3/bin/python
echo "== Python =="
$PY -V
echo "== PyTorch =="
$PY -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| cuda', torch.cuda.is_available())"
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== 安装依赖 =="
$PY -m pip install -r "$(dirname "$0")/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "完成"
