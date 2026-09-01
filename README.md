# CNN 图像分类 - AutoDL 训练项目

## 目录结构

代码按功能分类到子目录：

| 目录/文件 | 说明 |
| --- | --- |
| `train/` | 训练相关：`train.py`（**云端/本地通用**训练脚本：自定义残差 CNN / torchvision 预训练模型、AMP、加权采样、早停、断点续训、自动探测数据目录与环境）、`train_local.py`（兼容入口，转发到 train.py）、`augment_data.py`（数据增强）、`export_onnx.py` / `export_features_onnx.py` / `make_features_onnx.py`（ONNX 导出）、`plot_lr.py`（学习率曲线）、`_eval_best.py`（单进程评估）、`autodl_setup.sh` |
| `inference/` | 推理相关：`server.py`（Flask Web 服务）、`app.py`（旧 Gradio 界面）、`predict.py`（命令行批量推理 → CSV）、`webui/`（HTML/JS 前端） |
| `tools/` | 工具：`tb_logs.py`（metrics.csv → TensorBoard tfevents） |
| `docs/` | 文档与历史结果：`RESULTS.md`、模型对比/识别率测试报告（txt/png） |
| `models/` | ONNX 模型（`chuyi.onnx`，见下文模型说明） |
| `requirements.txt` | 依赖（torch/torchvision 已由 AutoDL 镜像提供） |
| `*.bat` | Windows 启动脚本（`启动训练.bat` 一键训练 / `启动推理界面-*.bat` 推理界面 / `启动TensorBoard.bat` / 防火墙放行） |

## 数据集

ImageFolder 结构（必须有 `train/` 和 `val/` 子目录，`test/` 可选）：

```
dataset/
├── train/{类别名}/  图片...
├── val/{类别名}/    图片...
└── test/{类别名}/   图片...（可选，训练结束后自动评估）
```

本项目对应数据集：`dataset/`（lty=洛天依, miku=初音未来，train 3205/7291，val 687/1563，test 687/1563）。

## 训练（云端 / 本地通用，一份脚本）

**`train.py` 云端、本地都能直接跑**，自动适配环境：
数据目录自动探测（增强集 v2 → dataset → AutoDL 挂载点）、
`--workers` 按平台自适应（Windows 0 / Linux 8）、有 GPU 自动用 GPU、路径基于脚本位置解析。

**一键方式**（仅本地 Windows）：双击 **`启动训练.bat`** —— 自动检查 Python/依赖 → 选择模式
(1 冒烟测试 / 2 默认训练 / 3 ResNet18 迁移学习 / 4 自定义参数)→ 开始训练。

### 本地（Windows / CPU / GPU）

```bash
cd cnn_classifier
# 默认:自动探测 dataset_augmented_v2/ 增强集,自定义 CNN,60 epochs
python train/train.py

# 预训练 ResNet18 迁移学习
python train/train.py --arch resnet18 --pretrained --epochs 40

# 冒烟测试(每集 32 张 × 2 epochs,快速验证环境)
python train/train.py --smoke
```

特点:自动探测数据目录(增强集 v2 优先)、`--workers` 默认 0(Windows 下最稳)、
无 GPU 自动 CPU、未装 torch 时提示安装命令;结果输出到 `项目根/outputs/<时间戳>_<arch>/`。
首次使用先装 PyTorch:`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`(GPU)
或 `.../whl/cpu`(纯 CPU)。详细说明见 `train/train.py` 文件头。

## AutoDL 使用步骤

### 1. 上传

将本目录上传到实例数据盘，推荐：

```
/root/autodl-tmp/cnn_classifier/    # 代码
/root/autodl-tmp/dataset/           # 数据集
```

（AutoDL 控制台 → 实例 → JupyterLab → 右上角上传；或 `scp -P <端口> -r`。）

### 2. 环境

AutoDL PyTorch 镜像已预装 torch 2.x + CUDA，只需补装少量依赖：

```bash
cd /root/autodl-tmp/cnn_classifier
pip install -r requirements.txt
```

### 3. 训练（同一份 train.py，无需改动）

```bash
# 自定义 CNN（默认，约 11M 参数）；--data-dir 可省略(自动探测 /root/autodl-tmp/dataset)
python train/train.py --data-dir /root/autodl-tmp/dataset \
    --epochs 60 --batch-size 32 --workers 8

# 预训练 ResNet18 迁移学习（精度更高）
python train/train.py --data-dir /root/autodl-tmp/dataset \
    --arch resnet18 --pretrained --epochs 40 --batch-size 64
```

> 兼容旧参数 `--num-workers`（等价 `--workers`），云端旧启动脚本可直接复用。

建议用 tmux/nohup 挂后台，防止 SSH 断开导致训练中断：

```bash
tmux new -s train        # 或 nohup python train/train.py ... > train.log 2>&1 &
python train/train.py --data-dir /root/autodl-tmp/dataset --epochs 60 --batch-size 32
# 查看进度：Ctrl+b d 退出 tmux；tmux attach -t train 恢复
```

### 4. 结果

输出在 `项目根/outputs/<时间戳>_<模型>/`（即 `cnn_classifier/../outputs/`）：

| 文件 | 内容 |
| --- | --- |
| `best.pth` | 验证集最优模型（含类别映射/架构信息） |
| `last.pth` | 最后一个 epoch 权重（配合 `--resume` 断点续训） |
| `metrics.csv` | 每 epoch 指标（loss/acc/precision/recall/F1/AUC/lr） |
| `curves.png` | loss / acc / AUC 曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `test_report.txt` | 测试集最终指标 |
| `args.json` | 训练参数 |

查看/下载：JupyterLab 文件树直接预览 PNG；或 `zip -r outputs.zip outputs` 后下载。

### 5. 推理

```bash
python inference/predict.py --checkpoint ../outputs/<时间戳>_cnn/best.pth --input test.jpg
python inference/predict.py --checkpoint ../outputs/<时间戳>_cnn/best.pth --input /path/images --output result.csv
```

## TensorBoard 可视化

**双击 `启动TensorBoard.bat`**：自动转换 metrics → tfevents、启动 TensorBoard、
打开浏览器 http://127.0.0.1:6006（局域网设备访问 http://192.168.31.74:6006）。

- 手动方式：
  ```bash
  python tools/tb_logs.py                 # 把 metrics.csv 转成 tfevents（无需 PyTorch）
  tensorboard --logdir tb_logs --port 6006 --host 0.0.0.0
  ```
- `train.py` 已内置 TensorBoard 埋点：每次训练自动在 `项目根/outputs/<时间戳>_<模型>/tensorboard/`
  写入日志（需 `pip install tensorboard`），训练完直接 `tensorboard --logdir outputs` 查看

## 推理验证界面（本地 Web UI）

### 网页版（推荐，纯 HTML/JS + Flask，全新界面）

双击 **`启动推理界面-网页版.bat`** 即可：自动检查/安装依赖（清华镜像）、启动服务、
自动打开浏览器 **http://127.0.0.1:7860**。

> 网页版是当前主力界面（替代旧 Gradio 界面），依赖已就绪时启动约 3 秒。

### 局域网版（同一 WiFi 内其他设备访问）

双击 **`启动推理界面-局域网版.bat`**：自动放行防火墙、显示局域网地址并启动服务。
手机/其他电脑连同一 WiFi 后，浏览器打开控制台显示的 `http://<本机IP>:7860` 即可使用。

**首次使用如其他设备打不开**，先双击 **`放行防火墙-局域网版.bat`**（自动请求管理员权限，
仅需执行一次），再启动局域网版：

```
本机:   http://127.0.0.1:7860
局域网: http://<本机IP>:7860   （如 http://192.168.31.74:7860，启动时自动显示）
```

- 手动放行防火墙（管理员 PowerShell/cmd 执行一次）：
  `netsh advfirewall firewall add rule name="CNN-Web-7860" dir=in action=allow protocol=TCP localport=7860`

功能：
- **图片推理**：拖拽/点选多张图片 → 预测类别 + 置信度 + 各类概率条形图
  - 模型：`chuyi.onnx`（**当前唯一推理模型**，自定义 CNN，输入 224×224 → 2 类概率；
    无配套特征模型，OOD 三信号拒识未启用，仅置信度 <0.60 判未知）
  - 汇总统计（洛天依 / 初音未来 / 未知 / 失败）、坏文件自动标注
- **准确率验证**：选择数据集目录（默认 `dataset/test`，ImageFolder 结构）→ 一键输出
  Accuracy、各类 precision/recall/F1、混淆矩阵、错误样本图集

技术栈：`inference/webui/index.html` + `style.css` + `app.js`（原生前端）+ `inference/server.py`（Flask API：
`/api/health`、`/api/predict`、`/api/evaluate`），基于 ONNX 模型，无需 PyTorch。
手动启动：`python inference/server.py`（端口冲突时自动尝试 7860→7862）。

手动启动：`pip install gradio onnxruntime numpy pillow matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple`
后执行 `python inference/app.py`。

## 常见问题

- **`CUDA out of memory`**：减小 `--batch-size`（32→16）、`--img-size`（224→192），或加 `--no-amp`。
- **训练很慢/GPU 跑不满**：增大 `--workers`（8~16），确认 `nvidia-smi` GPU-Util 接近 100%。
- **`python` 找不到**：使用 `/root/miniconda3/bin/python`，或 `source /root/miniconda3/bin/activate` 后再运行。
- **重新开始训练**：删除/更换 `--output-dir` 子目录（每次运行自动生成新时间戳目录，不会覆盖）。
- **结果不稳定**：固定 `--seed`；本数据 lty 样本较少，验证时留意 recall/F1 而非只看 acc。
