@echo off
rem ============================================================
rem  CNN 图像分类 - 本地一键训练
rem  双击启动: 检查 Python/依赖 -> 选择训练模式 -> 开始训练
rem  底层调用: train\train.py (云端/本地通用, 自动适配环境)
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title CNN Local Trainer

rem Prefer Python 3.13 (has deps installed)
set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if exist "%PY%" goto :py_ok
set "PY=python"
:py_ok

echo ============================================================
echo   CNN 图像分类 - 本地一键训练
echo   数据目录: 自动探测 dataset_augmented_v2/ 或 dataset/
echo ============================================================
echo.

rem ---- [1/4] Python ----
echo [1/4] Checking Python...
"%PY%" -V >nul 2>nul
if errorlevel 1 goto :err_py

rem ---- [2/4] 依赖 ----
echo [2/4] Checking dependencies (torch / torchvision / tqdm / matplotlib / numpy)...
"%PY%" -c "import torch, torchvision, tqdm, matplotlib, numpy" >nul 2>nul
if errorlevel 1 goto :deps_missing
goto :data_ok

:deps_missing
echo   Missing dependencies, trying to install tqdm/matplotlib/numpy (Tsinghua mirror)...
"%PY%" -m pip install -q tqdm matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
"%PY%" -c "import torch, torchvision" >nul 2>nul
if errorlevel 1 goto :err_torch
goto :data_ok

:err_py
echo   Python not found. Please install Python 3.10+ and retry.
pause
exit /b 1

:err_torch
echo.
echo   [错误] torch/torchvision 未安装,请手动安装后重试:
echo     "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo   纯 CPU 版(约 200MB):
echo     "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
pause
exit /b 1

rem ---- [3/4] 数据目录提示(不阻塞,train.py 会自行探测) ----
:data_ok
echo [3/4] Checking data directory...
if exist "..\dataset_augmented_v2\train" goto :data_v2
if exist "..\dataset\train" goto :data_v1
echo   [警告] 未在项目根找到 dataset_augmented_v2/ 或 dataset/,
echo   请确认数据目录存在,或用 --data-dir 显式指定。
goto :menu
:data_v2
echo   OK: dataset_augmented_v2/ (增强集, 训练实际使用)
goto :menu
:data_v1
echo   OK: dataset/ (高清原图)

rem ---- [4/4] 训练模式 ----
:menu
echo.
echo [4/4] Select training mode:
echo   1) 冒烟测试      - 每集 32 张 x 2 epochs, 快速验证环境 (约 1 分钟)
echo   2) 默认训练      - 自定义 CNN, 60 epochs, batch 32 (完整训练)
echo   3) 迁移学习      - ResNet18 预训练, 40 epochs, batch 64
echo   4) 自定义参数    - 输入额外参数 (如 --epochs 30 --img-size 192)
echo.
set /p MODE=请选择 (1-4, 默认 2): 
if "%MODE%"=="" set "MODE=2"
set "ARGS="
if "%MODE%"=="1" set "ARGS=--smoke"
if "%MODE%"=="1" goto :run
if "%MODE%"=="3" set "ARGS=--arch resnet18 --pretrained --epochs 40 --batch-size 64"
if "%MODE%"=="3" goto :run
if "%MODE%"=="4" goto :custom
goto :run

:custom
echo   示例: --epochs 30 --img-size 192 --arch mobilenet_v3_small
set /p ARGS=额外参数: 

:run
echo.
echo ============================================================
echo 启动训练: python train\train.py %ARGS% %*
echo ============================================================
"%PY%" -u train\train.py %ARGS% %*

echo.
echo 训练结束。结果在 项目根\outputs\时间戳_模型\ 下
echo (best.pth / metrics.csv / curves.png / confusion_matrix.png / test_report.txt)
pause
