@echo off
rem ============================================================
rem  CNN Inference UI - PUBLIC SHARE edition
rem  Same as the normal launcher, but also creates a temporary
rem  public https://xxxx.gradio.live link (internet access).
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title CNN Inference UI (Public)

set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Checking Python...
"%PY%" -V >nul 2>nul
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies...
"%PY%" -c "import gradio, onnxruntime, numpy, PIL, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo First run: installing dependencies via Tsinghua mirror, about 1-3 min...
    "%PY%" -m pip install gradio onnxruntime numpy pillow matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo Dependency install failed. Check network and retry.
        pause
        exit /b 1
    )
)

echo [3/3] Starting UI with public share link...
echo LAN:   http://127.0.0.1:7860
echo Public: see the gradio.live URL below (temporary).
set GRADIO_SHARE=1
set HOST=0.0.0.0
"%PY%" -u inference\app.py
echo.
echo UI stopped.
pause
