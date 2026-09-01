@echo off
rem ============================================================
rem  CNN Image Classification - Inference UI Launcher
rem  Double-click to start. Auto-installs deps, opens browser.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title CNN Inference UI

rem Prefer Python 3.13 (has deps installed)
set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Checking Python...
"%PY%" -V >nul 2>nul
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies (gradio / onnxruntime / numpy / pillow / matplotlib)...
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

echo [3/3] Starting UI: http://127.0.0.1:7860
echo Keep this window open while using the UI. Close it to stop.
start "" /min cmd /c "timeout /t 6 >nul & start http://127.0.0.1:7860"
"%PY%" -u inference\app.py
echo.
echo UI stopped.
pause
