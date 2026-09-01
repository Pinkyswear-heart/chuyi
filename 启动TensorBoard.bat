@echo off
rem ============================================================
rem  TensorBoard Visualization Launcher
rem  Converts metrics.csv to tfevents (if needed), starts
rem  TensorBoard on port 6006 and opens the browser.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title TensorBoard

set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Checking Python...
"%PY%" -V >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo [2/3] Checking TensorBoard...
"%PY%" -c "import tensorboard" >nul 2>nul
if errorlevel 1 (
    echo Installing TensorBoard via Tsinghua mirror...
    "%PY%" -m pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo Install failed. Check network and retry.
        pause
        exit /b 1
    )
)

echo [3/3] Converting metrics to tfevents...
"%PY%" -u tools\tb_logs.py

echo Starting TensorBoard: http://127.0.0.1:6006
echo LAN devices: http://192.168.31.74:6006  (check your IP in console output)
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:6006"
"%PY%" -m tensorboard.main --logdir "%~dp0tb_logs" --host 0.0.0.0 --port 6006
echo.
echo TensorBoard stopped.
pause
