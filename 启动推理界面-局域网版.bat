@echo off
rem ============================================================
rem  CNN 图像分类 - 局域网版推理界面
rem  同一 WiFi/局域网内的手机、其他电脑可通过
rem  http://<本机IP>:7860 访问（需先放行防火墙）
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
title CNN Inference UI (LAN)

set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"

echo [1/3] Checking Python...
"%PY%" -V >nul 2>nul
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies (flask / onnxruntime / numpy / pillow / matplotlib)...
"%PY%" -c "import flask, onnxruntime, numpy, PIL, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo First run: installing dependencies via Tsinghua mirror, about 1-3 min...
    "%PY%" -m pip install flask onnxruntime numpy pillow matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo Dependency install failed. Check network and retry.
        pause
        exit /b 1
    )
)

echo [3/3] Opening firewall port 7860 (needs admin, may fail silently)...
netsh advfirewall firewall add rule name="CNN-Web-7860" dir=in action=allow protocol=TCP localport=7860 >nul 2>nul
if errorlevel 1 (
    echo   Firewall rule not added (need admin).
    echo   LAN devices cannot connect until you allow port 7860:
    echo     Double-click  "放行防火墙-局域网版.bat"  (runs as admin)
    echo     or run manually as admin:
    echo       netsh advfirewall firewall add rule name="CNN-Web-7860" ^
dir=in action=allow protocol=TCP localport=7860
    echo.
)

echo.
echo ============================================================
echo   CNN 推理界面 - 局域网版
echo   本机访问:   http://127.0.0.1:7860
echo   局域网访问（同一 WiFi 的电脑/手机浏览器打开）:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=*" %%b in ("%%a") do echo     http://%%b:7860
)
echo.
echo   提示: 防火墙未放行时其他设备打不开页面，请先运行
echo         "放行防火墙-局域网版.bat"（会自动请求管理员权限）
echo ============================================================
echo.

echo Starting server... Keep this window open. Close it to stop.
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:7860"
"%PY%" -u inference\server.py
echo.
echo UI stopped.
pause
