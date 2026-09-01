@echo off
rem ============================================================
rem  放行 CNN 推理界面端口 7860（Windows 防火墙）
rem  双击运行，会自动请求管理员权限（UAC 弹窗请点"是"）
rem  仅需执行一次；之后局域网设备即可访问
rem ============================================================
chcp 65001 >nul
title CNN Firewall Open (7860)

echo Requesting admin permission to open port 7860...
powershell -NoProfile -Command "Start-Process -FilePath 'netsh' -ArgumentList 'advfirewall firewall add rule name=\"CNN-Web-7860\" dir=in action=allow protocol=TCP localport=7860' -Verb RunAs -Wait"

echo.
echo Done. If no error popup appeared, port 7860 is now open.
echo Verify on another device: http://<本机IP>:7860
echo (Get the IP from  ipconfig  or the LAN launcher screen.)
pause
