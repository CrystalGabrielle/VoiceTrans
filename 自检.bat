@echo off
cd /d "%~dp0"
title VOICE TRANS - 自检
if not exist ".venv\Scripts\python.exe" (
    echo [!] 还没装运行环境，请先双击 "安装环境.bat"
    pause
    exit /b 1
)
echo 变调精度 / 共振峰 / 预设性能 / 设备可打开性 自检
echo.
".venv\Scripts\python.exe" -m voicetrans test
echo.
pause
