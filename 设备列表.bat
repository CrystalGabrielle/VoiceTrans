@echo off
cd /d "%~dp0"
title VOICE TRANS - 设备列表
if not exist ".venv\Scripts\python.exe" (
    echo [!] 还没装运行环境，请先双击 "安装环境.bat"
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m voicetrans list
echo.
pause
