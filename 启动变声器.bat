@echo off
cd /d "%~dp0"
title VOICE TRANS - 实时变声器

if not exist ".venv\Scripts\python.exe" (
    echo [!] 还没装运行环境，先帮你装一下...
    echo.
    call "%~dp0安装环境.bat"
    if not exist ".venv\Scripts\python.exe" exit /b 1
)

echo 正在启动变声器，控制台会在浏览器里打开...
".venv\Scripts\python.exe" -m voicetrans run %*
echo.
echo 程序已退出。
pause
