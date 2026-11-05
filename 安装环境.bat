@echo off
cd /d "%~dp0"
title VOICE TRANS - 环境安装
echo ============================================
echo    VOICE TRANS  环境安装
echo ============================================
echo.

set "PY="
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    echo [x] 没有找到 Python。
    echo     请先安装 Python 3.10 或更高版本，安装时务必勾选
    echo     "Add Python to PATH" 这一项。
    echo     下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo     使用解释器: %PY%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境 .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [x] 创建虚拟环境失败。
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在，跳过。
)

echo [2/3] 升级 pip ...
".venv\Scripts\python.exe" -m pip install -q --upgrade pip

echo [3/3] 安装依赖 numpy / scipy / sounddevice ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo     默认源失败，改用清华镜像重试 ...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
if errorlevel 1 (
    echo.
    echo [x] 依赖安装失败，请检查网络后重试。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import numpy, scipy, sounddevice" >nul 2>nul
if errorlevel 1 (
    echo [x] 依赖校验未通过，请重新运行本脚本。
    pause
    exit /b 1
)

echo.
echo 安装完成。双击 "启动变声器.bat" 开始使用。
pause
