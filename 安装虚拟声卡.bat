@echo off
cd /d "%~dp0"
echo ============================================
echo    VB-CABLE 虚拟声卡安装
echo ============================================
echo.
echo 这个驱动做的事：把"变声后的声音"变成一根虚拟音频线，
echo 直播软件从线的另一头（CABLE Output）取声音当麦克风用。
echo.
echo 接下来会弹出 UAC 授权窗口，请点"是"；
echo 在安装程序里点 "Install Driver"；装完点确定，然后重启电脑。
echo.
if not exist "tools\VBCABLE\VBCABLE_Setup_x64.exe" (
    echo [x] 没找到安装文件，请手动下载：
    echo     https://vb-audio.com/Cable/
    pause
    exit /b 1
)
pause
powershell -NoProfile -Command "Start-Process -FilePath '%CD%\tools\VBCABLE\VBCABLE_Setup_x64.exe' -Verb RunAs"
echo.
echo 安装程序已启动。装完后重启电脑，再打开"启动变声器.bat"，
echo 在"设备 → 输出去向"里选择 CABLE Input 即可。
pause
