@echo off
cd /d "%~dp0"
echo ============================================
echo    VB-CABLE 需要自行从官网下载
echo ============================================
echo.
echo 本仓库不附带 VB-CABLE。
echo 它是 VB-Audio 的捐赠软件，版权归 Vincent Burel，
echo 不在本项目的 MIT 许可之内。
echo.
echo 请打开官网下载官方压缩包并解压：
echo     https://vb-audio.com/Cable/
echo 64 位系统：右键 VBCABLE_Setup_x64.exe
echo           选择“以管理员身份运行”
echo 32 位系统：右键 VBCABLE_Setup.exe，同样以管理员运行
echo 在安装窗口点 Install Driver，完成后重启电脑。
echo.
echo 来源 www.vb-cable.com 。VB-CABLE 是捐赠软件。
echo.
echo 正在打开官网...
start "" "https://vb-audio.com/Cable/"
echo.
pause
