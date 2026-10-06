@echo off
title [PTIT Sentiment] Live Crawler Monitor (Port 9222)
chcp 65001 >nul
cd /d "%~dp0\.."

echo ======================================================================
echo       CHUONG TRINH CAO DU LIEU FACEBOOK PTIT - LIVE MONITOR
echo ======================================================================
echo.

set PROFILE_DIR=%CD%\.cache\chrome_crawler_profile
if not exist "%PROFILE_DIR%" mkdir "%PROFILE_DIR%"

:: 1. Khoi dong Chrome voi cong debug 9222 va profile chuyen dung
echo [1/2] Dang mo Chrome voi cong Debugging 9222...
echo       Profile duoc luu tai: %PROFILE_DIR%
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --remote-allow-origins=* --user-data-dir="%PROFILE_DIR%" https://www.facebook.com/groups/2k5ptit
timeout /t 3 /nobreak >nul

set TARGET=%1
if "%TARGET%"=="" set TARGET=3000

:: 2. Chay bot thu thap va hien thi truc tiep
echo [2/2] Dang ket noi Chrome va bat dau tien trinh thu thap (Muc tieu: %TARGET% binh luan)...
echo.
echo ======================================================================
echo   TIEN DO DANG CHAY TRUC TIEP (REAL-TIME PROGRESS):
echo   - Neu Chrome mo len chua dang nhap Facebook, ban hay dang nhap tren Chrome do.
echo   - Ban se thay tung bai viet va so binh luan nhay lien tuc ben duoi.
echo   - Ban co the chuyen sang app khac lam viec (KHONG minimize Chrome).
echo ======================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\auto_collect_ptit.py --target %TARGET%
) else (
    python scripts\auto_collect_ptit.py --target %TARGET%
)

echo.
echo ======================================================================
echo   HOAN THANH DOT CAO! Nhan phim bat ky de dong cua so nay...
echo ======================================================================
pause >nul
