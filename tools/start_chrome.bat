@echo off
title Khoi dong Chrome CDP 9222 - PTIT Sentiment Analysis
chcp 65001 >nul
cd /d "%~dp0\.."

echo ======================================================================
echo   KHOI DONG CHROME DEBUGGING PORT 9222 (PTIT SENTIMENT ANALYSIS)
echo ======================================================================
echo.

set PROFILE_DIR=%CD%\.cache\chrome_crawler_profile
if not exist "%PROFILE_DIR%" mkdir "%PROFILE_DIR%"

echo Dang khoi dong Chrome voi cong Debugging 9222...
echo Profile data tai: %PROFILE_DIR%
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --remote-allow-origins=* --user-data-dir="%PROFILE_DIR%" https://www.facebook.com/groups/2k5ptit

echo.
echo ======================================================================
echo   DA KHOI DONG CHROME THANH CONG TREN CONG 9222!
echo   - Neu Chrome chua dang nhap Facebook, hay dang nhap tren cua so do.
echo   - Giu nguyen cua so Chrome nay (khong tat, khong minimize).
echo   - Ban co the click sang ung dung khac de tiep tuc lam viec.
echo ======================================================================
timeout /t 4 >nul
