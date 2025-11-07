@echo off
setlocal enabledelayedexpansion

set URL=http://127.0.0.1:5050/process
set DATA={\"job_folder\":\"D:/spi_ai/output/01/sfcTemp/2025-11-04/20251104144154\"}

for /l %%i in (1,1,10) do (
  echo [%%i/10] POST %URL%
  curl.exe -s -S -X POST "%URL%" -H "Content-Type: application/json" -d "%DATA%"
  if not "%%i"=="10" timeout /t 1 /nobreak >nul
)

endlocal