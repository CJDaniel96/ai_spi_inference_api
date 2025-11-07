@echo off
setlocal enabledelayedexpansion

set URL=http://127.0.0.1:5050/process
set DATA={\"job_folder\":\"D:/Dre/JQ_SPI_02_AI_API/data/20250523135357\"}

for /l %%i in (1,1,10) do (
  echo [%%i/10] POST %URL%
  curl.exe -s -S -X POST "%URL%" -H "Content-Type: application/json" -d "%DATA%"
  if not "%%i"=="10" timeout /t 1 /nobreak >nul
)

endlocal