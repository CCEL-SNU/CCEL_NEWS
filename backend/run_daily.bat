@echo off
REM CCEL Daily News - Windows Task Scheduler Runner
REM Full run: collect, PDFs, summarize, digest on schedule day, save, git push.
REM Flags: --collect  --summarize  --digest  --deploy
REM Task Scheduler: Program cmd.exe  Arguments /c "full\path\run_daily.bat"  Start in backend folder
REM Config: backend\config.yaml - API keys not in git

cd /d "%~dp0"

echo [%date% %time%] Starting CCEL Daily News pipeline...
.\venv\Scripts\python run_daily.py %*
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] Pipeline complete - exit code: %EXIT_CODE%.

exit /b %EXIT_CODE%
