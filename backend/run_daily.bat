@echo off
REM ============================================================
REM CCEL Daily News - Windows Task Scheduler Runner
REM ============================================================
REM Schedule this file in Windows Task Scheduler:
REM   1. Open Task Scheduler (taskschd.msc)
REM   2. Create Basic Task > "CCEL Daily News"
REM   3. Trigger: Daily at 07:00
REM   4. Action: Start a program
REM      Program: cmd.exe
REM      Arguments: /c "C:\path\to\ccel-daily-news\backend\run_daily.bat"
REM      Start in: C:\path\to\ccel-daily-news\backend
REM ============================================================
REM API keys are managed in config.yaml (gemini_api_key, openalex_api_key)
REM ============================================================

REM Navigate to backend directory
cd /d "%~dp0"

REM Activate virtual environment if you have one
REM call venv\Scripts\activate.bat

REM Run the daily pipeline
echo [%date% %time%] Starting CCEL Daily News pipeline...
python run_daily.py %*

echo [%date% %time%] Pipeline complete (exit code: %ERRORLEVEL%).

REM Keep window open if running manually (remove for scheduled task)
REM pause
