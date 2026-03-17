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

REM Set your Anthropic API key here (or set it in system env variables)
set ANTHROPIC_API_KEY=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

REM Optional: Semantic Scholar API key for higher rate limits
REM set SEMANTIC_SCHOLAR_API_KEY=your-s2-key-here

REM Navigate to backend directory
cd /d "%~dp0"

REM Activate virtual environment if you have one
REM call venv\Scripts\activate.bat

REM Run the daily pipeline
echo [%date% %time%] Starting CCEL Daily News pipeline...
python run_daily.py

echo [%date% %time%] Pipeline complete.

REM Keep window open if running manually (remove for scheduled task)
REM pause
