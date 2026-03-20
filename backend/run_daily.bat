@echo off
REM ============================================================
REM CCEL Daily News - Windows Task Scheduler Runner
REM ============================================================
REM
REM Pipeline steps (full run, no flags):
REM   1. Collect paper metadata (RSS + OpenAlex)
REM   2. Download PDFs
REM   3. AI Summarization (Gemini API, skips already-summarized papers)
REM   4. Weekly Digest + Category Trends + Group Digests (on digest day)
REM   5. Save news.json + history snapshot
REM   6. Git push to GitHub
REM
REM Available flags (pass via: run_daily.bat --flag):
REM   --collect    Only collect papers + download PDFs
REM   --summarize  Only summarize (loads existing papers)
REM   --digest     Force generate weekly digest + category/group digests
REM   --deploy     Only push to GitHub
REM
REM Schedule in Windows Task Scheduler:
REM   1. Open Task Scheduler (taskschd.msc)
REM   2. Create Basic Task > "CCEL Daily News"
REM   3. Trigger: Daily at 04:00
REM   4. Action: Start a program
REM      Program: cmd.exe
REM      Arguments: /c "C:\Users\parwg\Documents\개인프로젝트\01_CCEL_NEWS\ccel-daily-news\backend\run_daily.bat"
REM      Start in: C:\Users\parwg\Documents\개인프로젝트\01_CCEL_NEWS\ccel-daily-news\backend
REM
REM Configuration:
REM   - API keys & settings: backend/config.yaml (not tracked by git)
REM   - Gemini API for summarization & digest generation
REM   - OpenAlex API for researcher group tracking
REM   - Weekly digest auto-generates on configured day (config: schedule.weekly_digest_day)
REM
REM ============================================================

REM Navigate to backend directory
cd /d "%~dp0"

REM Run the daily pipeline using venv Python
echo [%date% %time%] Starting CCEL Daily News pipeline...
.\venv\Scripts\python run_daily.py %*
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] Pipeline complete (exit code: %EXIT_CODE%).

exit /b %EXIT_CODE%
