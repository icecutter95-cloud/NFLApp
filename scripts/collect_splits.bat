@echo off
REM Collect public betting splits for both leagues.
REM
REM Run from a scheduled task on THIS machine, not CI: Action Network returns
REM nothing to a datacenter IP, so the GitHub workflow steps always skip.
REM
REM Appends to logs\splits.log so a silent failure is visible after the fact.

cd /d "%~dp0.."
if not exist logs mkdir logs

echo. >> logs\splits.log
echo ==== %DATE% %TIME% ==== >> logs\splits.log
REM The page shows only Action Network's "current" week, which lags our logger
REM by up to a day around Monday night. Ask for the same weeks the logger
REM freezes -- the current one and the next -- via the API instead.
python scripts\fetch_nfl_splits.py --next-weeks >> logs\splits.log 2>&1
python scripts\fetch_cfb_splits.py >> logs\splits.log 2>&1
