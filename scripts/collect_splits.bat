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
python scripts\fetch_nfl_splits.py >> logs\splits.log 2>&1
python scripts\fetch_cfb_splits.py >> logs\splits.log 2>&1
