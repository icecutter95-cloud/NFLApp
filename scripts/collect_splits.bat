@echo off
REM Collect public betting splits for both leagues.
REM
REM FALLBACK ONLY since 2026-10-07. Primary collection is the Collect Betting
REM Splits workflow, every 2 hours in CI. The claim that "Action Network
REM returns nothing to a datacenter IP" was measured on the public-betting
REM PAGE and wrongly generalised to the whole module; the scoreboard API
REM answers a GitHub runner with the same numbers a home connection gets.
REM
REM Keep the scheduled task. If Action Network ever blocks GitHub's ranges,
REM this is the way back in with no code change -- and it is harmless running
REM alongside CI, because captured_at is floored to the minute and the upsert
REM key includes it, so a duplicate observation collapses onto one row.
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
