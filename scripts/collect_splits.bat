@echo off
REM Collect public betting splits for both leagues.
REM
REM NOT fallback only, despite what this comment claimed until 2026-11-20.
REM Measured over the 7 days to then: this task supplies 2 of the ~8.4 daily
REM CFB splits captures, and its captures are just as complete as CI's --
REM 99.7 rows per capture against 103.3. The Collect Betting Splits workflow
REM asks for every 2 hours but GitHub delivers about a third of scheduled
REM runs (see .github\workflows\README.md), so this task is a real primary
REM path, not a spare. It is also the ONLY line pull at these two times.
REM
REM The CI move itself was correct. The claim that "Action Network
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

REM CFB lines BEFORE CFB splits, added 2026-11-20. fetch_cfb_odds.py ran in
REM one place while fetch_cfb_splits.py ran in three, so the line went stale
REM against the tickets beside it: on 2026-11-20 the WYOMING/NEW_MEXICO spread
REM read a four-day-old 7.5 while the 16:00 splits carried 9.0 and DraftKings
REM was actually 9.5 -- readings of +3.0, +1.5 and +1.0 against the 10.5
REM opener, on a 1.5-point bar. NFL lines need no pull here: refresh-odds runs
REM hourly on pg_cron and delivers 90%. cfb_line_history is append-only with a
REM surrogate key, so this cannot overwrite an opener. 2 Odds API credits.
python scripts\fetch_cfb_odds.py >> logs\splits.log 2>&1
python scripts\fetch_cfb_splits.py >> logs\splits.log 2>&1
