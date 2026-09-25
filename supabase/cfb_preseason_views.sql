-- CFB and preseason DDL, recorded here because these objects were created
-- against the live database and were not in schema.sql. Applied 2026-08-20.
--
-- 1. cfb_api_cache        -- CFBD responses; the 3-hourly job was refetching
--                            seven static endpoints and exhausted the monthly
--                            call quota.
-- 2. preseason_results    -- ESPN scores so the preseason tab can grade itself.
-- 3. cfb_open_close       -- FIXED: one row per game, orientation normalised.
-- 4. cfb_tracking         -- FIXED: joins open/close on game_id, not team pair.
--
-- The cfb_open_close fix is the one worth reading. It had two defects:
--
--   * the window partitioned by commence_time::date while the GROUP BY used the
--     full commence_time, so a kickoff corrected across midnight split one game
--     into two rows, which then fanned out through cfb_tracking's pair join --
--     73 predictions rendering as 75 games.
--
--   * OKLAHOMA @ TEXAS is neutral-site and the feed swapped which team it calls
--     home partway through: 33 snapshots at spread_home -6, then one at +6.
--     spread_home is signed RELATIVE TO HOME, so collapsing by game_id without
--     normalising orientation would have mixed lines of opposite sign and
--     corrupted the opener and closer that CLV is measured against.
--
-- Canonical orientation is the one a game was seen in most often, ties broken by
-- earliest sighting; snapshots recorded the other way round have spread_home
-- negated.
--
-- game_id was called "safe as the key" here on 2026-08-20 because zero
-- matchups carried more than one id. That held until 2026-09-13, when
-- HOUSTON @ TEXAS_TECH moved from Saturday to Friday and the Odds API
-- re-issued it under a new event id. The prediction stayed keyed to the first
-- id, whose history stopped that day at -10, while the real number kept moving
-- under the second id (-7.5 by kickoff) that nothing joined to; the tab also
-- kept showing the old Saturday kickoff. Migration
-- cfb_open_close_follow_reissued_event_ids (2026-09-18) folds every id for the
-- same team pair with kickoffs within 4 days into the first-seen id, which is
-- the one the logger keys on, and cfb_tracking now reports the kickoff from the
-- latest snapshot rather than the frozen prediction. Rematches (title games,
-- bowls) are months apart and stay separate.
--
-- Found the same day while analysing the first three weeks: the hourly odds
-- refresh does not stop at kickoff, so a LIVE line recorded during the game
-- was landing as the "close" -- 98 of 150 graded games had one (MICHIGAN -34.5
-- at kickoff, -15.5 "closing" ninety minutes in). Every CLV and direction
-- figure on the college tab was contaminated, and the first pass of the
-- analysis reported a 71% "follow the move" edge that was nothing but the
-- scoreboard leaking into the closer. Migration
-- cfb_open_close_close_is_pre_kickoff (2026-09-20) takes the close from the
-- last PRE-kickoff snapshot, as line_open_close already did for the NFL.
-- scripts/cfb_signal_check.py is the analysis, rerunnable as the sample grows.

create table if not exists public.cfb_api_cache (
  cache_key   text primary key,
  payload     jsonb       not null,
  fetched_at  timestamptz not null default now()
);

create table if not exists public.preseason_results (
  game_id     text primary key,
  season      int  not null,
  home_team   text not null,
  away_team   text not null,
  home_score  int  not null,
  away_score  int  not null,
  fetched_at  timestamptz not null default now()
);

-- See supabase migrations cfb_open_close_one_row_per_game and
-- cfb_tracking_join_on_game_id for the full view bodies as applied.

-- 5. cfb_public_splits  -- hand-captured bets%/money%, applied 2026-08-31.
-- 6. cfb_signal         -- model lean x public money x line movement.
--
-- There is no automated source for splits. Action Network's are a proprietary
-- product behind Cloudflare and JS rendering, and every free feed checked
-- (ESPN scoreboard, game summary, pickcenter) carries prices but no ticket or
-- money percentages. Captures come from a person reading the page and running
-- scripts/record_cfb_splits.py; captured_at is the OBSERVATION time so a
-- capture pairs with the line snapshot live at that moment.
--
-- cfb_signal classifies observable facts and claims nothing about which bucket
-- wins. Buckets: reverse line movement (minority tickets, number still moved to
-- us), sharp agreement (money share exceeds ticket share by 10+), public side
-- (popular and already moved), public trap (popular and moved away), against
-- the money.


-- 7. cfb_prediction_log  -- append-only audit trail, applied 2026-09-25.
--
-- cfb_predictions upserts on (game_id, bet_type) and re-stamps predicted_at on
-- every run, so it holds only the CURRENT prediction: all 264 rows carry one
-- timestamp and the table cannot evidence what was on screen before a kickoff.
-- The values are reproducible -- snapshotting the table, re-running the logger
-- and diffing every field returned bit-identical rows, because every input is
-- the first snapshot of append-only line history, a prior-season team rating,
-- or a model file frozen 2026-08-06 before any 2026 game -- but that is a
-- property of today's code, demonstrable only by re-running it. A retrain or a
-- rule change would silently restate the whole season with nothing recording
-- that it happened.
--
-- The log closes that gap. One row per prediction per CHANGE (the 3-hourly cron
-- would otherwise append ~1,600 identical rows a day), carrying the prediction,
-- the live pre-kickoff line, the latest consensus splits, the sha256 of the
-- model files loaded and the git sha of the code. Rows are never updated or
-- deleted. scripts/check_cfb_prediction_log.py reads it and fails if any
-- prediction changed after its kickoff.
--
-- Two things worth knowing about the seed:
--   * Rows for games already played are labelled "backfill after kickoff", not
--     "first observation". They show the numbers are reproducible; they are not
--     evidence of a call made in time. 153 of the first 264 are backfill.
--   * line_now takes the last PRE-kickoff snapshot for the same reason
--     cfb_open_close does (see above). Taken naively it read TULSA +13.5 at the
--     open against -13.5 "now" -- a live line from 90 minutes into the game.
--
-- log_clv_predictions has the same upsert-and-restamp shape and is deliberately
-- left alone; the NFL half of the app is not touched by this.
