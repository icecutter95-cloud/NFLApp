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
-- negated. game_id is safe as the key -- zero matchups on this board carry more
-- than one id, which is NOT true of the NFL feed.

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
