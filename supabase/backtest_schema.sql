-- Run this in Supabase Dashboard → SQL Editor
-- Adds the backtest_results table for historical model performance

create table if not exists backtest_results (
  id             bigint generated always as identity primary key,
  season         int     not null,
  week           int,
  game_id        text,
  home_team      text    not null,
  away_team      text    not null,
  bet_type       text    not null,   -- 'spread' | 'total'
  side           text,               -- team name, 'over', or 'under'
  model_line     numeric(6,2),
  closing_line   numeric(6,2),
  actual_result  numeric(6,2),       -- home_margin (spread) or combined_score (total)
  edge_points    numeric(5,2),
  ev_pct         numeric(8,4),
  result         text,               -- 'win' | 'loss' | 'push'
  units          numeric(6,4),       -- +payout or -1.0 or 0
  created_at     timestamptz default now()
);

create index if not exists backtest_results_season_type on backtest_results(season, bet_type);
create index if not exists backtest_results_edge        on backtest_results(bet_type, edge_points);

alter table backtest_results enable row level security;
create policy "anon read backtest" on backtest_results for select using (true);
