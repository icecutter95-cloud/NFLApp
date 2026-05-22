-- NFL Betting App — Supabase Schema
-- Run this in the Supabase SQL editor to initialize all tables.
-- Re-running is safe: all CREATE TABLE statements use IF NOT EXISTS.

-- ============================================================
-- PROJECTIONS — written by Python scoring script weekly
-- ============================================================
CREATE TABLE IF NOT EXISTS projections (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id          TEXT NOT NULL,
    season           INTEGER,
    week             INTEGER,
    game_date        DATE,
    game_time        TIMESTAMPTZ,
    home_team        TEXT,
    away_team        TEXT,
    bet_type         TEXT,          -- 'spread' | 'total'
    side             TEXT,          -- team name | 'over' | 'under'
    model_line       REAL,
    dk_line          REAL,
    edge_points      REAL,
    ev_pct           REAL,
    win_probability  REAL,
    confidence_tier  TEXT,          -- 'A' | 'B' | 'C' | 'watch'
    steam_flag       BOOLEAN DEFAULT FALSE,
    rlm_flag         BOOLEAN DEFAULT FALSE,
    rlm_sharp_side   TEXT,
    conflict_flag    BOOLEAN DEFAULT FALSE,
    weather_adj      REAL DEFAULT 0,
    is_dome          BOOLEAN DEFAULT FALSE,
    qb_override      BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- LINE_HISTORY — DraftKings spread + total per refresh
-- ============================================================
CREATE TABLE IF NOT EXISTS line_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id     TEXT NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    spread_home REAL,
    total       REAL,
    book        TEXT DEFAULT 'draftkings',
    is_opening  BOOLEAN DEFAULT FALSE
);

-- ============================================================
-- PUBLIC_BETTING — bet % and money % from ActionNetwork / Covers
-- ============================================================
CREATE TABLE IF NOT EXISTS public_betting (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id         TEXT NOT NULL,
    recorded_at     TIMESTAMPTZ DEFAULT NOW(),
    bet_pct_home    REAL,
    money_pct_home  REAL,
    bet_pct_over    REAL,
    money_pct_over  REAL,
    source          TEXT    -- 'actionnetwork' | 'covers'
);

-- ============================================================
-- WEATHER — per outdoor game, fetched Wednesday of game week
-- ============================================================
CREATE TABLE IF NOT EXISTS weather (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id             TEXT NOT NULL UNIQUE,
    stadium             TEXT,
    is_dome             BOOLEAN DEFAULT FALSE,
    wind_speed_mph      REAL DEFAULT 0,
    wind_direction      TEXT,
    temp_fahrenheit     REAL,
    precipitation_prob  REAL DEFAULT 0,
    fetched_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- INJURY_FLAGS — QB override and key injury flags
-- ============================================================
CREATE TABLE IF NOT EXISTS injury_flags (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team              TEXT NOT NULL,
    player_name       TEXT,
    position          TEXT,
    status            TEXT,   -- 'out' | 'doubtful' | 'questionable'
    is_qb_override    BOOLEAN DEFAULT FALSE,
    qb_downgrade_pts  REAL DEFAULT 0,
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- BETS — user bet log (written by React frontend)
-- ============================================================
CREATE TABLE IF NOT EXISTS bets (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id          TEXT,
    game_date        DATE,
    week             INTEGER,
    season           INTEGER,
    home_team        TEXT,
    away_team        TEXT,
    bet_type         TEXT,          -- 'spread' | 'total'
    side             TEXT,
    model_line       REAL,
    dk_line          REAL,
    edge_points      REAL,
    ev_pct           REAL,
    confidence_tier  TEXT,
    steam_flag       BOOLEAN,
    rlm_flag         BOOLEAN,
    public_bet_pct   REAL,
    odds             INTEGER,       -- American odds at time of bet (e.g. -110)
    units            REAL,
    result           TEXT DEFAULT 'pending',  -- 'win' | 'loss' | 'push' | 'pending'
    closing_line     REAL,          -- filled post-game for CLV calculation
    clv              REAL,          -- model_line vs closing_line delta
    notes            TEXT,
    logged_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- GAME_RESULTS — historical scores + ATS/OU results
-- ============================================================
CREATE TABLE IF NOT EXISTS game_results (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id              TEXT NOT NULL UNIQUE,
    season               INTEGER,
    week                 INTEGER,
    game_date            DATE,
    home_team            TEXT,
    away_team            TEXT,
    home_score           INTEGER,
    away_score           INTEGER,
    home_margin          INTEGER,
    total_points         INTEGER,
    closing_spread_home  REAL,
    closing_total        REAL,
    spread_result        TEXT,   -- 'home_covered' | 'away_covered' | 'push'
    total_result         TEXT    -- 'over' | 'under' | 'push'
);

-- ============================================================
-- TEAM_METRICS — written by Python pipeline weekly
-- ============================================================
CREATE TABLE IF NOT EXISTS team_metrics (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team                  TEXT NOT NULL,
    season                INTEGER,
    week                  INTEGER,
    epa_off_L4            REAL,
    epa_off_L8            REAL,
    epa_def_L4            REAL,
    epa_def_L8            REAL,
    epa_pass_off_L4       REAL,
    epa_rush_off_L4       REAL,
    success_rate_off_L4   REAL,
    success_rate_def_L4   REAL,
    cpoe_L4               REAL,
    cpoe_L8               REAL,
    third_down_conv_off   REAL,
    third_down_stop_def   REAL,
    rz_td_pct_off         REAL,
    pressure_rate_off     REAL,
    pressure_rate_def     REAL,
    pace_plays_per_game   REAL,
    time_to_throw         REAL,
    turnover_luck_adj     REAL,
    dvoa_off              REAL,
    dvoa_def              REAL,
    pff_team_grade        REAL,
    pff_qb_grade          REAL,
    elo_rating            REAL,
    updated_at            TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (team, season, week)
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_projections_week   ON projections (season, week);
CREATE INDEX IF NOT EXISTS idx_projections_game   ON projections (game_id);
CREATE INDEX IF NOT EXISTS idx_line_history_game  ON line_history (game_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_bets_season        ON bets (season, week);
CREATE INDEX IF NOT EXISTS idx_public_betting_game ON public_betting (game_id);
CREATE INDEX IF NOT EXISTS idx_team_metrics_team  ON team_metrics (team, season, week);

-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

-- Bets: full access (single-user personal app)
ALTER TABLE bets ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "allow_all_bets" ON bets;
CREATE POLICY "allow_all_bets" ON bets FOR ALL USING (true) WITH CHECK (true);

-- Injury flags: allow UI to write QB overrides
ALTER TABLE injury_flags ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "allow_all_injury_flags" ON injury_flags;
CREATE POLICY "allow_all_injury_flags" ON injury_flags FOR ALL USING (true) WITH CHECK (true);

-- Read-only tables (frontend reads via anon key; Python writes via service role)
ALTER TABLE projections   ENABLE ROW LEVEL SECURITY;
ALTER TABLE line_history  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public_betting ENABLE ROW LEVEL SECURITY;
ALTER TABLE weather       ENABLE ROW LEVEL SECURITY;
ALTER TABLE game_results  ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_metrics  ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "read_only_projections"    ON projections;
DROP POLICY IF EXISTS "read_only_line_history"   ON line_history;
DROP POLICY IF EXISTS "read_only_public_betting" ON public_betting;
DROP POLICY IF EXISTS "read_only_weather"        ON weather;
DROP POLICY IF EXISTS "read_only_game_results"   ON game_results;
DROP POLICY IF EXISTS "read_only_team_metrics"   ON team_metrics;

CREATE POLICY "read_only_projections"    ON projections    FOR SELECT USING (true);
CREATE POLICY "read_only_line_history"   ON line_history   FOR SELECT USING (true);
CREATE POLICY "read_only_public_betting" ON public_betting FOR SELECT USING (true);
CREATE POLICY "read_only_weather"        ON weather        FOR SELECT USING (true);
CREATE POLICY "read_only_game_results"   ON game_results   FOR SELECT USING (true);
CREATE POLICY "read_only_team_metrics"   ON team_metrics   FOR SELECT USING (true);
