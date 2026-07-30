"""
Weekly scoring script — runs Wednesday, Saturday, and Sunday mornings.

Pulls team metrics + DK lines from Supabase, runs the trained XGBoost models,
computes EV / edge / steam / RLM / confidence tier, and upserts all projections
to the Supabase `projections` table.

Usage:
    python score_week.py                    # auto-detect current season + week
    python score_week.py 2025 8             # explicit season + week
"""

import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timezone, date, timedelta
from supabase import create_client

warnings.filterwarnings("ignore")

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES,
    HFA_OVERRIDES, HFA_DEFAULT, DOME_TEAMS,
    EDGE_PER_WIN_PCT_POINT, SPREAD_MIN_EDGE, TOTALS_MIN_EDGE, METRIC_BLEND_PSEUDO_COUNT,
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Calibrated edge -> win-probability curves, fit by calibrate_ev.py on real
# honest out-of-sample outcomes (isotonic regression, monotonic in edge size).
# Falls back to the old guessed linear formula (EDGE_PER_WIN_PCT_POINT, capped
# at 85%) only if calibrate_ev.py hasn't been run yet.
_SPREAD_CALIBRATION_PATH = MODELS_DIR / "spread_calibration.joblib"
_TOTAL_CALIBRATION_PATH  = MODELS_DIR / "total_calibration.joblib"
SPREAD_CALIBRATOR = joblib.load(_SPREAD_CALIBRATION_PATH) if _SPREAD_CALIBRATION_PATH.exists() else None
TOTAL_CALIBRATOR  = joblib.load(_TOTAL_CALIBRATION_PATH)  if _TOTAL_CALIBRATION_PATH.exists()  else None


# ---------------------------------------------------------------------------
# EV / weather / tier helpers
# ---------------------------------------------------------------------------

def calculate_ev(edge_points: float, calibrator=None, vig: int = -110) -> dict:
    """edge_points: signed edge already computed (model vs spread), passed as abs value."""
    edge_pts = abs(edge_points)
    implied_prob = 110 / (110 + 100)  # 0.5238 at -110
    if calibrator is not None:
        win_prob = float(calibrator.predict([edge_pts])[0])
    else:
        win_prob = min(implied_prob + edge_pts * EDGE_PER_WIN_PCT_POINT, 0.85)
    payout = 100 / 110
    ev_pct = (win_prob * payout) - ((1 - win_prob) * 1.0)
    return {
        "edge_points": round(edge_pts, 2),
        "win_probability": round(win_prob, 4),
        "ev_pct": round(ev_pct, 4),
        "is_positive_ev": ev_pct > 0,
    }


def weather_total_adjustment(wind_mph: float, temp_f: float, precip_prob: float) -> float:
    adj = 0.0
    if wind_mph >= 15: adj -= 1.5
    if wind_mph >= 20: adj -= 1.5   # cumulative: -3.0 at 20 mph
    if wind_mph >= 25: adj -= 2.0   # cumulative: -5.0 at 25 mph
    if temp_f <= 32:   adj -= 1.5
    if temp_f <= 20:   adj -= 1.0   # cumulative: -2.5 below 20 F
    if precip_prob >= 0.5: adj -= 1.5
    if precip_prob >= 0.8: adj -= 1.0   # cumulative: -2.5
    return adj


def assign_confidence_tier(edge: float, steam: bool, rlm_flag: bool,
                            steam_same_side: bool, rlm_same_side: bool) -> str:
    if edge >= 3 and steam and steam_same_side:
        return "A"
    if edge >= 2 and steam and steam_same_side and rlm_flag and rlm_same_side:
        return "A"
    if edge >= 2 and rlm_flag and rlm_same_side:
        return "B"
    if edge >= 2:
        return "B"
    if edge >= 1.5 and steam and steam_same_side:
        return "B"
    if edge >= 1.5:
        return "C"
    if edge >= 1 and steam and steam_same_side and rlm_flag and rlm_same_side:
        return "C"
    if edge < 1 and steam and rlm_flag:
        return "watch"
    return "watch"


# ---------------------------------------------------------------------------
# Steam / RLM detection
# ---------------------------------------------------------------------------

def detect_steam(line_history: list, market: str = "spread",
                 window_hours: int = 2, threshold: float = 1.0) -> dict:
    """True if the line moved >= threshold points within the last window_hours.

    Returns the signed movement too (not just a boolean) so callers can tell
    which DIRECTION it moved — needed to check whether the move agrees with
    the model's pick. `market` selects which column to track: line_history
    rows carry both spread_home and total in the same row, so a total bet's
    "steam" must be measured off `total`, not `spread_home` (previously every
    total projection reused the spread's steam boolean — see build_projections).
    """
    col = "spread_home" if market == "spread" else "total"
    now = datetime.now(timezone.utc)
    recent = []
    for row in line_history:
        ts = pd.Timestamp(row["recorded_at"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_hours = (now - ts.to_pydatetime()).total_seconds() / 3600
        if age_hours <= window_hours:
            recent.append(row)

    if len(recent) < 2:
        return {"flag": False, "movement": 0.0}
    movement = recent[-1][col] - recent[0][col]
    return {"flag": abs(movement) >= threshold, "movement": movement}


def _market_polarity(market: str) -> tuple[str, str]:
    """(positive_side_label, negative_side_label) for a market."""
    return ("home", "away") if market == "spread" else ("over", "under")


def _moved_toward_positive(line_movement: float, market: str) -> bool:
    """True if `line_movement` moved the market toward its "positive" side
    (home for spread, over for total); False if it moved the other way, or if
    there was no movement at all.

    The two markets have OPPOSITE sign conventions, and conflating them is
    exactly what previously inverted both conflict_flag and detect_rlm's
    sharp_side:
      spread: home_spread is NEGATIVE when home is favored (e.g. -3), so a
              move TOWARD home makes the number MORE negative.
      total:  a HIGHER total directly means more scoring expected, so a move
              TOWARD over makes the number INCREASE.
    """
    if line_movement == 0:
        return False
    return line_movement < 0 if market == "spread" else line_movement > 0


def _pick_agrees_with_movement(pick_is_positive: bool, line_movement: float, market: str) -> bool:
    """True if the market moved in the same direction as the model's pick."""
    if line_movement == 0:
        return False
    moved_positive = _moved_toward_positive(line_movement, market)
    return moved_positive if pick_is_positive else not moved_positive


def _pick_conflicts_with_movement(pick_is_positive: bool, line_movement: float, market: str) -> bool:
    """True if the market moved AGAINST the model's pick (zero movement is
    neither agreement nor conflict)."""
    if line_movement == 0:
        return False
    return not _pick_agrees_with_movement(pick_is_positive, line_movement, market)


def detect_rlm(public_pct: float | None, line_movement: float, market: str = "spread") -> dict:
    """Reverse Line Movement: the public is heavily on one side, but the line
    moved the OTHER way — sharp/professional money pushing back against the
    public tide.

    `public_pct` = % of public bets on the market's "positive" side (home for
    spread, over for total). `line_movement` = current - opening line value.
    """
    if public_pct is None or line_movement == 0:
        return {"flag": False}

    pos_label, neg_label = _market_polarity(market)
    moved_positive = _moved_toward_positive(line_movement, market)

    if public_pct > 55 and not moved_positive:
        return {"flag": True, "sharp_side": neg_label, "public_pct": public_pct}
    if public_pct < 45 and moved_positive:
        return {"flag": True, "sharp_side": pos_label, "public_pct": 100 - public_pct}
    return {"flag": False}


# ---------------------------------------------------------------------------
# Data fetching from Supabase
# ---------------------------------------------------------------------------

_NON_METRIC_COLS = {"id", "team", "season", "week", "updated_at"}


def _fetch_in_season_metrics(season: int, week: int) -> pd.DataFrame:
    """Metrics going INTO `week` of `season`.

    NOTE the join key: compute_metrics.py builds the week-W row from games
    strictly BEFORE week W, and build_dataset.py merges training data on `week`
    directly. So a week-W game must use the week-W row. This previously queried
    `week - 1`, serving the model metrics one week staler than it trained on.
    """
    resp = (supabase.table("team_metrics")
            .select("*")
            .eq("season", season)
            .eq("week", week)
            .execute())
    if resp.data:
        return pd.DataFrame(resp.data)

    # Fall back to the most recent earlier week available this season
    resp = (supabase.table("team_metrics")
            .select("*")
            .eq("season", season)
            .lt("week", week)
            .order("week", desc=True)
            .limit(32)
            .execute())
    if resp.data:
        df = pd.DataFrame(resp.data)
        print(f"  Metrics: exact week {week} unavailable, using week {int(df['week'].max())}")
        return df

    return pd.DataFrame()


def _fetch_prior_season_end_metrics(season: int) -> pd.DataFrame:
    """Each team's metrics as of the final week of the previous season."""
    prev = season - 1
    resp = (supabase.table("team_metrics")
            .select("*")
            .eq("season", prev)
            .execute())
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    idx = df.groupby("team")["week"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _blend_metrics(in_season: pd.DataFrame, prior: pd.DataFrame,
                   n: float, k: float) -> pd.DataFrame:
    """blended = (n * in_season + k * prior) / (n + k), per team per metric."""
    metric_cols = [c for c in in_season.columns
                   if c not in _NON_METRIC_COLS
                   and pd.api.types.is_numeric_dtype(in_season[c])]

    prior_idx = prior.set_index("team")
    out = in_season.copy()

    for i in out.index:
        team = out.at[i, "team"]
        if team not in prior_idx.index:
            continue
        for c in metric_cols:
            if c not in prior_idx.columns:
                continue
            cur, pri = out.at[i, c], prior_idx.at[team, c]
            if pd.isna(pri):
                continue
            if pd.isna(cur):
                out.at[i, c] = pri
            else:
                out.at[i, c] = (n * float(cur) + k * float(pri)) / (n + k)
    return out


def fetch_team_metrics(season: int, week: int) -> pd.DataFrame:
    """Team metrics going into `week`, blended with end-of-prior-season form.

    Early in a season the current-season rolling window is built from only a
    handful of games and is noisier than simply using last season (measured:
    weeks 2-4 went 63.6% -> 69.5% win rate when substituting prior-season-end
    wholesale). So we blend the two, weighting the prior by a fitted
    pseudo-count and letting it fade as real games accumulate:

        blended = (n * in_season + k * prior) / (n + k)

    with n = games played this season. Week 1 has no in-season data at all
    (n=0), so it resolves to pure prior — which is what the old fallback chain
    did, and which diagnose_preseason.py showed lifts week-1 correlation from
    +0.03 (all-zero features) to +0.47.
    """
    k = METRIC_BLEND_PSEUDO_COUNT
    in_season = _fetch_in_season_metrics(season, week)
    prior = _fetch_prior_season_end_metrics(season)

    if not in_season.empty and not prior.empty:
        # week-W row is built from W-1 games, so that's our in-season sample size
        n = float(max(int(in_season["week"].max()) - 1, 0))
        weight = k / (n + k) * 100
        print(f"  Metrics: blending {season} wk{int(in_season['week'].max())} "
              f"with end of {season - 1} (n={n:.0f}, k={k:g} -> {weight:.0f}% prior)")
        return _blend_metrics(in_season, prior, n, k)

    if not in_season.empty:
        print(f"  Metrics: no {season - 1} data to blend with — using {season} in-season only")
        return in_season

    if not prior.empty:
        print(f"  Metrics: no {season} data yet — using end of {season - 1} (pure prior)")
        return prior

    print("  WARNING: No team metrics found — model inputs will be zeroed out")
    return pd.DataFrame()


_PAGE_SIZE = 1000   # PostgREST caps a single response at 1000 rows


def _fetch_paged(build_query, page_size: int = _PAGE_SIZE) -> list:
    """Run a PostgREST query in pages until exhausted.

    Supabase silently truncates any single response at 1000 rows -- it does not
    error, it just returns fewer rows than exist. line_history grows by ~272
    rows per refresh (one per game on the full-season slate), so unpaginated
    reads here would quietly start dropping data within days of the cron going
    live, corrupting steam detection with no visible failure. `build_query`
    must return a FRESH query builder each call, since .range() mutates it.
    """
    rows: list = []
    start = 0
    while True:
        resp = build_query().range(start, start + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        start += page_size


def fetch_latest_lines(team_pairs: list) -> dict:
    """Most recent line_history entry per (home_team, away_team) game.

    line_history.game_id is The Odds API's own opaque event ID, not the
    nfl_data_py game_id used everywhere else in the pipeline — the two never
    match. refresh-odds stores home_team/away_team (translated to standard
    abbreviations) alongside it specifically so this join can happen here.
    """
    if not team_pairs:
        return {}
    home_teams = list({h for h, _ in team_pairs})
    valid_pairs = set(team_pairs)
    rows = _fetch_paged(lambda: supabase.table("line_history")
                        .select("home_team, away_team, spread_home, total, recorded_at")
                        .in_("home_team", home_teams)
                        .order("recorded_at", desc=True))
    lines: dict = {}
    for row in rows:
        key = (row.get("home_team"), row.get("away_team"))
        if key in valid_pairs and key not in lines:
            lines[key] = row
    return lines


def fetch_line_history(team_pairs: list, lookback_hours: int = 48) -> dict:
    """Recent line history per (home_team, away_team) game, for steam detection.

    Bounded to the last `lookback_hours` because detect_steam only inspects a
    2-hour window -- pulling a whole season of snapshots per team would be
    wasteful and, once the table is large, slow.
    """
    if not team_pairs:
        return {}
    home_teams = list({h for h, _ in team_pairs})
    valid_pairs = set(team_pairs)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    rows = _fetch_paged(lambda: supabase.table("line_history")
                        .select("home_team, away_team, spread_home, total, recorded_at")
                        .in_("home_team", home_teams)
                        .gte("recorded_at", cutoff)
                        .order("recorded_at"))
    history: dict = {}
    for row in rows:
        key = (row.get("home_team"), row.get("away_team"))
        if key in valid_pairs:
            history.setdefault(key, []).append(row)
    return history


def fetch_opening_lines(team_pairs: list) -> dict:
    """Opening line per game, read from the line_open_close view.

    The view is aggregated to one row per game, so it cannot be truncated by
    the 1000-row cap the way a raw line_history scan can. It also derives the
    opener from the earliest recorded_at rather than trusting the is_opening
    flag, which is set at insert time and therefore can't be corrected after
    the fact.
    """
    if not team_pairs:
        return {}
    home_teams = list({h for h, _ in team_pairs})
    valid_pairs = set(team_pairs)
    rows = _fetch_paged(lambda: supabase.table("line_open_close")
                        .select("home_team, away_team, opening_spread_home, opening_total")
                        .in_("home_team", home_teams))
    out: dict = {}
    for row in rows:
        key = (row.get("home_team"), row.get("away_team"))
        if key in valid_pairs:
            # Normalise to the column names build_projections expects.
            out[key] = {"spread_home": row.get("opening_spread_home"),
                        "total": row.get("opening_total")}
    return out


def fetch_weather(team_pairs: list) -> dict:
    """Forecast conditions at kickoff, keyed by (home_team, away_team).

    weather.game_id is The Odds API's event ID (refresh-weather reads upcoming
    games from line_open_close), NOT the nfl_data_py game_id used elsewhere —
    the same mismatch that broke the line joins. So join on the team pair,
    consistently with fetch_latest_lines/fetch_opening_lines.
    """
    if not team_pairs:
        return {}
    home_teams = list({h for h, _ in team_pairs})
    valid_pairs = set(team_pairs)
    rows = _fetch_paged(lambda: supabase.table("weather")
                        .select("*")
                        .in_("home_team", home_teams))
    out: dict = {}
    for row in rows:
        key = (row.get("home_team"), row.get("away_team"))
        if key in valid_pairs:
            out[key] = row
    return out


def fetch_public_betting(game_ids: list) -> dict:
    """Most recent public betting entry per game."""
    resp = (supabase.table("public_betting")
            .select("*")
            .in_("game_id", game_ids)
            .order("recorded_at", desc=True)
            .execute())
    pub: dict = {}
    for row in resp.data:
        if row["game_id"] not in pub:
            pub[row["game_id"]] = row
    return pub


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def fetch_current_schedule(season: int, week: int) -> pd.DataFrame:
    sched = nfl.import_schedules([season])
    return sched[(sched["week"] == week) & (sched["game_type"] == "REG")].copy()


def current_week_number(season: int) -> int:
    sched = nfl.import_schedules([season])
    today = date.today()
    upcoming = sched[pd.to_datetime(sched["gameday"]).dt.date >= today]
    if upcoming.empty:
        return 1
    return int(upcoming["week"].min())


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

# Maps Supabase DB column name → feature name the model was trained on.
# Postgres lowercases all unquoted identifiers, so DB columns are all-lowercase.
METRIC_RENAME = {
    "epa_off_l4":          "epa_per_play_off_L4",
    "epa_off_l8":          "epa_per_play_off_L8",
    "epa_def_l4":          "epa_per_play_def_L4",
    "epa_def_l8":          "epa_per_play_def_L8",
    "epa_pass_off_l4":     "epa_pass_off_L4",
    "epa_rush_off_l4":     "epa_rush_off_L4",
    "success_rate_off_l4": "success_rate_off_L4",
    "success_rate_def_l4": "success_rate_def_L4",
    "cpoe_l4":             "cpoe_L4",
    "cpoe_l8":             "cpoe_L8",
    "third_down_conv_off":  "third_down_conv_off_season",
    "third_down_stop_def":  "third_down_stop_def_season",
    "rz_td_pct_off":        "rz_td_pct_off_season",
    "pace_plays_per_game":  "plays_per_game_L4",
    "turnover_luck_adj":    "turnover_margin_L8",
    # Scoring volume (totals model)
    "pts_scored_off_l4":    "points_scored_off_L4",
    "pts_scored_off_l8":    "points_scored_off_L8",
    "pts_allowed_def_l4":   "points_allowed_def_L4",
    "pts_allowed_def_l8":   "points_allowed_def_L8",
}


def _build_game_time(gameday, gametime) -> str | None:
    """Combine nfl_data_py gameday ('2026-09-10') + gametime ('20:20')
    into a full ISO timestamp Postgres can parse as timestamptz."""
    try:
        day = str(gameday).strip()
        t   = str(gametime).strip()
        if day and t and t not in ("", "nan", "None", "NaT"):
            return f"{day}T{t}:00"   # e.g. "2026-09-10T20:20:00"
    except Exception:
        pass
    return None


def build_feature_matrix(games: pd.DataFrame, metrics: pd.DataFrame,
                         lines: dict, weather: dict) -> pd.DataFrame:
    """Build one feature row per game, ready to pass to the models."""
    metric_cols = [c for c in metrics.columns
                   if c not in ("id", "team", "season", "week", "updated_at")]

    has_metrics = not metrics.empty and "team" in metrics.columns

    rows = []
    for _, game in games.iterrows():
        home_m = metrics[metrics["team"] == game.get("home_team", "")] if has_metrics else pd.DataFrame()
        away_m = metrics[metrics["team"] == game.get("away_team", "")] if has_metrics else pd.DataFrame()

        game_id = game.get("game_id", f"{game['home_team']}_{game['away_team']}_{game['week']}")

        row: dict = {
            "game_id": game_id,
            "season": int(game["season"]),
            "week": int(game["week"]),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "game_date": str(game.get("gameday", "")),
            "game_time": _build_game_time(game.get("gameday"), game.get("gametime")),
            "home_field_advantage": HFA_OVERRIDES.get(game["home_team"], HFA_DEFAULT),
            "is_divisional": int(game.get("div_game", 0) or 0),
            "week_number": int(game["week"]),
            "rest_days_home": float(game.get("home_rest", 7) or 7),
            "rest_days_away": float(game.get("away_rest", 7) or 7),
            "had_bye_home": 0,
            "had_bye_away": 0,
            "is_short_week_home": 0,
            "is_short_week_away": 0,
        }
        row["rest_diff"] = row["rest_days_home"] - row["rest_days_away"]

        for col in metric_cols:
            feat_name = METRIC_RENAME.get(col, col)
            h_val = home_m[col].iloc[0] if len(home_m) > 0 and col in home_m.columns else None
            a_val = away_m[col].iloc[0] if len(away_m) > 0 and col in away_m.columns else None
            row[f"{feat_name}_home"] = float(h_val) if h_val is not None else np.nan
            row[f"{feat_name}_away"] = float(a_val) if a_val is not None else np.nan

        line_data = lines.get((game["home_team"], game["away_team"]), {})
        row["dk_spread"] = float(line_data.get("spread_home", 0) or 0)
        row["dk_total"] = float(line_data.get("total", 45) or 45)

        w = weather.get((game["home_team"], game["away_team"]), {})
        row["wind_speed_mph"] = float(w.get("wind_speed_mph", 0) or 0)
        row["temp_fahrenheit"] = float(w.get("temp_fahrenheit", 72) or 72)
        row["precipitation_prob"] = float(w.get("precipitation_prob", 0) or 0)
        row["is_dome"] = int(game["home_team"] in DOME_TEAMS or bool(w.get("is_dome", False)))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Projection rows
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    """Convert to a JSON-serializable float, or None for NaN/missing.
    (Postgres/Supabase reject NaN as a REAL value.)"""
    if val is None:
        return None
    val = float(val)
    return None if np.isnan(val) else val


def build_projections(features: pd.DataFrame, lh_by_game: dict,
                      pub_by_game: dict, opening_by_game: dict) -> list:
    projections = []

    for _, row in features.iterrows():
        game_id = row["game_id"]
        team_pair = (row["home_team"], row["away_team"])
        lh = lh_by_game.get(team_pair, [])
        pub = pub_by_game.get(game_id, {})
        opening = opening_by_game.get(team_pair, {})

        # --- Spread market signals ---
        open_spread = opening.get("spread_home", row["dk_spread"])
        spread_line_movement = float(row["dk_spread"]) - float(open_spread)

        spread_steam = detect_steam(lh, market="spread")
        public_bet_pct = pub.get("bet_pct_home")
        rlm = detect_rlm(public_bet_pct, spread_line_movement, market="spread")

        # --- Spread ---
        # model_spread now predicts home_cover_surplus directly (positive = home covers).
        # No need to combine with dk_spread — the model output IS the edge.
        spread_edge = float(row["model_spread"])
        spread_ev = calculate_ev(abs(spread_edge), SPREAD_CALIBRATOR)
        spread_pick_home = spread_edge > 0
        spread_side = row["home_team"] if spread_pick_home else row["away_team"]

        spread_steam_same_side = _pick_agrees_with_movement(
            spread_pick_home, spread_steam["movement"], "spread")
        spread_rlm_same_side = rlm["flag"] and (
            (rlm.get("sharp_side") == "home") == spread_pick_home)

        spread_tier = assign_confidence_tier(
            abs(spread_edge), spread_steam["flag"], rlm["flag"],
            steam_same_side=spread_steam_same_side, rlm_same_side=spread_rlm_same_side
        )
        conflict = _pick_conflicts_with_movement(spread_pick_home, spread_line_movement, "spread")

        # For display: convert cover_surplus back to projected home margin
        # home_cover_surplus = home_margin + dk_spread
        # → projected_home_margin = model_spread - dk_spread
        projected_margin = round(float(row["model_spread"]) - float(row["dk_spread"]), 1)

        # Only surface a spread bet if it clears the gate. SPREAD_MIN_EDGE is
        # currently 999 (effectively off) because the honest backtest shows
        # 52.5% at edge>=1.5 against a 52.38% break-even — no edge. Gated
        # independently of the totals block below so either can be re-enabled
        # on its own.
        if abs(spread_edge) >= SPREAD_MIN_EDGE:
            projections.append({
                "game_id": game_id,
                "season": int(row["season"]),
                "week": int(row["week"]),
                "game_date": row["game_date"],
                "game_time": row["game_time"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_type": "spread",
                "side": spread_side,
                "model_line": projected_margin,  # projected home margin (more intuitive than surplus)
                "dk_line": float(row["dk_spread"]),
                "edge_points": spread_ev["edge_points"],
                "ev_pct": spread_ev["ev_pct"],
                "win_probability": spread_ev["win_probability"],
                "confidence_tier": spread_tier,
                "steam_flag": spread_steam["flag"],
                "rlm_flag": rlm["flag"],
                "rlm_sharp_side": rlm.get("sharp_side"),
                "conflict_flag": bool(conflict),
                "weather_adj": 0.0,
                "is_dome": bool(row["is_dome"]),
                "qb_override": False,
                "home_epa_off": _safe_float(row.get("epa_per_play_off_L4_home")),
                "away_epa_off": _safe_float(row.get("epa_per_play_off_L4_away")),
                "home_epa_def": _safe_float(row.get("epa_per_play_def_L4_home")),
                "away_epa_def": _safe_float(row.get("epa_per_play_def_L4_away")),
                "home_cpoe": _safe_float(row.get("cpoe_L4_home")),
                "away_cpoe": _safe_float(row.get("cpoe_L4_away")),
            })

        # --- Total market signals ---
        open_total = opening.get("total", row["dk_total"])
        total_line_movement = float(row["dk_total"]) - float(open_total)

        total_steam = detect_steam(lh, market="total")
        public_bet_pct_over = pub.get("bet_pct_over")
        total_rlm = detect_rlm(public_bet_pct_over, total_line_movement, market="total")

        # --- Total ---
        # model_total predicts ou_surplus directly (positive = over hits).
        # Apply weather adjustment (negative = suppress total in bad conditions).
        weather_adj = weather_total_adjustment(
            row["wind_speed_mph"], row["temp_fahrenheit"], row["precipitation_prob"]
        ) if not row["is_dome"] else 0.0
        total_edge = float(row["model_total"]) + weather_adj
        total_ev = calculate_ev(abs(total_edge), TOTAL_CALIBRATOR)
        total_pick_over = total_edge > 0
        total_side = "over" if total_pick_over else "under"

        total_steam_same_side = _pick_agrees_with_movement(
            total_pick_over, total_steam["movement"], "total")
        total_rlm_same_side = total_rlm["flag"] and (
            (total_rlm.get("sharp_side") == "over") == total_pick_over)

        total_tier = assign_confidence_tier(
            abs(total_edge), total_steam["flag"], total_rlm["flag"],
            steam_same_side=total_steam_same_side, rlm_same_side=total_rlm_same_side
        )
        total_conflict = _pick_conflicts_with_movement(total_pick_over, total_line_movement, "total")

        # Only include total if edge exceeds the minimum threshold.
        # TOTALS_MIN_EDGE is set high (10) until weather data is in the training set —
        # model validation corr = -0.028 means no predictive power on totals right now.
        if abs(total_edge) < TOTALS_MIN_EDGE:
            continue  # skip this total bet, move to next game

        # For display: convert ou_surplus back to projected combined score
        # ou_surplus = combined_score - dk_total
        # → projected_combined_score = model_total + dk_total
        projected_total = round(float(row["model_total"]) + weather_adj + float(row["dk_total"]), 1)

        projections.append({
            "game_id": game_id,
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_date": row["game_date"],
            "game_time": row["game_time"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "bet_type": "total",
            "side": total_side,
            "model_line": projected_total,  # projected combined score
            "dk_line": float(row["dk_total"]),
            "edge_points": total_ev["edge_points"],
            "ev_pct": total_ev["ev_pct"],
            "win_probability": total_ev["win_probability"],
            "confidence_tier": total_tier,
            "steam_flag": total_steam["flag"],
            "rlm_flag": total_rlm["flag"],
            "rlm_sharp_side": total_rlm.get("sharp_side"),
            "conflict_flag": bool(total_conflict),
            "weather_adj": round(weather_adj, 1),
            "is_dome": bool(row["is_dome"]),
            "qb_override": False,
            "home_epa_off": _safe_float(row.get("epa_per_play_off_L4_home")),
            "away_epa_off": _safe_float(row.get("epa_per_play_off_L4_away")),
            "home_epa_def": _safe_float(row.get("epa_per_play_def_L4_home")),
            "away_epa_def": _safe_float(row.get("epa_per_play_def_L4_away")),
            "home_cpoe": _safe_float(row.get("cpoe_L4_home")),
            "away_cpoe": _safe_float(row.get("cpoe_L4_away")),
        })

    return projections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_weekly_scoring(season: int, week: int):
    print(f"\nScoring: season={season}, week={week}")

    # Load schedule
    games = fetch_current_schedule(season, week)
    if games.empty:
        print("No regular-season games found for this week.")
        return
    print(f"  {len(games)} games found")

    game_ids = games["game_id"].tolist() if "game_id" in games.columns else []
    team_pairs = list(zip(games["home_team"], games["away_team"]))

    # Fetch all data sources
    metrics = fetch_team_metrics(season, week)
    lines = fetch_latest_lines(team_pairs)
    lh_by_game = fetch_line_history(team_pairs)
    opening = fetch_opening_lines(team_pairs)
    weather = fetch_weather(team_pairs)
    pub = fetch_public_betting(game_ids)

    # Build feature matrix
    features = build_feature_matrix(games, metrics, lines, weather)

    # Load models
    spread_model = joblib.load(MODELS_DIR / "spread_model.joblib")
    total_model = joblib.load(MODELS_DIR / "total_model.joblib")

    # Expose current DK lines as the market feature the model was trained with
    features["market_spread_home"] = features["dk_spread"]
    features["market_total"]       = features["dk_total"]

    # Score — use the exact feature list the model was trained on,
    # not the config list (they may differ if weather cols weren't in training data).
    spread_feat_cols = spread_model.get_booster().feature_names
    total_feat_cols  = total_model.get_booster().feature_names

    # Add any missing columns the model expects (fill with 0)
    for col in spread_feat_cols + total_feat_cols:
        if col not in features.columns:
            features[col] = 0.0

    features["model_spread"] = spread_model.predict(features[spread_feat_cols].fillna(0))
    features["model_total"]  = total_model.predict(features[total_feat_cols].fillna(0))

    # Build projection rows + signals
    projections = build_projections(features, lh_by_game, pub, opening)

    # Fully replace this week's projections rather than upsert-in-place.
    # A row that qualified in a previous run (e.g. a total bet that cleared
    # TOTALS_MIN_EDGE due to a stale/mismatched dk_total) but doesn't qualify
    # this run would otherwise be silently orphaned forever — upsert() only
    # touches rows present in the current payload, it never removes rows that
    # are no longer emitted. Delete-then-insert guarantees the table always
    # reflects exactly the current model's output for this week, nothing stale.
    supabase.table("projections").delete().eq("season", season).eq("week", week).execute()
    if projections:
        supabase.table("projections").insert(projections).execute()
    print(f"  Upserted {len(projections)} projection rows")

    # Print summary
    for p in projections:
        tier_label = {"A": "🔥A", "B": "⭐B", "C": "📊C", "watch": "👀"}.get(p["confidence_tier"], p["confidence_tier"])
        steam_label = " ⚡STEAM" if p["steam_flag"] else ""
        rlm_label = " 🔄RLM" if p["rlm_flag"] else ""
        print(f"  {p['home_team']} vs {p['away_team']} | {p['bet_type'].upper()} {p['side']} "
              f"| edge={p['edge_points']:+.1f} EV={p['ev_pct']:+.1%} {tier_label}{steam_label}{rlm_label}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _season = int(sys.argv[1])
        _week = int(sys.argv[2])
    else:
        _season = datetime.now().year
        _week = current_week_number(_season)

    run_weekly_scoring(_season, _week)
