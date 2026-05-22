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
from datetime import datetime, timezone, date
from supabase import create_client

warnings.filterwarnings("ignore")

from config import (
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES,
    HFA_OVERRIDES, HFA_DEFAULT, DOME_TEAMS,
    EDGE_PER_WIN_PCT_POINT,
)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# EV / weather / tier helpers
# ---------------------------------------------------------------------------

def calculate_ev(model_line: float, dk_line: float, vig: int = -110) -> dict:
    edge_points = abs(model_line - dk_line)
    implied_prob = 110 / (110 + 100)  # 0.5238
    win_prob = min(implied_prob + edge_points * EDGE_PER_WIN_PCT_POINT, 0.85)
    payout = 100 / 110
    ev_pct = (win_prob * payout) - ((1 - win_prob) * 1.0)
    return {
        "edge_points": round(edge_points, 2),
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

def detect_steam(line_history: list, window_hours: int = 2, threshold: float = 1.0) -> bool:
    """True if line moved >= threshold points within the last window_hours."""
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
        return False
    movement = recent[-1]["spread_home"] - recent[0]["spread_home"]
    return abs(movement) >= threshold


def detect_rlm(public_bet_pct: float | None, line_movement: float) -> dict:
    """Reverse Line Movement: public on one side, line moves the other way."""
    if public_bet_pct is None:
        return {"flag": False}
    if public_bet_pct > 55 and line_movement < 0:
        return {"flag": True, "sharp_side": "away", "public_pct": public_bet_pct}
    if public_bet_pct < 45 and line_movement > 0:
        return {"flag": True, "sharp_side": "home", "public_pct": 100 - public_bet_pct}
    return {"flag": False}


# ---------------------------------------------------------------------------
# Data fetching from Supabase
# ---------------------------------------------------------------------------

def fetch_team_metrics(season: int, week: int) -> pd.DataFrame:
    """Pull team_metrics for (season, week-1) — metrics going INTO this week.

    Fallback chain when no data exists for the requested season/week:
      1. Any earlier week in the same season (most recent)
      2. End of the previous season (pre-season / week 1 of new season)
    """
    # Primary: exact week
    resp = (supabase.table("team_metrics")
            .select("*")
            .eq("season", season)
            .eq("week", week - 1)
            .execute())
    if resp.data:
        return pd.DataFrame(resp.data)

    # Fallback 1: most recent week available in this season
    resp = (supabase.table("team_metrics")
            .select("*")
            .eq("season", season)
            .order("week", desc=True)
            .limit(32)
            .execute())
    if resp.data:
        print(f"  Metrics fallback: using latest available week in {season} season")
        return pd.DataFrame(resp.data)

    # Fallback 2: end of previous season (handles pre-season / new-season week 1)
    prev = season - 1
    resp = (supabase.table("team_metrics")
            .select("*")
            .eq("season", prev)
            .order("week", desc=True)
            .limit(32)
            .execute())
    if resp.data:
        print(f"  Metrics fallback: no {season} data found, using end of {prev} season")
        return pd.DataFrame(resp.data)

    print("  WARNING: No team metrics found — model inputs will be zeroed out")
    return pd.DataFrame()


def fetch_latest_lines(game_ids: list) -> dict:
    """Most recent line_history entry per game."""
    resp = (supabase.table("line_history")
            .select("game_id, spread_home, total, recorded_at")
            .in_("game_id", game_ids)
            .order("recorded_at", desc=True)
            .execute())
    lines: dict = {}
    for row in resp.data:
        if row["game_id"] not in lines:
            lines[row["game_id"]] = row
    return lines


def fetch_line_history(game_ids: list) -> dict:
    """Full line history per game (for steam detection)."""
    resp = (supabase.table("line_history")
            .select("game_id, spread_home, total, recorded_at")
            .in_("game_id", game_ids)
            .order("recorded_at")
            .execute())
    history: dict = {}
    for row in resp.data:
        history.setdefault(row["game_id"], []).append(row)
    return history


def fetch_opening_lines(game_ids: list) -> dict:
    resp = (supabase.table("line_history")
            .select("game_id, spread_home, total")
            .eq("is_opening", True)
            .in_("game_id", game_ids)
            .execute())
    return {row["game_id"]: row for row in resp.data}


def fetch_weather(game_ids: list) -> dict:
    resp = (supabase.table("weather")
            .select("*")
            .in_("game_id", game_ids)
            .execute())
    return {row["game_id"]: row for row in resp.data}


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

        line_data = lines.get(game_id, {})
        row["dk_spread"] = float(line_data.get("spread_home", 0) or 0)
        row["dk_total"] = float(line_data.get("total", 45) or 45)

        w = weather.get(game_id, {})
        row["wind_speed_mph"] = float(w.get("wind_speed_mph", 0) or 0)
        row["temp_fahrenheit"] = float(w.get("temp_fahrenheit", 72) or 72)
        row["precipitation_prob"] = float(w.get("precipitation_prob", 0) or 0)
        row["is_dome"] = int(game["home_team"] in DOME_TEAMS or bool(w.get("is_dome", False)))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Projection rows
# ---------------------------------------------------------------------------

def build_projections(features: pd.DataFrame, lh_by_game: dict,
                      pub_by_game: dict, opening_by_game: dict) -> list:
    projections = []

    for _, row in features.iterrows():
        game_id = row["game_id"]
        lh = lh_by_game.get(game_id, [])
        pub = pub_by_game.get(game_id, {})
        opening = opening_by_game.get(game_id, {})

        # Line movement since open
        open_spread = opening.get("spread_home", row["dk_spread"])
        line_movement = float(row["dk_spread"]) - float(open_spread)

        steam = detect_steam(lh)
        public_bet_pct = pub.get("bet_pct_home")
        rlm = detect_rlm(public_bet_pct, line_movement)

        # --- Spread ---
        spread_edge = float(row["model_spread"]) - float(row["dk_spread"])
        spread_ev = calculate_ev(row["model_spread"], row["dk_spread"])
        spread_side = row["home_team"] if spread_edge > 0 else row["away_team"]
        spread_tier = assign_confidence_tier(
            abs(spread_edge), steam, rlm["flag"],
            steam_same_side=True, rlm_same_side=True
        )
        conflict = (spread_edge > 0 and line_movement < 0) or (spread_edge < 0 and line_movement > 0)

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
            "model_line": round(float(row["model_spread"]), 1),
            "dk_line": float(row["dk_spread"]),
            "edge_points": spread_ev["edge_points"],
            "ev_pct": spread_ev["ev_pct"],
            "win_probability": spread_ev["win_probability"],
            "confidence_tier": spread_tier,
            "steam_flag": steam,
            "rlm_flag": rlm["flag"],
            "rlm_sharp_side": rlm.get("sharp_side"),
            "conflict_flag": bool(conflict),
            "weather_adj": 0.0,
            "is_dome": bool(row["is_dome"]),
            "qb_override": False,
        })

        # --- Total ---
        weather_adj = weather_total_adjustment(
            row["wind_speed_mph"], row["temp_fahrenheit"], row["precipitation_prob"]
        ) if not row["is_dome"] else 0.0
        adj_model_total = float(row["model_total"]) + weather_adj
        total_edge = adj_model_total - float(row["dk_total"])
        total_ev = calculate_ev(adj_model_total, row["dk_total"])
        total_side = "over" if total_edge > 0 else "under"
        total_tier = assign_confidence_tier(abs(total_edge), steam, False, True, False)

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
            "model_line": round(adj_model_total, 1),
            "dk_line": float(row["dk_total"]),
            "edge_points": total_ev["edge_points"],
            "ev_pct": total_ev["ev_pct"],
            "win_probability": total_ev["win_probability"],
            "confidence_tier": total_tier,
            "steam_flag": steam,
            "rlm_flag": False,
            "rlm_sharp_side": None,
            "conflict_flag": False,
            "weather_adj": round(weather_adj, 1),
            "is_dome": bool(row["is_dome"]),
            "qb_override": False,
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

    # Fetch all data sources
    metrics = fetch_team_metrics(season, week)
    lines = fetch_latest_lines(game_ids)
    lh_by_game = fetch_line_history(game_ids)
    opening = fetch_opening_lines(game_ids)
    weather = fetch_weather(game_ids)
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

    # Upsert to Supabase
    supabase.table("projections").upsert(projections).execute()
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
