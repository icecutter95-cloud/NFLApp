"""Shared constants, paths, and feature lists for the NFL betting pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Supabase / API credentials (loaded from .env)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY", "")

# ---------------------------------------------------------------------------
# Season splits
# ---------------------------------------------------------------------------
TRAIN_SEASONS = list(range(2018, 2026))       # 2018–2025 (all 8 confirmed seasons)
VALIDATE_SEASONS = [2023, 2024]               # kept for reference / re-evaluation
TEST_SEASON = 2025                             # confirmed — now part of training
ALL_HISTORICAL_SEASONS = list(range(2018, 2026))  # 2018–2025 (includes completed 2025 season)
CURRENT_SEASON = 2026                          # upcoming season for live projections

# ---------------------------------------------------------------------------
# Home field advantage
# ---------------------------------------------------------------------------
HFA_DEFAULT = 2.5
HFA_OVERRIDES = {
    "GB": 3.5,
    "KC": 3.2,
    "SEA": 3.0,
    "BUF": 3.0,
    "SF": 2.8,
    "LV": 1.8,
    "LA": 1.5,   # Rams — nfl_data_py labels this team "LA", not "LAR"
    "LAC": 1.5,
}

# ---------------------------------------------------------------------------
# Stadium classification
# ---------------------------------------------------------------------------
DOME_TEAMS = {"NO", "ATL", "LV", "LA", "LAC", "MIN", "IND", "HOU", "ARI", "DET", "DAL"}
# NOTE: Rams are "LA" in nfl_data_py's schedule data, not "LAR" — LA and LAC both moved
# into SoFi Stadium (a dome) in 2020; before that they were outdoor (see build_dataset.py
# and fetch_historical_weather.py for the season-gated dome overrides for these two teams).
OUTDOOR_COLD_TEAMS = {"GB", "CHI", "BUF", "PIT", "CLE", "NYG", "NYJ", "NE", "KC", "DEN"}

# ---------------------------------------------------------------------------
# Feature lists (must match column names produced by build_dataset.py)
# ---------------------------------------------------------------------------

_TEAM_METRIC_COLS = [
    # Opponent-ADJUSTED EPA / success rate. Raw versions are still computed by
    # compute_metrics.py (epa_per_play_off_L4 etc.) — swap these back to compare.
    # Adjustment subtracts the faced unit's established strength, so a big number
    # against a bad defense no longer counts the same as one against a good defense.
    "epa_per_play_off_adj_L4", "epa_per_play_off_adj_L8",
    "epa_per_play_def_adj_L4", "epa_per_play_def_adj_L8",
    "epa_pass_off_L4", "epa_rush_off_L4",
    "success_rate_off_adj_L4", "success_rate_def_adj_L4",
    "cpoe_L4", "cpoe_L8",
    "third_down_conv_off_season", "third_down_stop_def_season",
    "rz_td_pct_off_season",
    "plays_per_game_L4",
    "turnover_margin_L8",
    # Scoring volume — key for totals model
    "points_scored_off_L4", "points_scored_off_L8",
    "points_allowed_def_L4", "points_allowed_def_L8",
]

SPREAD_FEATURES = (
    [f"{c}_home" for c in _TEAM_METRIC_COLS]
    + [f"{c}_away" for c in _TEAM_METRIC_COLS]
    + [
        "home_field_advantage",
        "rest_diff",
        "is_divisional",
        "is_short_week_home",
        "is_short_week_away",
        "had_bye_home",
        "had_bye_away",
        "week_number",
        # --- Tier 2 additions, each measured independently on held-out
        # 2023-2025 (margin corr / MAE vs a 0.354 / 10.40 base):
        #   travel            0.369 / 10.38   kept
        #   injuries          0.370 / 10.39   kept
        #   travel+injuries   0.374 / 10.38   kept (best)
        #   trench metrics    0.356 / 10.50   REJECTED — corr flat, MAE worse
        #   all three         0.365 / 10.49   worse than the pair: feature bloat
        # Sack/QB-hit rates are still computed by compute_metrics.py so they can
        # be re-tested, they are just not fed to the model.
        "away_travel_miles",
        "tz_delta",
        "abs_tz_delta",
        "inj_qb_out_home", "inj_qb_out_away",
        "inj_out_off_home", "inj_out_off_away",
        "inj_out_def_home", "inj_out_def_away",
        "inj_out_total_home", "inj_out_total_away",
        "inj_questionable_home", "inj_questionable_away",
        # NOTE: market_spread_home intentionally excluded.
        # Including the closing line as a feature creates a circular dependency:
        # edge = model_pred + closing_line, but model was trained ON the closing line,
        # so model_pred ≈ -closing_line + noise → edge ≈ noise (not genuine alpha).
        # Without market lines, model predicts from team efficiency only and the edge
        # vs. the closing line is a pure, interpretable measure of market disagreement.
    ]
)

TOTAL_FEATURES = SPREAD_FEATURES + [
    "wind_speed_mph",
    "temp_fahrenheit",
    "precipitation_prob",
    "is_dome",
    # market_total also excluded for the same reason as market_spread_home above.
]

# ---------------------------------------------------------------------------
# EV / confidence defaults
# ---------------------------------------------------------------------------
EV_DISPLAY_THRESHOLD = 0.0    # show bets with EV% > 0%
EV_ACTION_THRESHOLD = 0.03    # recommend bets with EV% > 3%
EDGE_PER_WIN_PCT_POINT = 0.03  # each point of edge ≈ 3% win prob shift (tune in backtesting)

# Empirical-Bayes pseudo-count for blending current-season rolling team metrics
# with end-of-prior-season metrics:
#     blended = (n * in_season + k * prior) / (n + k),   n = games played this season
# Week 1 (n=0) is therefore pure prior; the prior fades as real games accumulate.
#
# NOT CURRENTLY VALIDATED. k=3 was originally fitted via calibrate_metric_blend.py,
# but that fit ran against the spread-sign-corrupted target (see the sign
# normalisation note in build_dataset.add_closing_lines). Re-running the sweep on
# the corrected target produces pure noise -- no unimodal optimum, correlations
# ~0.06, ROI ordering essentially random -- so there is no empirical basis for any
# particular k right now.
#
# The blend MECHANISM is still kept because it is structurally sound independent
# of the fit: compute_metrics.py cannot build a week-1 rolling window (no prior
# games), so without a prior every week-1 team feature is NaN -> filled to 0.
# Falling back to last season's form is plainly better than feeding the model
# all zeros. The specific value below is a reasonable default, not a fitted one.
METRIC_BLEND_PSEUDO_COUNT = 3.0

# ---------------------------------------------------------------------------
# Live recommendation gates
# ---------------------------------------------------------------------------
# Both models are currently gated OFF because neither demonstrates an edge on
# an honest out-of-sample backtest (train 2018-2022, evaluate 2023-2025).
#
# SPREAD: 52.5% win rate at edge>=1.5 (261-236) vs a 52.38% break-even at -110,
# i.e. ROI +0.3% -- statistically indistinguishable from zero. The previously
# reported 70-75% was entirely an artifact of an inverted spread sign that made
# the training target `home_margin + spread_line` instead of
# `home_margin - spread_line`; the corrupted target equalled
# `correct + 2*spread_line`, so the model largely learned to reconstruct the
# market's own line (corr(model, closing_line) was +0.741, vs +0.009 after the
# fix) and was then scored against a target containing the same term on both
# sides. Raise/lower this gate only on the back of a fresh honest backtest.
#
# TOTALS: no signal either -- validation corr ~0, ROI negative across seasons,
# and unaffected by the sign bug (totals have no home/away polarity). Adding
# real historical weather did not help, because the market prices weather in.
SPREAD_MIN_EDGE = 999.0   # effectively disabled; no demonstrated edge
TOTALS_MIN_EDGE = 999.0   # effectively disabled; no demonstrated edge
