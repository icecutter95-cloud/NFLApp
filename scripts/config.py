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
TRAIN_SEASONS = list(range(2018, 2023))       # 2018–2022 (5 seasons)
VALIDATE_SEASONS = [2023, 2024]               # out-of-sample evaluation
TEST_SEASON = 2025                             # 2025 is now complete — holdout
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
    "LAR": 1.5,
    "LAC": 1.5,
}

# ---------------------------------------------------------------------------
# Stadium classification
# ---------------------------------------------------------------------------
DOME_TEAMS = {"NO", "ATL", "LV", "LAR", "LAC", "MIN", "IND", "HOU"}
OUTDOOR_COLD_TEAMS = {"GB", "CHI", "BUF", "PIT", "CLE", "NYG", "NYJ", "NE", "KC", "DEN"}

# ---------------------------------------------------------------------------
# Feature lists (must match column names produced by build_dataset.py)
# ---------------------------------------------------------------------------

_TEAM_METRIC_COLS = [
    "epa_per_play_off_L4", "epa_per_play_off_L8",
    "epa_per_play_def_L4", "epa_per_play_def_L8",
    "epa_pass_off_L4", "epa_rush_off_L4",
    "success_rate_off_L4", "success_rate_def_L4",
    "cpoe_L4", "cpoe_L8",
    "third_down_conv_off_season", "third_down_stop_def_season",
    "rz_td_pct_off_season",
    "plays_per_game_L4",
    "turnover_margin_L8",
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
    ]
)

TOTAL_FEATURES = SPREAD_FEATURES + [
    "wind_speed_mph",
    "temp_fahrenheit",
    "precipitation_prob",
    "is_dome",
]

# ---------------------------------------------------------------------------
# EV / confidence defaults
# ---------------------------------------------------------------------------
EV_DISPLAY_THRESHOLD = 0.0    # show bets with EV% > 0%
EV_ACTION_THRESHOLD = 0.03    # recommend bets with EV% > 3%
EDGE_PER_WIN_PCT_POINT = 0.03  # each point of edge ≈ 3% win prob shift (tune in backtesting)
