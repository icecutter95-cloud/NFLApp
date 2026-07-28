"""
Fetch historical weather for every NFL game 2018-2025 using the Open-Meteo
free historical archive API (no API key required, 10k calls/day).

Only outdoor stadium games get API calls — dome games receive default values
(72 °F, 0 wind, 0 precip, is_dome=1) without touching the network.

Caching: already-fetched game_ids in the output parquet are skipped, so the
script is safe to interrupt and re-run.

Output: data/historical_weather.parquet
    game_id, temp_fahrenheit, wind_speed_mph, precipitation_prob, is_dome

Usage:
    python fetch_historical_weather.py                  # all seasons 2018-2025
    python fetch_historical_weather.py --seasons 2024   # single season
    python fetch_historical_weather.py --force          # re-fetch everything
"""

import argparse
import time
import warnings
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import nfl_data_py as nfl

warnings.filterwarnings("ignore")

from config import DATA_DIR, ALL_HISTORICAL_SEASONS, DOME_TEAMS

# ---------------------------------------------------------------------------
# Rate limiting & retry config
# ---------------------------------------------------------------------------
SLEEP_BETWEEN_CALLS = 0.35   # seconds between Open-Meteo requests
MAX_RETRIES         = 3      # per-game retry attempts before recording NaN
RETRY_BACKOFF       = 2.0    # seconds; doubles each attempt (2, 4, 8)
CONSEC_ERROR_LIMIT  = 10     # pause for 30s if this many errors in a row


def _make_session() -> requests.Session:
    """Return a requests Session with connection-level retry + backoff."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,         # 0s, 1.5s, 3s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


_SESSION = _make_session()

# ---------------------------------------------------------------------------
# Stadium coordinates (latitude, longitude)
# Teams that relocated or moved stadiums mid-period have historical entries.
# Format: {team_abbr: (lat, lon)} or {(team_abbr, season): (lat, lon)} for
# historical overrides applied in get_stadium_coords().
# ---------------------------------------------------------------------------

STADIUM_COORDS = {
    # AFC East
    "BUF": (42.7738, -78.7870),   # Highmark Stadium, Orchard Park NY
    "MIA": (25.9580, -80.2389),   # Hard Rock Stadium, Miami Gardens FL
    "NE":  (42.0909, -71.2643),   # Gillette Stadium, Foxborough MA
    "NYJ": (40.8135, -74.0745),   # MetLife Stadium, East Rutherford NJ (open air)
    # AFC North
    "BAL": (39.2779, -76.6227),   # M&T Bank Stadium, Baltimore MD
    "CIN": (39.0955, -84.5160),   # Paycor Stadium, Cincinnati OH
    "CLE": (41.5061, -81.6995),   # Cleveland Browns Stadium, Cleveland OH
    "PIT": (40.4468, -80.0158),   # Acrisure Stadium, Pittsburgh PA
    # AFC South
    "HOU": (29.6847, -95.4107),   # NRG Stadium, Houston TX  ← dome
    "IND": (39.7601, -86.1639),   # Lucas Oil Stadium, Indianapolis IN  ← dome
    "JAX": (30.3240, -81.6373),   # EverBank Stadium, Jacksonville FL
    "TEN": (36.1665, -86.7713),   # Nissan Stadium, Nashville TN
    # AFC West
    "DEN": (39.7439, -105.0201),  # Empower Field, Denver CO
    "KC":  (39.0489, -94.4839),   # Arrowhead Stadium, Kansas City MO
    "LV":  (36.0909, -115.1833),  # Allegiant Stadium, Las Vegas NV  ← dome
    "LAC": (33.8644, -118.2611),  # SoFi Stadium, Inglewood CA  ← dome (from 2020)
    # NFC East
    "DAL": (32.7480, -97.0930),   # AT&T Stadium, Arlington TX  ← dome
    "NYG": (40.8135, -74.0745),   # MetLife Stadium (shares with NYJ)
    "PHI": (39.9008, -75.1675),   # Lincoln Financial Field, Philadelphia PA
    "WAS": (38.9076, -76.8645),   # Northwest Stadium, Landover MD
    # NFC North
    "CHI": (41.8623, -87.6167),   # Soldier Field, Chicago IL
    "DET": (42.3400, -83.0456),   # Ford Field, Detroit MI  ← dome
    "GB":  (44.5013, -88.0622),   # Lambeau Field, Green Bay WI
    "MIN": (44.9740, -93.2575),   # US Bank Stadium, Minneapolis MN  ← dome
    # NFC South
    "ATL": (33.7554, -84.4010),   # Mercedes-Benz Stadium, Atlanta GA  ← dome
    "CAR": (35.2258, -80.8531),   # Bank of America Stadium, Charlotte NC
    "NO":  (29.9511, -90.0812),   # Caesars Superdome, New Orleans LA  ← dome
    "TB":  (27.9759, -82.5033),   # Raymond James Stadium, Tampa FL
    # NFC West
    "ARI": (33.5277, -112.2626),  # State Farm Stadium, Glendale AZ  ← dome
    "LA":  (33.8644, -118.2611),  # SoFi Stadium, Inglewood CA (from 2020) — Rams; nfl_data_py uses "LA"
    "SF":  (37.4032, -121.9700),  # Levi's Stadium, Santa Clara CA
    "SEA": (47.5952, -122.3316),  # Lumen Field, Seattle WA
}

# Historical venue overrides — (team, season) → (lat, lon)
# OAK moved to LV after 2019; LA (Rams)/LAC at LA Coliseum / StubHub 2018-2019
STADIUM_OVERRIDES = {
    ("OAK", 2018): (37.7516, -122.2005),  # Oakland Coliseum
    ("OAK", 2019): (37.7516, -122.2005),
    ("LA",  2018): (34.0141, -118.2879),  # LA Coliseum — Rams (nfl_data_py: "LA")
    ("LA",  2019): (34.0141, -118.2879),
    ("LAC", 2018): (33.8644, -118.1926),  # Dignity Health Sports Park (StubHub), Carson CA
    ("LAC", 2019): (33.8644, -118.1926),
    # WAS played at FedExField (same coords as WAS) all years
}

# Teams that play in a dome (no weather effects)
# Note: LA (Rams) and LAC both moved to SoFi (dome) in 2020; before that both were outdoor
DOME_OVERRIDES = {
    # team: dome from this season onward
    "LA":  2020,   # LA Coliseum was outdoor 2018-2019, SoFi (dome) from 2020
    "LAC": 2020,   # StubHub Center was outdoor 2018-2019, SoFi (dome) from 2020
}

# Full extended dome set for easy lookup
_ALL_DOME_TEAMS = DOME_TEAMS | {"ARI", "DET", "DAL"}   # add domes missing from config


def is_dome_game(team: str, season: int) -> bool:
    """Return True if this team's home stadium is a dome for this season."""
    if team in DOME_OVERRIDES:
        return season >= DOME_OVERRIDES[team]
    return team in _ALL_DOME_TEAMS


def get_stadium_coords(team: str, season: int):
    """Return (lat, lon) for the team's home stadium in this season."""
    key = (team, season)
    if key in STADIUM_OVERRIDES:
        return STADIUM_OVERRIDES[key]
    if team in STADIUM_COORDS:
        return STADIUM_COORDS[team]
    return None   # unknown team — caller will skip


# ---------------------------------------------------------------------------
# Open-Meteo API helpers
# ---------------------------------------------------------------------------
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def precip_mm_to_prob(mm: float) -> float:
    """Convert hourly precipitation (mm) to a 0-1 probability proxy.
    0 mm → 0.0, 2 mm → ~0.5, 5+ mm → 1.0 (sigmoidal)."""
    if mm <= 0:
        return 0.0
    return float(min(1.0, mm / 5.0))


def fetch_weather_for_game(lat: float, lon: float, game_date: str,
                           kickoff_hour: int) -> dict | None:
    """
    Call Open-Meteo archive API for a single game, with manual retry + backoff.

    Returns dict with:
        temp_fahrenheit      – temperature at kickoff (°F)
        wind_speed_mph       – wind speed at kickoff (mph)
        precipitation_prob   – hourly precip converted to 0-1 probability
    Returns None after MAX_RETRIES failures.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": game_date,
        "end_date":   game_date,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "timezone": "America/New_York",
        "models": "best_match",
    }

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _SESSION.get(OPEN_METEO_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            last_exc = None
            break   # success
        except KeyboardInterrupt:
            raise   # always let the user abort
        except BaseException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"    attempt {attempt+1} failed ({exc.__class__.__name__}); "
                      f"retrying in {wait:.0f}s...")
                time.sleep(wait)
    else:
        # All retries exhausted
        print(f"    API error ({lat},{lon} {game_date}): {last_exc}")
        return None

    hourly = data.get("hourly", {})
    temps  = hourly.get("temperature_2m", [])
    winds  = hourly.get("wind_speed_10m", [])
    precip = hourly.get("precipitation",  [])

    if not temps:
        return None

    idx = kickoff_hour if kickoff_hour < len(temps) else 13   # 1 PM default

    def pick(series, h):
        for offset in range(3):
            i = min(h + offset, len(series) - 1)
            if series[i] is not None:
                return float(series[i])
        return None

    temp = pick(temps,  idx)
    wind = pick(winds,  idx)
    prec = pick(precip, idx)

    if temp is None or wind is None:
        return None

    return {
        "temp_fahrenheit":    round(temp, 1),
        "wind_speed_mph":     round(wind, 1),
        "precipitation_prob": round(precip_mm_to_prob(prec or 0.0), 3),
    }


# ---------------------------------------------------------------------------
# Kickoff hour parsing
# ---------------------------------------------------------------------------

def parse_kickoff_hour(gametime: str | None) -> int:
    """
    Parse nfl_data_py gametime string ('4:25PM', '8:20PM', etc.) → 24h hour int.
    Returns 13 (1 PM) if unparseable.
    """
    if not gametime or pd.isna(gametime):
        return 13
    try:
        gametime = str(gametime).strip().upper()
        # Handle formats: "1:00PM", "13:00", "4:25PM", "20:20"
        if "PM" in gametime or "AM" in gametime:
            t = pd.Timestamp("2000-01-01 " + gametime)
        else:
            t = pd.Timestamp("2000-01-01 " + gametime)
        hour = t.hour
        return hour
    except Exception:
        return 13


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def load_existing(out_path) -> set:
    """Return set of game_ids already in the output parquet."""
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["game_id"])
        return set(existing["game_id"].dropna())
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all games (ignore existing cache)")
    args = parser.parse_args()

    seasons = args.seasons or ALL_HISTORICAL_SEASONS
    out_path = DATA_DIR / "historical_weather.parquet"

    print(f"Fetching weather for seasons: {seasons}")
    print(f"Output: {out_path}")

    # Load schedule
    print("\nLoading schedules...")
    sched = nfl.import_schedules(seasons)
    needed = ["game_id", "home_team", "away_team", "gameday", "gametime",
              "season", "week", "game_type"]
    sched = sched[[c for c in needed if c in sched.columns]].copy()
    sched = sched[sched["game_type"] == "REG"].copy()   # regular season only
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    sched = sched.dropna(subset=["gameday"])

    print(f"  {len(sched)} regular season games to process")

    # Load existing results (for resume)
    already_done = set() if args.force else load_existing(out_path)
    if already_done:
        print(f"  {len(already_done)} games already cached — skipping")

    results = []
    api_calls = 0
    skipped_dome = 0
    skipped_missing = 0
    errors = 0
    consec_errors = 0   # consecutive API failures — triggers a long pause

    games = sched[~sched["game_id"].isin(already_done)].reset_index(drop=True)
    total = len(games)
    print(f"  {total} games to fetch\n")

    for i, row in games.iterrows():
        game_id   = row["game_id"]
        team      = row["home_team"]
        season    = int(row["season"])
        game_date = row["gameday"].strftime("%Y-%m-%d")
        kickoff_h = parse_kickoff_hour(row.get("gametime"))

        is_dome = is_dome_game(team, season)

        if is_dome:
            results.append({
                "game_id":            game_id,
                "temp_fahrenheit":    72.0,
                "wind_speed_mph":     0.0,
                "precipitation_prob": 0.0,
                "is_dome":            1,
            })
            skipped_dome += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] dome/skip: {skipped_dome}  api: {api_calls}  err: {errors}")
            continue

        coords = get_stadium_coords(team, season)
        if coords is None:
            print(f"  [{i+1}/{total}] WARNING: no coords for {team} {season} — skipping")
            skipped_missing += 1
            continue

        lat, lon = coords

        # Safety net: even if fetch_weather_for_game has a bug, never crash the loop
        try:
            weather = fetch_weather_for_game(lat, lon, game_date, kickoff_h)
        except KeyboardInterrupt:
            print("\nInterrupted by user — saving checkpoint...")
            _save(results, out_path)
            return
        except BaseException as exc:
            print(f"  [{i+1}/{total}] UNEXPECTED ERROR for {team} {season} W{row['week']}: {exc}")
            weather = None

        api_calls += 1

        if weather is None:
            if consec_errors == 0:
                # Only print once per error streak to avoid log spam
                print(f"  [{i+1}/{total}] ERROR: no weather for {team} {season} W{row['week']} ({game_date})")
            errors += 1
            consec_errors += 1

            # Store NaNs so we don't retry on next run unless --force
            results.append({
                "game_id":            game_id,
                "temp_fahrenheit":    np.nan,
                "wind_speed_mph":     np.nan,
                "precipitation_prob": np.nan,
                "is_dome":            0,
            })

            # If the API is consistently failing, take a longer break
            if consec_errors >= CONSEC_ERROR_LIMIT:
                print(f"  {consec_errors} consecutive errors — pausing 30s to let API recover...")
                _save(results, out_path)
                results = []
                time.sleep(30)
                consec_errors = 0
        else:
            consec_errors = 0   # reset streak on success
            results.append({
                "game_id":            game_id,
                "temp_fahrenheit":    weather["temp_fahrenheit"],
                "wind_speed_mph":     weather["wind_speed_mph"],
                "precipitation_prob": weather["precipitation_prob"],
                "is_dome":            0,
            })

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{total}] dome: {skipped_dome}  api: {api_calls}  err: {errors}")

        # Save checkpoint every 100 games
        if len(results) > 0 and (i + 1) % 100 == 0:
            _save(results, out_path)
            results = []

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Final save
    if results:
        _save(results, out_path)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  API calls made:   {api_calls}")
    print(f"  Dome (no call):   {skipped_dome}")
    print(f"  Missing coords:   {skipped_missing}")
    print(f"  API errors:       {errors}")
    print(f"  Output:           {out_path}")

    if out_path.exists():
        final = pd.read_parquet(out_path)
        print(f"  Total rows in file: {len(final)}")
        print(f"  Weather coverage:")
        print(f"    temp non-null:   {final['temp_fahrenheit'].notna().sum()}/{len(final)}")
        print(f"    wind non-null:   {final['wind_speed_mph'].notna().sum()}/{len(final)}")


def _save(new_rows: list, out_path):
    """Append new_rows to the parquet file (merge with existing on disk).

    Always reads whatever is currently on disk and merges into it — this
    must NOT depend on the `already_done` set captured at run start, since
    that set doesn't reflect rows written by earlier checkpoints in the
    *current* run. (A prior bug here gated the merge on that stale set,
    which stayed empty/falsy for a fresh run and caused each checkpoint to
    silently overwrite all earlier checkpoints instead of appending.)
    """
    new_df = pd.DataFrame(new_rows)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(out_path, index=False)
    print(f"    -> Checkpoint saved ({len(combined)} total rows in {out_path.name})")


if __name__ == "__main__":
    main()
