"""
Backfill historical OPENING lines and intraweek line movement from The Odds API.

Why this exists
---------------
nflverse only publishes CLOSING lines, and the closing line is the hardest
benchmark in sports betting -- it already contains every public signal we have
tested (weather, roster turnover, coaching changes, EPA). Three tiers of feature
work moved margin prediction from corr 0.340 to 0.374 without moving the betting
metric at all.

Opening lines are a different, softer target: posted by one book with limited
information, hours or days before the market corrects them. With open->close
history we can finally ask a tractable question -- does our disagreement with
the OPENER predict which way the line moves? -- instead of the intractable one,
which is whether we beat the close.

The Odds API historical endpoint covers June 2020 onward at 5-minute
granularity, and one call returns every currently-listed game.

Cost: ~10 credits per market per call. Snapshots are sampled a few times per
week rather than daily; the exact opening/closing values are derived afterward
from the snapshot series plus each game's commence_time, so denser sampling
buys precision, not correctness.

Usage:
    python fetch_historical_lines.py 2023          # one season
    python fetch_historical_lines.py 2020 2025     # inclusive range
    python fetch_historical_lines.py 2023 --dry-run  # show plan + cost, no calls
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import requests
import nfl_data_py as nfl
from datetime import timedelta

warnings.filterwarnings("ignore")

from config import DATA_DIR, ODDS_API_KEY

HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
BOOKMAKER = "draftkings"
MARKETS = "spreads,totals"
REGIONS = "us"
SLEEP_BETWEEN_CALLS = 1.0
EARLIEST_SUPPORTED = pd.Timestamp("2020-06-01T00:00:00Z")
# A game's weekly board typically posts the Sunday/Monday before kickoff.
# Snapshots inside this window represent the actionable weekly opener.
WEEKLY_OPEN_WINDOW_DAYS = 10

# The Odds API returns full team names; everything else in this pipeline keys on
# standard abbreviations. NOTE the Rams are "LA", not "LAR", matching
# nfl_data_py (this exact mismatch previously made every Rams game invisible).
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    # Pre-rename franchises, so 2020-2021 snapshots map correctly.
    "Oakland Raiders": "LV", "Washington Football Team": "WAS",
    "Washington Redskins": "WAS", "San Diego Chargers": "LAC",
    "St. Louis Rams": "LA",
}


def snapshot_schedule(season: int) -> list:
    """Timestamps to sample for a season.

    Anchored to each week's first kickoff so the cadence follows the real
    betting week rather than the calendar:
      -3d  Monday-ish, when next week's numbers are freshly posted
      -1d  midweek, after early money has moved things
      +3d  Sunday ~noon ET, immediately before the main slate kicks
    Plus a few preseason samples, since many games are listed months early and
    their true opener predates week 1.
    """
    sched = nfl.import_schedules([season])
    sched = sched[sched["game_type"] == "REG"].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    sched = sched.dropna(subset=["gameday"])
    if sched.empty:
        return []

    stamps = []
    season_start = sched["gameday"].min()
    for back in (75, 55, 35, 15):           # preseason / early-listing sweeps
        stamps.append(season_start - timedelta(days=back) + timedelta(hours=12))

    for _, grp in sched.groupby("week"):
        first_kick = grp["gameday"].min()
        stamps.append(first_kick - timedelta(days=3) + timedelta(hours=12))
        stamps.append(first_kick - timedelta(days=1) + timedelta(hours=12))
        stamps.append(first_kick + timedelta(days=3) + timedelta(hours=16))

    now = pd.Timestamp.now(tz="UTC")
    out = []
    for t in sorted(set(stamps)):
        ts = pd.Timestamp(t)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if EARLIEST_SUPPORTED <= ts <= now:
            out.append(ts)
    return out


def fetch_snapshot(ts: pd.Timestamp) -> tuple:
    """One historical snapshot -> (rows, credits_used, remaining)."""
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "bookmakers": BOOKMAKER,
        "date": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        r = requests.get(HIST_URL, params=params, timeout=30)
    except Exception as exc:
        print(f"    {ts:%Y-%m-%d %H:%M}  request failed: {exc}")
        return [], 0, None

    if r.status_code != 200:
        print(f"    {ts:%Y-%m-%d %H:%M}  HTTP {r.status_code}: {r.text[:120]}")
        return [], 0, r.headers.get("x-requests-remaining")

    payload = r.json()
    snap_ts = payload.get("timestamp")
    rows = []
    for game in payload.get("data", []):
        home = TEAM_NAME_TO_ABBR.get(game.get("home_team"))
        away = TEAM_NAME_TO_ABBR.get(game.get("away_team"))
        if not home or not away:
            continue
        spread_home = total = None
        for bk in game.get("bookmakers", []):
            if bk.get("key") != BOOKMAKER:
                continue
            for mkt in bk.get("markets", []):
                if mkt["key"] == "spreads":
                    for o in mkt.get("outcomes", []):
                        if TEAM_NAME_TO_ABBR.get(o.get("name")) == home:
                            spread_home = o.get("point")
                elif mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if o.get("name") == "Over":
                            total = o.get("point")
        if spread_home is None and total is None:
            continue
        rows.append({
            "snapshot_at": snap_ts,
            "requested_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "game_id": game.get("id"),
            "home_team": home,
            "away_team": away,
            "commence_time": game.get("commence_time"),
            "spread_home": spread_home,
            "total": total,
            "book": BOOKMAKER,
        })

    used = int(r.headers.get("x-requests-last", 0) or 0)
    return rows, used, r.headers.get("x-requests-remaining")


def fetch_season(season: int, dry_run: bool = False) -> pd.DataFrame:
    out_path = DATA_DIR / f"historical_lines_{season}.parquet"
    stamps = snapshot_schedule(season)

    existing = pd.DataFrame()
    done = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(existing["requested_at"].unique())

    todo = [t for t in stamps if t.strftime("%Y-%m-%dT%H:%M:%SZ") not in done]
    print(f"\n=== {season} ===")
    print(f"  {len(stamps)} snapshots planned, {len(done)} already cached, "
          f"{len(todo)} to fetch  (~{len(todo) * 20} credits at 2 markets)")
    if dry_run or not todo:
        return existing

    collected, remaining = [], None
    for i, ts in enumerate(todo, 1):
        rows, used, remaining = fetch_snapshot(ts)
        collected.extend(rows)
        if i % 10 == 0 or i == len(todo):
            print(f"    [{i}/{len(todo)}] {ts:%Y-%m-%d}  rows so far {len(collected)}  "
                  f"credits remaining {remaining}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(collected)
    if not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
    if not df.empty:
        df = df.drop_duplicates(subset=["requested_at", "game_id"], keep="last")
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path.name}")
    return df


def derive_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the snapshot series to one row per game.

    Opening  = earliest snapshot carrying a line.
    Closing  = latest snapshot strictly BEFORE kickoff (never after, so a
               post-kickoff artifact can't masquerade as the close).

    Keyed on (home, away, kickoff DATE) rather than game_id, because The Odds
    API's event IDs are NOT stable for a given game: a matchup gets one id
    while it sits in the season-long listing and a different one once it
    becomes the active weekly event. Grouping by id splits a single game's
    history into fragments -- in 2023 that affected 190 of 283 matchups, and
    produced an "opening" line taken from November and a "closing" line taken
    from September for the same game.

    Kickoff date (not exact timestamp) absorbs flex scheduling moving a game
    between time slots on the same day.
    """
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["snapshot_at"] = pd.to_datetime(d["snapshot_at"], utc=True, errors="coerce")
    d["commence_time"] = pd.to_datetime(d["commence_time"], utc=True, errors="coerce")
    d = d.dropna(subset=["snapshot_at", "commence_time"])
    d = d[d["spread_home"].notna()].sort_values("snapshot_at")
    d["kick_date"] = d["commence_time"].dt.strftime("%Y-%m-%d")

    rows = []
    for (home, away, kick), g in d.groupby(["home_team", "away_team", "kick_date"]):
        # Flex can move kickoff; trust the latest reported time for this game.
        kickoff = g["commence_time"].max()
        pre = g[g["snapshot_at"] < kickoff]
        if pre.empty:
            continue
        first, last = pre.iloc[0], pre.iloc[-1]

        # The earliest observation is usually a LOOKAHEAD line posted months
        # out (median ~134 days before kickoff), not the number a bettor sees
        # when that week's board goes up. Capture the weekly opener separately:
        # the first snapshot inside WEEKLY_OPEN_WINDOW_DAYS of kickoff. That is
        # the actionable "beat the opener" price; the lookahead line is context.
        days_out = (kickoff - pre["snapshot_at"]).dt.total_seconds() / 86400
        weekly = pre[days_out <= WEEKLY_OPEN_WINDOW_DAYS]
        wk_first = weekly.iloc[0] if len(weekly) else None

        rows.append({
            "home_team": home,
            "away_team": away,
            "kick_date": kick,
            "commence_time": kickoff,
            "opened_at": first["snapshot_at"],
            "opening_spread_home": first["spread_home"],
            "opening_total": first["total"],
            "week_opened_at": wk_first["snapshot_at"] if wk_first is not None else pd.NaT,
            "week_open_spread_home": wk_first["spread_home"] if wk_first is not None else np.nan,
            "week_open_total": wk_first["total"] if wk_first is not None else np.nan,
            "closed_at": last["snapshot_at"],
            "closing_spread_home": last["spread_home"],
            "closing_total": last["total"],
            "n_snapshots": len(pre),
            "n_snapshots_week": len(weekly),
            "n_event_ids": g["game_id"].nunique(),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Lookahead-to-close drift (context) vs weekly-open-to-close (actionable).
        out["spread_movement"] = out["closing_spread_home"] - out["opening_spread_home"]
        out["total_movement"] = out["closing_total"] - out["opening_total"]
        out["week_spread_movement"] = out["closing_spread_home"] - out["week_open_spread_home"]
        out["week_total_movement"] = out["closing_total"] - out["week_open_total"]
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print("usage: fetch_historical_lines.py SEASON [END_SEASON] [--dry-run]")
        return
    seasons = list(range(int(args[0]), int(args[-1]) + 1))

    frames = []
    for s in seasons:
        frames.append(fetch_season(s, dry_run=dry))
    if dry:
        return

    allrows = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()
    if allrows.empty:
        print("\nNo rows collected.")
        return

    combined = DATA_DIR / "historical_lines_all.parquet"
    allrows.to_parquet(combined, index=False)
    print(f"\nAll snapshots: {len(allrows)} rows -> {combined.name}")

    oc = derive_open_close(allrows)
    oc_path = DATA_DIR / "historical_open_close.parquet"
    oc.to_parquet(oc_path, index=False)
    print(f"Derived open/close for {len(oc)} games -> {oc_path.name}")
    if not oc.empty:
        moved = oc["spread_movement"].abs()
        print(f"  spread movement: mean |{moved.mean():.2f}| pts, "
              f"{(moved >= 1).mean()*100:.0f}% moved 1+ pt, "
              f"{(moved == 0).mean()*100:.0f}% never moved")


if __name__ == "__main__":
    main()
