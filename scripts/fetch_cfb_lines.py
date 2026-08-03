"""
Backfill college football opening and closing lines, 2020-2025.

This is the go/no-go for the whole CFB effort. The NFL app's one working idea is
that the weekly OPENER is soft even though the close is not -- predict where the
number travels, take the early price, collect CLV. That thesis has to be
re-tested on college football before anything else gets built, because if CFB
openers are efficient there is nothing here worth modelling and no amount of
feature work will rescue it.

Crucially, the first test needs no football data at all. Baseline B in
model_line_movement.py is opener-only. So this backfill alone is enough to learn
whether the approach travels, for a few thousand credits and no CFBD
integration.

Snapshot design
---------------
Three per week, chosen around how the college card actually behaves:
  Mon 12:00 UTC  the week's board posts Sunday night/Monday -> the OPENER
  Sat 16:00 UTC  noon ET, the close for early kickoffs
  Sat 22:00 UTC  6pm ET, the close for afternoon and night games

Open and close are then derived per game from its own commence_time, exactly as
the NFL backfill does, rather than assuming any one snapshot is "the" close.

Cost is per CALL, not per game: 10 credits x 1 market x 1 region. One call
returns the entire slate, so ~75 college games cost the same as 16 NFL ones.
Spreads only for now -- totals are a second market and double the bill, and they
are worth nothing until spreads clear the bar.

Written to data/cfb_lines_*.parquet. Touches no NFL file or table.

Usage:
    python fetch_cfb_lines.py 2024 --dry-run   # plan and credit cost
    python fetch_cfb_lines.py 2024             # one season
    python fetch_cfb_lines.py 2020 2025        # inclusive range
"""

import sys
import time
import warnings
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

from config import DATA_DIR, ODDS_API_KEY
from cfb_teams import to_key, is_fbs

HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_ncaaf/odds"
MARKETS = "spreads"
REGIONS = "us"
BOOKMAKER = "draftkings"
CREDITS_PER_CALL = 10 * len(MARKETS.split(",")) * len(REGIONS.split(","))
SLEEP = 1.0
EARLIEST = pd.Timestamp("2020-06-01T00:00:00Z")
# Same definition the NFL side uses: snapshots inside this window represent the
# actionable weekly opener, not a lookahead number posted months out.
WEEKLY_OPEN_WINDOW_DAYS = 10


def season_stamps(season: int) -> list:
    """Mon/Sat/Sat snapshots from late August through the bowls."""
    start = pd.Timestamp(f"{season}-08-20T00:00:00Z")
    end = pd.Timestamp(f"{season + 1}-01-15T00:00:00Z")
    now = pd.Timestamp.now(tz="UTC")

    stamps, d = [], start
    while d <= end:
        if d.dayofweek == 6:                      # Sunday
            # Added after measuring that the Monday-only opener sat a median
            # 129h before kickoff against the NFL's 169h -- we were reading
            # college lines 40 hours later in their life than NFL ones, which
            # is most of the window where CLV is actually earned.
            stamps.append(d + timedelta(hours=2))    # Sat ~9pm ET, board posts
            stamps.append(d + timedelta(hours=18))   # Sun ~1pm ET
        elif d.dayofweek == 0:                    # Monday
            stamps.append(d + timedelta(hours=12))
        elif d.dayofweek == 5:                    # Saturday
            stamps.append(d + timedelta(hours=16))
            stamps.append(d + timedelta(hours=22))
        d += timedelta(days=1)
    return [s for s in stamps if EARLIEST <= s <= now]


def fetch(ts) -> tuple:
    params = {"apiKey": ODDS_API_KEY, "regions": REGIONS, "markets": MARKETS,
              "bookmakers": BOOKMAKER,
              "date": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        r = requests.get(HIST_URL, params=params, timeout=40)
    except Exception as exc:
        print(f"    {ts:%Y-%m-%d %H:%M}  request failed: {exc}")
        return [], None
    if r.status_code != 200:
        print(f"    {ts:%Y-%m-%d %H:%M}  HTTP {r.status_code}: {r.text[:100]}")
        return [], r.headers.get("x-requests-remaining")

    payload = r.json()
    rows, skipped_non_fbs = [], 0
    for g in payload.get("data", []):
        home_raw, away_raw = g.get("home_team"), g.get("away_team")
        # FBS-vs-FBS only. Roughly a hundred games a year are against FCS
        # opponents with no usable stats; they are dropped explicitly here
        # rather than zero-filled into a feature matrix later.
        if not (is_fbs(home_raw) and is_fbs(away_raw)):
            skipped_non_fbs += 1
            continue
        spread = None
        for bk in g.get("bookmakers", []):
            if bk.get("key") != BOOKMAKER:
                continue
            for mk in bk.get("markets", []):
                if mk.get("key") != "spreads":
                    continue
                for oc in mk.get("outcomes", []):
                    if oc.get("name") == home_raw:
                        spread = oc.get("point")
        if spread is None:
            continue
        rows.append({
            "snapshot_at": payload.get("timestamp"),
            "requested_at": params["date"],
            "home_team": to_key(home_raw), "away_team": to_key(away_raw),
            "commence_time": g.get("commence_time"),
            # The Odds API already quotes the standard convention: negative
            # means the home team is favoured.
            "spread_home": spread,
        })
    return rows, r.headers.get("x-requests-remaining"), skipped_non_fbs


def derive_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Per game: weekly opening line, closing line, and the movement between.

    Keyed on (home, away, kick_date) rather than any event id. The Odds API
    mints unstable ids -- roughly 1.67 per game on the NFL side -- so joining on
    them loses games silently.
    """
    df = df.copy()
    df["snap"] = pd.to_datetime(df["snapshot_at"], utc=True)
    df["kick"] = pd.to_datetime(df["commence_time"], utc=True)
    df["kick_date"] = df["kick"].dt.date
    df = df[df["snap"] < df["kick"]].sort_values("snap")

    out = []
    for (h, a, kd), g in df.groupby(["home_team", "away_team", "kick_date"]):
        kick = g["kick"].iloc[0]
        wk = g[g["snap"] >= kick - timedelta(days=WEEKLY_OPEN_WINDOW_DAYS)]
        if wk.empty or len(g) < 2:
            continue
        out.append({
            "home_team": h, "away_team": a, "kick_date": kd,
            "commence_time": kick,
            "week_opened_at": wk["snap"].iloc[0],
            "week_open_spread_home": float(wk["spread_home"].iloc[0]),
            "closed_at": g["snap"].iloc[-1],
            "closing_spread_home": float(g["spread_home"].iloc[-1]),
            "n_snapshots": len(g), "n_snapshots_week": len(wk),
        })
    o = pd.DataFrame(out)
    if not o.empty:
        o["week_spread_movement"] = o["closing_spread_home"] - o["week_open_spread_home"]
    return o


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print("usage: python fetch_cfb_lines.py <season> [end_season] [--dry-run]")
        return
    lo, hi = int(args[0]), int(args[1]) if len(args) > 1 else int(args[0])

    for season in range(lo, hi + 1):
        stamps = season_stamps(season)
        print(f"\n{season}: {len(stamps)} snapshots x {CREDITS_PER_CALL} = "
              f"{len(stamps) * CREDITS_PER_CALL} credits")
        if dry:
            continue

        out = DATA_DIR / f"cfb_lines_{season}.parquet"
        # Progress is always read from disk. An earlier backfill in this project
        # gated its checkpoint on an in-memory flag and each save wiped the last.
        done = set()
        if out.exists():
            done = set(pd.read_parquet(out)["requested_at"].unique())
            print(f"  resuming: {len(done)} snapshots cached")

        skipped_total = 0
        for i, ts in enumerate(stamps, 1):
            key = pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
            if key in done:
                continue
            rows, remaining, skipped = fetch(ts)
            skipped_total += skipped
            if rows:
                df = pd.DataFrame(rows)
                if out.exists():
                    df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
                df.to_parquet(out, index=False)
            if i % 10 == 0 or rows:
                print(f"  [{i}/{len(stamps)}] {key}  {len(rows):>3} FBS games "
                      f"({skipped} non-FBS dropped)  credits left {remaining}")
            time.sleep(SLEEP)

        if out.exists():
            raw = pd.read_parquet(out)
            oc = derive_open_close(raw)
            oc_path = DATA_DIR / f"cfb_open_close_{season}.parquet"
            oc.to_parquet(oc_path, index=False)
            print(f"  {season}: {len(raw)} quotes -> {len(oc)} games with open+close")
            if not oc.empty:
                print(f"    mean |movement| {oc.week_spread_movement.abs().mean():.2f} pts, "
                      f"moved at all {(oc.week_spread_movement.abs() > 0.01).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
