"""
Backfill the FULL bookmaker panel at each weekly-opener snapshot.

Why this exists
---------------
Every line we have is DraftKings. That was fine for defining an opener, but it
throws away the one thing a panel of books tells you that a single book cannot:
how much the market DISAGREES with itself, and which way the outlier sits.

That matters because of what has and has not worked here. Six of nine football
feature ideas failed -- the closing line already prices public information about
football. The movement model worked because it predicts PEOPLE, not games.
Cross-book dispersion is more of the thing that worked: it is market structure,
not football information. When one book hangs -3 and the field is -3.5, the
outlier converges, and that convergence is mechanical rather than a forecast.

What this deliberately does NOT do
----------------------------------
It does not redefine the opener as the best available price.

  1. Best-of-N is biased upward. Given N books quoting the same true line plus
     noise, the max of N beats the true price by construction. Grading a
     best-of-N opener against a consensus close would manufacture CLV out of
     that bias and it would look like a real result -- the same shape of error
     as the sign bug and the margin/movement contamination.
  2. DK stays the spine so every previously validated number remains
     comparable. The panel arrives as additional columns, never as a
     redefinition.

Best-available IS recorded, as its own column, so line shopping can be
MEASURED rather than assumed.

Cost
----
The historical endpoint bills 10 credits x [markets] x [regions]. Spreads only
across us+eu is 20 credits per snapshot, ~49 snapshots for a season. Timestamps
are taken from the week_opened_at values already derived in
historical_open_close.parquet, so the panel lines up exactly with the openers
the model was trained against.

Usage:
    python fetch_multibook_lines.py 2024 --dry-run   # plan + credit cost
    python fetch_multibook_lines.py 2024             # pilot one season
    python fetch_multibook_lines.py 2020 2025        # inclusive range
"""

import sys
import time
import warnings
import pandas as pd
import requests

warnings.filterwarnings("ignore")

from config import DATA_DIR, ODDS_API_KEY
from fetch_historical_lines import TEAM_NAME_TO_ABBR, HIST_URL

MARKETS = "spreads"
REGIONS = "us,eu"
CREDITS_PER_CALL = 10 * len(MARKETS.split(",")) * len(REGIONS.split(","))
SLEEP_BETWEEN_CALLS = 1.0


def snapshot_times(season: int) -> list:
    """The exact instants our weekly openers were read from.

    Reusing week_opened_at rather than recomputing a schedule guarantees the
    panel is measured at the same moment as the DK opener it will be compared
    to. An offset here would show up later as phantom dispersion.
    """
    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    ct = pd.to_datetime(oc["commence_time"], utc=True)
    oc = oc[(ct.dt.year.where(ct.dt.month >= 3, ct.dt.year - 1)) == season]
    wo = pd.to_datetime(oc["week_opened_at"], utc=True).dropna()
    return sorted(wo.unique())


def fetch_snapshot(ts) -> tuple:
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "date": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        r = requests.get(HIST_URL, params=params, timeout=30)
    except Exception as exc:
        print(f"    {pd.Timestamp(ts):%Y-%m-%d %H:%M}  request failed: {exc}")
        return [], None
    if r.status_code != 200:
        print(f"    {pd.Timestamp(ts):%Y-%m-%d %H:%M}  HTTP {r.status_code}: {r.text[:120]}")
        return [], r.headers.get("x-requests-remaining")

    payload = r.json()
    rows = []
    for game in payload.get("data", []):
        home = TEAM_NAME_TO_ABBR.get(game.get("home_team"))
        away = TEAM_NAME_TO_ABBR.get(game.get("away_team"))
        if not home or not away:
            continue
        for bk in game.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "spreads":
                    continue
                for oc_ in mk.get("outcomes", []):
                    if TEAM_NAME_TO_ABBR.get(oc_.get("name")) != home:
                        continue
                    rows.append({
                        "snapshot_at": payload.get("timestamp"),
                        "requested_at": params["date"],
                        "home_team": home,
                        "away_team": away,
                        "commence_time": game.get("commence_time"),
                        "book": bk.get("key"),
                        # Standard convention: negative = home favored. The Odds
                        # API already quotes it this way.
                        "spread_home": oc_.get("point"),
                        "price_home": oc_.get("price"),
                    })
    return rows, r.headers.get("x-requests-remaining")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print("usage: python fetch_multibook_lines.py <season> [end_season] [--dry-run]")
        return
    lo = int(args[0])
    hi = int(args[1]) if len(args) > 1 else lo

    for season in range(lo, hi + 1):
        stamps = snapshot_times(season)
        cost = len(stamps) * CREDITS_PER_CALL
        print(f"\n{season}: {len(stamps)} snapshots x {CREDITS_PER_CALL} credits = {cost}")
        if dry:
            continue

        out = DATA_DIR / f"multibook_{season}.parquet"
        # Always read progress from disk. A previous backfill in this project
        # gated its checkpoint on an in-memory flag and each save silently
        # erased the last one.
        done = set()
        if out.exists():
            prev = pd.read_parquet(out)
            done = set(prev["requested_at"].unique())
            print(f"  resuming: {len(done)} snapshots already cached")

        for i, ts in enumerate(stamps, 1):
            key = pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
            if key in done:
                continue
            rows, remaining = fetch_snapshot(ts)
            if rows:
                df = pd.DataFrame(rows)
                if out.exists():
                    df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
                df.to_parquet(out, index=False)
                books = df[df.requested_at == key]["book"].nunique()
                print(f"  [{i}/{len(stamps)}] {key}  {len(rows):>4} quotes, "
                      f"{books:>2} books  (credits left {remaining})")
            else:
                print(f"  [{i}/{len(stamps)}] {key}  no data")
            time.sleep(SLEEP_BETWEEN_CALLS)

        if out.exists():
            fin = pd.read_parquet(out)
            print(f"  {season} done: {len(fin)} quotes, {fin['book'].nunique()} books, "
                  f"{fin['requested_at'].nunique()} snapshots -> {out.name}")


if __name__ == "__main__":
    main()
