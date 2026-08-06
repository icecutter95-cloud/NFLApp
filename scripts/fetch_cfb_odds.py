"""
Pull current college football lines into cfb_line_history.

The NFL side gets its lines from the refresh-odds edge function on a pg_cron
schedule. College deliberately does NOT reuse that function: refresh-odds feeds
line_history, which feeds line_open_close, which is where every validated NFL
number comes from. Adding a second sport to that path would put the one thing
with a demonstrated edge in the blast radius of college plumbing.

So this is a separate script writing to separate tables. Same DraftKings anchor,
same 9-book panel for shopping, same conventions.

Cost: 1 credit per market per region on the current-odds endpoint, and one call
returns the whole slate.

Usage:
    python fetch_cfb_odds.py
    python fetch_cfb_odds.py --dry-run
"""

import sys
import warnings

import requests

warnings.filterwarnings("ignore")

from config import ODDS_API_KEY
from cfb_teams import to_key, is_fbs
from score_week import supabase

URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
BOOKMAKER = "draftkings"


def main():
    dry = "--dry-run" in sys.argv
    r = requests.get(URL, params={
        "apiKey": ODDS_API_KEY, "regions": "us", "markets": "spreads,totals",
        "oddsFormat": "american", "bookmakers": BOOKMAKER,
    }, timeout=40)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        return

    games = r.json()
    rows, skipped, unmapped = [], 0, set()
    for g in games:
        home_raw, away_raw = g.get("home_team"), g.get("away_team")
        # FBS-vs-FBS only, matching how the models were trained. An unmapped
        # name is recorded rather than silently dropped -- that is how the Rams
        # went missing for weeks on the NFL side.
        if not (is_fbs(home_raw) and is_fbs(away_raw)):
            skipped += 1
            if to_key(home_raw) is None and home_raw:
                unmapped.add(home_raw)
            if to_key(away_raw) is None and away_raw:
                unmapped.add(away_raw)
            continue

        spread = total = None
        for bk in g.get("bookmakers", []):
            if bk.get("key") != BOOKMAKER:
                continue
            for mk in bk.get("markets", []):
                if mk["key"] == "spreads":
                    for oc in mk["outcomes"]:
                        if oc.get("name") == home_raw:
                            spread = oc.get("point")
                elif mk["key"] == "totals":
                    for oc in mk["outcomes"]:
                        if oc.get("name") == "Over":
                            total = oc.get("point")
        if spread is None and total is None:
            continue

        rows.append({
            "game_id": g["id"],
            "home_team": to_key(home_raw), "away_team": to_key(away_raw),
            "commence_time": g.get("commence_time"),
            "spread_home": spread, "total": total, "book": BOOKMAKER,
        })

    print(f"{len(games)} NCAAF games on the board | {len(rows)} FBS-vs-FBS "
          f"| {skipped} skipped (non-FBS)")
    if unmapped:
        # Non-FBS names dominate this list; an FBS school appearing here is a
        # crosswalk gap worth fixing.
        print(f"  note: {len(unmapped)} unmapped names, e.g. {sorted(unmapped)[:4]}")
    for r_ in rows[:8]:
        print(f"  {r_['away_team']:>16} @ {r_['home_team']:<16} "
              f"spread {r_['spread_home']}  total {r_['total']}")

    if dry:
        print("  --dry-run: nothing written")
        return
    if rows:
        # Append-only: the first snapshot of a game is its opener and must never
        # be overwritten, which is the one number CLV cannot reconstruct later.
        supabase.table("cfb_line_history").insert(rows).execute()
        print(f"  wrote {len(rows)} rows to cfb_line_history")


if __name__ == "__main__":
    main()
