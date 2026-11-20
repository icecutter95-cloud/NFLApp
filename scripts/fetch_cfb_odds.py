"""
Pull current college football lines into cfb_line_history, one row per book.

The NFL side gets its lines from the refresh-odds edge function on a pg_cron
schedule. College deliberately does NOT reuse that function: refresh-odds feeds
line_history, which feeds line_open_close, which is where every validated NFL
number comes from. Adding a second sport to that path would put the one thing
with a demonstrated edge in the blast radius of college plumbing.

So this is a separate script writing to separate tables. Same DraftKings anchor,
same conventions.

DRAFTKINGS IS THE OFFICIAL MEASURE AND NOTHING HERE CHANGES THAT
Until 2026-11-20 this pulled DraftKings alone, and every graded college number
-- cfb_open_close, cfb_tracking, cfb_signal, the rule-1 movement gate -- is
measured on it. The other books are stored ALONGSIDE, to answer two questions
in a few weeks that cannot be asked retrospectively because the data was never
kept:

  1. does movement measured at BetOnline predict better than at DraftKings?
  2. are games where DraftKings LAGS BetOnline an edge in their own right?

Both need a stored history of the two moving against each other, so collection
starts now and the analysis waits for sample. Nothing reads the non-anchor rows
yet.

(An earlier version of this docstring claimed "same 9-book panel for shopping".
That was never true -- the code pulled one book -- and is the same class of
mistake as the workflow comments corrected in .github/workflows/README.md.)

EVERY READER OF cfb_line_history MUST FILTER ON book
The table held one book for its whole life, so nothing filtered. cfb_open_close
took the opener as the first row by recorded_at, and Postgres now() is
transaction time, so the four books written by one INSERT share an identical
recorded_at: without a filter the opener and close become an arbitrary book and
n_snapshots quadruples, which would have silently rewritten clv_points, result
and direction_correct for the whole college record. Guarded in three places
before this script changed -- the cfb_open_close view (migration
cfb_open_close_anchor_on_draftkings, verified a byte-identical no-op against
693 rows) and both reads in log_cfb_predictions.py.

Cost: UNCHANGED at 2 credits. The endpoint charges per market per region, and
the bookmakers parameter counts each 10 books as one region, so four books cost
exactly what one did. Measured 2026-11-20 from the x-requests-last header: 2
for draftkings alone, 2 for all four.

Coverage measured the same day, of 80 NCAAF games on the board:
    draftkings 75, fanduel 78, betonlineag 64, lowvig 62
FanDuel carries MORE games than the anchor, so a game can have a non-anchor
line and no DraftKings line at all. That is why rows are per book rather than
one row with four columns -- a missing book is an absent row, not a zero.

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

# THE official measure. Imported by log_cfb_predictions.py rather than restated
# there, so the anchor has one definition.
ANCHOR = "draftkings"

# Stored for comparison only. lowvig is BetOnline's sister book and prices
# close to it; both are kept because a disagreement between them is a useful
# control on a disagreement between BetOnline and DraftKings. Keep this at 10
# or fewer books or the credit cost per call goes up.
PANEL = [ANCHOR, "fanduel", "betonlineag", "lowvig"]


def book_line(bk, home_raw):
    """(spread_home, total) for one bookmaker block, or None if it has neither.

    spread_home is signed from the home side, matching the rest of the project:
    negative means the home team is favoured.
    """
    spread = total = None
    for mk in bk.get("markets", []):
        if mk.get("key") == "spreads":
            for oc in mk.get("outcomes", []):
                if oc.get("name") == home_raw:
                    spread = oc.get("point")
        elif mk.get("key") == "totals":
            for oc in mk.get("outcomes", []):
                if oc.get("name") == "Over":
                    total = oc.get("point")
    if spread is None and total is None:
        return None
    return spread, total


def main():
    dry = "--dry-run" in sys.argv
    r = requests.get(URL, params={
        "apiKey": ODDS_API_KEY, "regions": "us", "markets": "spreads,totals",
        "oddsFormat": "american", "bookmakers": ",".join(PANEL),
    }, timeout=40)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        return

    games = r.json()
    rows, skipped, unmapped = [], 0, set()
    per_book = {b: 0 for b in PANEL}
    anchor_rows, fbs_games = [], 0
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

        got = {}
        for bk in g.get("bookmakers", []):
            key = bk.get("key")
            if key not in per_book:
                continue  # not in the panel; ignore rather than store untagged
            line = book_line(bk, home_raw)
            if line is not None:
                got[key] = line
        if not got:
            continue

        fbs_games += 1
        for key, (spread, total) in got.items():
            per_book[key] += 1
            row = {
                "game_id": g["id"],
                "home_team": to_key(home_raw), "away_team": to_key(away_raw),
                "commence_time": g.get("commence_time"),
                "spread_home": spread, "total": total, "book": key,
            }
            rows.append(row)
            if key == ANCHOR:
                anchor_rows.append(row)

    print(f"{len(games)} NCAAF games on the board | {fbs_games} FBS-vs-FBS "
          f"| {skipped} skipped (non-FBS)")
    print("  by book: " + ", ".join(
        f"{b}{' (anchor)' if b == ANCHOR else ''} {per_book[b]}" for b in PANEL))
    if not per_book[ANCHOR]:
        # Without the anchor nothing downstream can grade, so this is louder
        # than a missing comparison book.
        print(f"  WARNING: no {ANCHOR} lines in this pull")
    if unmapped:
        # Non-FBS names dominate this list; an FBS school appearing here is a
        # crosswalk gap worth fixing.
        print(f"  note: {len(unmapped)} unmapped names, e.g. {sorted(unmapped)[:4]}")
    for r_ in anchor_rows[:8]:
        print(f"  {r_['away_team']:>16} @ {r_['home_team']:<16} "
              f"spread {r_['spread_home']}  total {r_['total']}")

    if dry:
        print(f"  --dry-run: nothing written ({len(rows)} rows prepared)")
        return
    if rows:
        # Append-only: the first snapshot of a game is its opener and must never
        # be overwritten, which is the one number CLV cannot reconstruct later.
        # Chunked because one pull is now ~280 rows, not ~70.
        for i in range(0, len(rows), 500):
            supabase.table("cfb_line_history").insert(rows[i:i + 500]).execute()
        print(f"  wrote {len(rows)} rows to cfb_line_history "
              f"({per_book[ANCHOR]} {ANCHOR})")


if __name__ == "__main__":
    main()
