"""
Public betting splits for college games -- fully automated.

This supersedes the hand-capture path in record_cfb_splits.py, which was written
on the belief that these numbers could not be fetched. That was wrong twice
over: DraftKings publishes Bets%/Handle% on its own site, and Action Network's
public-betting page ships the whole thing as JSON inside __NEXT_DATA__ -- per
BOOK and per MARKET, to an unauthenticated request. No login, no browser, no
Cloudflare challenge. record_cfb_splits.py stays for manual corrections and for
sources this cannot reach.

What comes back
---------------
Roughly 99 games, each with a markets dict keyed by book id, each holding
"spread" (home/away) and "total" (over/under) outcomes carrying:

    bet_info.tickets.percent    share of wagers
    bet_info.money.percent      share of dollars

Book ids, resolved from the page's own allBooks map:

    15  Consensus       68  DraftKings      69  FanDuel     71  BetRivers
    75  BetMGM          79  bet365        2988  Fanatics    123  Caesars

The per-book breakout is NOT real on this payload
-------------------------------------------------
Measured across the live board: 194 game-markets where every book id carries
IDENTICAL percentages, against 4 where they differ at all -- and most of those
four are duplicate-row artifacts rather than genuine divergence. Action Network
appears to serve one consensus figure under all eight labels here; the real
book-specific numbers are presumably what Pro gates in its UI.

So treat what this collects as ONE source: Action Network consensus. Only book
15 is stored by default for that reason. Do not present these as DraftKings'
numbers -- DK's own page showed 34% on a side where Action showed 97%, and this
feed does not resolve that, it just reports Action's side of it.

Extreme splits also live in the moneyline (94/6 on a game whose spread was
47/53), so comparing a moneyline figure against a spread table manufactures a
contradiction that was never there. Only spread and total are stored.

Timestamps: the page is CDN-cached, so the age header is subtracted from the
response time to recover roughly when the numbers were generated. captured_at is
the OBSERVATION time, which is what pairs a capture with the line snapshot that
was live beside it.

Where this runs
---------------
In CI, since 2026-10-07. This used to read the public-betting PAGE, which
returns no __NEXT_DATA__ to a GitHub runner, so collection ran from a Windows
scheduled task on a home machine. Re-measured from a runner that day: the page
still fails, and the scoreboard API returns 116 college markets with real
percentages, identical to a home connection. The page was the blocked thing,
not the data.

It now reads the API for the current week and the next one -- fetch_current()
reports which week it answered with, so no CFBD call is spent discovering it
and the Saturday-to-Sunday rollover cannot leave next week's games uncovered.
The scheduled task still works and is kept as a fallback.

Usage:
    python fetch_cfb_splits.py
    python fetch_cfb_splits.py --dry-run
    python fetch_cfb_splits.py --books 68,15     # default: consensus only
"""

import json
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone

import requests

warnings.filterwarnings("ignore")

from action_splits import (fetch_current, fetch_week, outcomes, kickoff,
                           CONSENSUS_BOOK)
from cfb_teams import to_key, cfbd_to_key
from config import CURRENT_SEASON as SEASON
from score_week import supabase

URL = "https://www.actionnetwork.com/ncaaf/public-betting"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")


def team_key(team):
    """Our crosswalk key, or None for the FCS opponents we never predict on."""
    for cand in (team.get("location"), team.get("full_name"),
                 team.get("display_name"), team.get("abbr")):
        if not cand:
            continue
        for fn in (to_key, cfbd_to_key):
            try:
                k = fn(cand)
            except Exception:
                k = None
            if k:
                return k
    return None


# Consensus only by default. Every other book id on this payload repeats the
# same numbers, so storing all eight would be one source wearing eight hats --
# and would make any later "the books disagree" analysis a measurement of
# nothing. Pass --books to override if that ever changes.
DEFAULT_BOOKS = {"15"}


def wanted_books(argv):
    if "--books" not in argv:
        return DEFAULT_BOOKS
    i = argv.index("--books")
    if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
        return set(argv[i + 1].split(","))
    return None


def main():
    dry = "--dry-run" in sys.argv
    want = wanted_books(sys.argv)

    # Current week, then the next one. A college week rolls over on Sunday
    # while next Saturday's markets are already up, so asking only for
    # "current" leaves the games being bet soonest uncovered for a day.
    games, books, observed, age, week = fetch_current('ncaaf', SEASON)
    print(f"Action Network ncaaf week {week}: {len(games)} games, "
          f"{age}s old (generated ~{observed.isoformat()[:19]}Z)")
    if week is not None and "--this-week" not in sys.argv:
        # Only network failures are tolerated here. A bare `except Exception`
        # reported "week 7 unavailable" for a plain unpacking mistake on this
        # very line -- fetch_week returns four values, fetch_current five --
        # which is the shape of every silent-degradation bug in this project.
        try:
            nxt, _, _, nage = fetch_week('ncaaf', week + 1, SEASON)
        except requests.RequestException as e:
            print(f"  week {week + 1} unavailable ({type(e).__name__})")
        else:
            games = games + nxt
            print(f"  + week {week + 1}: {len(nxt)} games ({nage}s old)")

    board = (supabase.table("cfb_predictions")
             .select("game_id, home_team, away_team, commence_time")
             .execute().data or [])
    by_pair = {}
    for g in board:
        by_pair.setdefault((g["home_team"], g["away_team"]), []).append(g)
    print(f"board: {len({g['game_id'] for g in board})} games")

    rows, matched, skipped_fcs = [], set(), 0
    for g in games:
        tm = {t["id"]: t for t in g["teams"]}
        home = team_key(tm.get(g.get("home_team_id")) or {})
        away = team_key(tm.get(g.get("away_team_id")) or {})
        if not home or not away:
            skipped_fcs += 1
            continue
        cands = by_pair.get((home, away))
        if not cands:
            continue
        kick = kickoff(g)
        hit = [c for c in cands
               if abs((datetime.fromisoformat(c["commence_time"]).date() - kick).days) <= 1]
        if not hit:
            continue
        assert len(hit) == 1, f"{away} @ {home} matched {len(hit)} board games"
        gid = hit[0]["game_id"]

        for bid in (want or {CONSENSUS_BOOK}):
            for market, vals in outcomes(g, bid).items():
                rows.append({
                    "game_id": gid, "captured_at": observed.isoformat(),
                    "source": f"action_network:{bid}", "bet_type": market,
                    "note": books.get(bid), **vals,
                })
                matched.add(gid)

    print(f"  {len(matched)} board games matched, {len(rows)} split rows "
          f"({skipped_fcs} non-FBS games skipped)")
    if not rows:
        print("  nothing to write")
        return

    per_book = {}
    for r in rows:
        per_book[r["source"]] = per_book.get(r["source"], 0) + 1
    for src, n in sorted(per_book.items(), key=lambda x: -x[1]):
        print(f"    {src:22} {str(books.get(src.split(':')[1])):12} {n:>4} rows")

    if dry:
        for r in rows[:8]:
            print(f"    {r['game_id'][:8]} {r['bet_type']:6} {str(r['note']):11} "
                  f"tix {r['away_bets_pct']}/{r['home_bets_pct']}  "
                  f"money {r['away_money_pct']}/{r['home_money_pct']}")
        print("  --dry-run: nothing written")
        return

    for i in range(0, len(rows), 500):
        supabase.table("cfb_public_splits").upsert(
            rows[i:i + 500],
            on_conflict="game_id,bet_type,source,captured_at").execute()
    print(f"  wrote {len(rows)} split rows")


if __name__ == "__main__":
    main()
