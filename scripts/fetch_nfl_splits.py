"""
NFL public betting splits from Action Network.

Mirror of fetch_cfb_splits.py; the parsing and both of its constraints live in
action_splits.py. Read that module's docstring before changing anything here --
in particular, this cannot run on GitHub Actions, and the per-book breakout is
not real so only the Consensus figure is stored.

Team abbreviations line up with ours already, with exactly one exception found
by diffing the two sets: Action says JAC, we say JAX. All 31 others match.

Usage:
    python fetch_nfl_splits.py
    python fetch_nfl_splits.py --dry-run
"""

import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from action_splits import fetch, outcomes, kickoff, CONSENSUS_BOOK
from score_week import supabase

# The only abbreviation Action Network spells differently to us.
FIX = {"JAC": "JAX"}


def abbr(team):
    a = team.get("abbr")
    return FIX.get(a, a)


def main():
    dry = "--dry-run" in sys.argv

    games, books, observed, age = fetch("nfl")
    print(f"Action Network NFL: {len(games)} games, page {age}s old "
          f"(generated ~{observed.isoformat()[:19]}Z)")

    board = (supabase.table("line_predictions")
             .select("game_id, bet_type, home_team, away_team, commence_time")
             .execute().data or [])
    by_key = {}
    for b in board:
        by_key.setdefault((b["home_team"], b["away_team"], b["bet_type"]), []).append(b)
    print(f"board: {len({b['game_id'] for b in board})} games")

    rows, matched, unmatched = [], set(), []
    for g in games:
        tm = {t["id"]: t for t in g["teams"]}
        home = abbr(tm.get(g.get("home_team_id")) or {})
        away = abbr(tm.get(g.get("away_team_id")) or {})
        got = outcomes(g, CONSENSUS_BOOK)
        if not got:
            continue
        kick = kickoff(g)
        for market, vals in got.items():
            cands = by_key.get((home, away, market))
            if not cands:
                unmatched.append(f"{away} @ {home} {market}")
                continue
            hit = [c for c in cands
                   if abs((datetime.fromisoformat(c["commence_time"]).date() - kick).days) <= 1]
            if not hit:
                unmatched.append(f"{away} @ {home} {market} (date)")
                continue
            assert len(hit) == 1, f"{away} @ {home} {market} matched {len(hit)} board rows"
            rows.append({"game_id": hit[0]["game_id"], "bet_type": market,
                         "captured_at": observed.isoformat(),
                         "source": f"action_network:{CONSENSUS_BOOK}",
                         "note": books.get(CONSENSUS_BOOK), **vals})
            matched.add(hit[0]["game_id"])

    print(f"  {len(matched)} board games matched, {len(rows)} split rows")
    for u in unmatched[:6]:
        print(f"    unmatched: {u}")
    if not rows:
        print("  nothing to write")
        return

    if dry:
        for r in rows[:8]:
            print(f"    {r['game_id'][:10]} {r['bet_type']:6} "
                  f"tix {r['away_bets_pct']}/{r['home_bets_pct']}  "
                  f"money {r['away_money_pct']}/{r['home_money_pct']}")
        print("  --dry-run: nothing written")
        return

    supabase.table("nfl_public_splits").upsert(
        rows, on_conflict="game_id,bet_type,source,captured_at").execute()
    print(f"  wrote {len(rows)} split rows")


if __name__ == "__main__":
    main()
