"""
Does Action Network answer THIS machine, and with real numbers?

Temporary probe. action_splits.py has carried a measured warning since
2026-09-06 that the public-betting page returns no __NEXT_DATA__ to a GitHub
Actions runner, and the whole splits pipeline was moved to a home machine on
that basis. A separate project reports the API path works from Vercel. This
runs our own parser, wherever it is pointed, and reports what came back.

The distinction that matters: a payload can arrive with bet_info present but
every percentage zero. outcomes() already prefers the populated row over the
(0, 0) placeholder, so this counts markets with a NON-ZERO split -- a job that
cheerfully writes zeros would be worse than no job at all.

Usage:
    python scripts/probe_action_egress.py           # both leagues, both paths
    python scripts/probe_action_egress.py --week 6
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from action_splits import fetch, fetch_week, outcomes, CONSENSUS_BOOK
from config import CURRENT_SEASON as SEASON


def summarise(label, games, observed, age):
    real = 0
    total = 0
    sample = None
    for g in games:
        got = outcomes(g, CONSENSUS_BOOK)
        for market, v in got.items():
            total += 1
            if (v["home_bets_pct"] or 0) > 0 or (v["away_bets_pct"] or 0) > 0:
                real += 1
                if sample is None:
                    tm = {t["id"]: t.get("abbr") for t in g.get("teams") or []}
                    sample = (f"{tm.get(g.get('away_team_id'))}@"
                              f"{tm.get(g.get('home_team_id'))} {market} "
                              f"tickets {v['away_bets_pct']}/{v['home_bets_pct']} "
                              f"money {v['away_money_pct']}/{v['home_money_pct']}")
    print(f"  {label:26} {len(games):3d} games, {total:3d} markets, "
          f"{real:3d} with a real split  (age {age}s)")
    if sample:
        print(f"      e.g. {sample}")
    return real


def main():
    week = None
    if "--week" in sys.argv:
        week = int(sys.argv[sys.argv.index("--week") + 1])

    ok = {}
    for league in ("nfl", "ncaaf"):
        print(f"{league}:")
        try:
            g, _, obs, age = fetch(league)
            ok[f"{league} page"] = summarise("page (__NEXT_DATA__)", g, obs, age)
        except Exception as e:
            print(f"  page                       FAILED: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:110]}")
            ok[f"{league} page"] = 0
        try:
            w = week if week else (6 if league == "nfl" else 7)
            g, _, obs, age = fetch_week(league, w, SEASON)
            ok[f"{league} api"] = summarise(f"api (week {w})", g, obs, age)
        except Exception as e:
            print(f"  api                        FAILED: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:110]}")
            ok[f"{league} api"] = 0

    print("\nverdict from this machine:")
    for k, v in ok.items():
        print(f"  {k:12} {'USABLE' if v else 'no usable splits'}  ({v} markets)")
    if not any(ok.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
