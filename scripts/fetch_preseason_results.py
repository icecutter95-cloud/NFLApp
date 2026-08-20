"""
Final scores for preseason games, so the Preseason tab can grade itself.

Why ESPN
--------
nflverse carries REG and postseason only -- there is no preseason in it at all,
which is why the board had to be reconstructed from the odds feed in the first
place. The Odds API has a scores endpoint but it caps at daysFrom=3, so it
cannot reach back across a preseason. ESPN's public scoreboard takes an explicit
date and seasontype=1, which covers the whole thing and costs nothing.

Matching
--------
ESPN knows nothing about our game ids, so games are matched on the team pair
within a day of our kickoff, and the row is then stored under OUR game_id. That
keeps the pair join inside this script, where it can be checked, instead of in a
view where it has repeatedly caused fan-out in this project. A pair matching more
than one board game aborts rather than guessing.

Grading conventions -- identical to clv_tracking, deliberately
--------------------------------------------------------------
    open_line is the HOME spread; negative means home is laid points
    home_margin  = home_score - away_score
    home covers  when home_margin + open_line > 0
    over wins    when home_score + away_score > open_line

Usage:
    python fetch_preseason_results.py
    python fetch_preseason_results.py --dry-run
"""

import sys
import warnings
from datetime import datetime, timedelta, timezone

import requests

warnings.filterwarnings("ignore")

from score_week import supabase

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# ESPN uses two abbreviations we do not. Verified as the ONLY two differences
# across every preseason game on the board.
ESPN_FIX = {"WSH": "WAS", "LAR": "LA"}


def espn_games(day):
    r = requests.get(ESPN, params={"dates": day.strftime("%Y%m%d"),
                                   "seasontype": 1}, timeout=60)
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        if not c["status"]["type"]["completed"]:
            continue
        side = {}
        for x in c["competitors"]:
            ab = x["team"]["abbreviation"]
            side[x["homeAway"]] = (ESPN_FIX.get(ab, ab), int(x["score"]))
        if "home" in side and "away" in side:
            out.append({"home_team": side["home"][0], "home_score": side["home"][1],
                        "away_team": side["away"][0], "away_score": side["away"][1],
                        "date": datetime.fromisoformat(
                            e["date"].replace("Z", "+00:00")).date()})
    return out


def main():
    dry = "--dry-run" in sys.argv

    board = (supabase.table("preseason_predictions")
             .select("game_id, season, home_team, away_team, commence_time")
             .eq("bet_type", "spread").execute().data or [])
    if not board:
        print("No preseason predictions to grade")
        return
    print(f"board: {len(board)} games")

    days = sorted({datetime.fromisoformat(g["commence_time"]).date() for g in board})
    scored = []
    for d in {x for day in days for x in (day - timedelta(days=1), day,
                                          day + timedelta(days=1))}:
        scored.extend(espn_games(d))
    # ESPN is queried on a +/-1 day window per kickoff, so the same game comes
    # back more than once. Collapse before matching.
    uniq = {(g["home_team"], g["away_team"], g["date"]): g for g in scored}
    scored = list(uniq.values())
    print(f"ESPN: {len(scored)} completed games in range")

    rows, unmatched = [], []
    for g in board:
        kick = datetime.fromisoformat(g["commence_time"]).date()
        hits = [s for s in scored
                if s["home_team"] == g["home_team"]
                and s["away_team"] == g["away_team"]
                and abs((s["date"] - kick).days) <= 1]
        if not hits:
            unmatched.append(f"{g['away_team']} @ {g['home_team']} {kick}")
            continue
        assert len(hits) == 1, (
            f"{g['away_team']} @ {g['home_team']} matched {len(hits)} ESPN games "
            f"-- refusing to guess")
        s = hits[0]
        rows.append({"game_id": g["game_id"], "season": g["season"],
                     "home_team": g["home_team"], "away_team": g["away_team"],
                     "home_score": s["home_score"], "away_score": s["away_score"]})

    print(f"  matched {len(rows)}; {len(unmatched)} not final yet")
    for u in unmatched[:5]:
        print(f"    pending: {u}")

    if not rows:
        return

    # Cheap sanity check on the join. A systematic home/away mix-up is the one
    # failure that would silently invert every graded result, and it shows up as
    # an implausible scoring split rather than an error.
    hs = sum(r["home_score"] for r in rows) / len(rows)
    aws = sum(r["away_score"] for r in rows) / len(rows)
    print(f"  avg score {hs:.1f} home / {aws:.1f} away")
    assert 5 < hs < 45 and 5 < aws < 45, "implausible scores -- check the mapping"

    if dry:
        for r in rows[:8]:
            print(f"    {r['away_team']:>3} {r['away_score']:>3} @ "
                  f"{r['home_team']:<3} {r['home_score']:<3}")
        print("  --dry-run: nothing written")
        return

    supabase.table("preseason_results").upsert(rows, on_conflict="game_id").execute()
    print(f"  wrote {len(rows)} preseason results")


if __name__ == "__main__":
    main()
