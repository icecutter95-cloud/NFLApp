"""
Final scores for college games, so the CFB tab builds an ATS record.

Until now the college side logged predictions and tracked CLV, but nothing ever
fetched a score -- so "did the pick win" had no answer and no history was
accumulating for the one thing that ultimately matters.

Why ESPN and not CFBD
---------------------
CFBD is the source for everything else here, but its monthly quota was exhausted
on 2026-08-11 and does not reset until September. Grading through it would have
left the opening weekend ungraded for a fortnight. ESPN's college scoreboard
needs no key.

ESPN team names go through the SAME crosswalk as the rest of the project
(cfb_teams.to_key / cfbd_to_key). On a sample of 45 games, 86 of 90 sides
mapped; every miss was an FCS opponent, which is exactly what we never predict
on, so unmapped names are skipped rather than treated as an error. A game whose
BOTH sides fail to map simply is not one of ours.

Grading conventions -- identical to clv_tracking and preseason, deliberately
---------------------------------------------------------------------------
    open_line is the HOME spread; negative means home is laid points
    home covers when (home_score - away_score) + open_line > 0

Usage:
    python fetch_cfb_results.py
    python fetch_cfb_results.py --dry-run
"""

import sys
import warnings
from datetime import datetime, timedelta

import requests

warnings.filterwarnings("ignore")

from cfb_teams import to_key, cfbd_to_key
from score_week import supabase

ESPN = ("https://site.api.espn.com/apis/site/v2/sports/football/"
        "college-football/scoreboard")
FBS_GROUP = 80


def espn_key(team: dict):
    """Our crosswalk key for an ESPN team, or None if it is not FBS."""
    for cand in (team.get("location"), team.get("displayName"),
                 team.get("shortDisplayName"), team.get("name")):
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


def espn_games(day):
    r = requests.get(ESPN, params={"dates": day.strftime("%Y%m%d"),
                                   "groups": FBS_GROUP, "limit": 400}, timeout=90)
    r.raise_for_status()
    out = []
    for e in r.json().get("events", []):
        c = e["competitions"][0]
        if not c["status"]["type"]["completed"]:
            continue
        side = {}
        for x in c["competitors"]:
            k = espn_key(x["team"])
            if k is None:
                break
            side[x["homeAway"]] = (k, int(x["score"]))
        if "home" in side and "away" in side:
            out.append({"home_team": side["home"][0], "home_score": side["home"][1],
                        "away_team": side["away"][0], "away_score": side["away"][1],
                        "date": datetime.fromisoformat(
                            e["date"].replace("Z", "+00:00")).date()})
    return out


def main():
    dry = "--dry-run" in sys.argv

    board = (supabase.table("cfb_predictions")
             .select("game_id, season, home_team, away_team, commence_time")
             .eq("bet_type", "spread").execute().data or [])
    if not board:
        print("No CFB predictions to grade")
        return
    print(f"board: {len(board)} games")

    kicks = sorted({datetime.fromisoformat(g["commence_time"]).date() for g in board})
    # Only look at days that have actually happened; the board runs months ahead.
    today = datetime.utcnow().date()
    days = sorted({d for k in kicks if k <= today
                   for d in (k - timedelta(days=1), k, k + timedelta(days=1))
                   if d <= today})
    if not days:
        print("  no games have kicked off yet — nothing to grade")
        return
    print(f"  checking {len(days)} dates through {max(days)}")

    scored = {}
    for d in days:
        for g in espn_games(d):
            scored[(g["home_team"], g["away_team"], g["date"])] = g
    print(f"ESPN: {len(scored)} completed FBS games in range")

    rows, pending = [], 0
    for g in board:
        kick = datetime.fromisoformat(g["commence_time"]).date()
        hits = [s for s in scored.values()
                if s["home_team"] == g["home_team"]
                and s["away_team"] == g["away_team"]
                and abs((s["date"] - kick).days) <= 1]
        if not hits:
            pending += 1
            continue
        assert len(hits) == 1, (
            f"{g['away_team']} @ {g['home_team']} matched {len(hits)} ESPN games "
            f"-- refusing to guess")
        s = hits[0]
        rows.append({"game_id": g["game_id"], "season": g["season"],
                     "home_team": g["home_team"], "away_team": g["away_team"],
                     "home_score": s["home_score"], "away_score": s["away_score"]})

    print(f"  matched {len(rows)}; {pending} not final yet")
    if not rows:
        return

    # A systematic home/away swap is the failure that would silently invert every
    # graded result, and it shows up as implausible scoring rather than an error.
    hs = sum(r["home_score"] for r in rows) / len(rows)
    aws = sum(r["away_score"] for r in rows) / len(rows)
    print(f"  avg score {hs:.1f} home / {aws:.1f} away")
    assert 3 < hs < 70 and 3 < aws < 70, "implausible scores -- check the mapping"

    if dry:
        for r in rows[:10]:
            print(f"    {r['away_team']:>18} {r['away_score']:>3} @ "
                  f"{r['home_team']:<18} {r['home_score']}")
        print("  --dry-run: nothing written")
        return

    supabase.table("cfb_results").upsert(rows, on_conflict="game_id").execute()
    print(f"  wrote {len(rows)} CFB results")


if __name__ == "__main__":
    main()
