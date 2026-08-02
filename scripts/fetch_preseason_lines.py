"""
Pull NFL preseason lines. For looking at, not for betting.

Preseason is deliberately outside the model. Results turn on which backups play
and for how long, decided by coaches on the morning of the game -- none of which
any feature in this project observes. There is no edge here and none is claimed.

Isolation matters more than the feature does. These rows go to preseason_lines
and nowhere else:

  * book_lines feeds best_book_lines, which joins on the TEAM PAIR alone. A
    preseason DAL @ LAR would collide with the regular-season DAL @ LAR and put
    a wrong "best number" on the CLV screen.
  * line_history feeds line_open_close, which is where week_open and closing
    lines come from -- and therefore the model's training target. A preseason
    row landing there corrupts the one number CLV tracking cannot rebuild.

Usage:
    python fetch_preseason_lines.py
    python fetch_preseason_lines.py --dry-run
"""

import sys
import warnings
import requests

warnings.filterwarnings("ignore")

from config import ODDS_API_KEY
from fetch_historical_lines import TEAM_NAME_TO_ABBR
from score_week import supabase

URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl_preseason/odds"
# Same panel the regular-season shopping view uses, so the numbers are
# comparable if you ever look at both.
PANEL = ["draftkings", "fanduel", "betmgm", "williamhill_us",
         "betrivers", "espnbet", "betonlineag", "lowvig", "bovada"]


def main():
    dry = "--dry-run" in sys.argv
    r = requests.get(URL, params={
        "apiKey": ODDS_API_KEY, "regions": "us",
        "markets": "spreads,totals", "oddsFormat": "american",
        "bookmakers": ",".join(PANEL),
    }, timeout=30)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:200]}")
        return

    games = r.json()
    print(f"{len(games)} preseason games on the board "
          f"(credits left {r.headers.get('x-requests-remaining')})")

    rows, unmapped = [], set()
    for g in games:
        home = TEAM_NAME_TO_ABBR.get(g.get("home_team"))
        away = TEAM_NAME_TO_ABBR.get(g.get("away_team"))
        if not home or not away:
            # Loudly, not silently: an unmapped team is how the Rams went
            # invisible for weeks earlier in this project.
            unmapped.add(g.get("home_team") if not home else g.get("away_team"))
            continue

        for bk in g.get("bookmakers", []):
            spreads = next((m for m in bk["markets"] if m["key"] == "spreads"), None)
            totals = next((m for m in bk["markets"] if m["key"] == "totals"), None)
            sh = next((o for o in spreads["outcomes"] if o["name"] == g["home_team"]), None) if spreads else None
            sa = next((o for o in spreads["outcomes"] if o["name"] == g["away_team"]), None) if spreads else None
            ov = next((o for o in totals["outcomes"] if o["name"] == "Over"), None) if totals else None
            un = next((o for o in totals["outcomes"] if o["name"] == "Under"), None) if totals else None
            if sh is None and ov is None:
                continue
            rows.append({
                "game_id": g["id"], "book": bk["key"],
                "home_team": home, "away_team": away,
                "commence_time": g.get("commence_time"),
                "spread_home": sh.get("point") if sh else None,
                "spread_home_price": sh.get("price") if sh else None,
                "spread_away_price": sa.get("price") if sa else None,
                "total": ov.get("point") if ov else None,
                "over_price": ov.get("price") if ov else None,
                "under_price": un.get("price") if un else None,
            })

    if unmapped:
        print(f"  WARNING unmapped teams: {sorted(unmapped)}")

    by_game = {}
    for r_ in rows:
        by_game.setdefault((r_["away_team"], r_["home_team"], r_["commence_time"][:10]), []).append(r_)
    for (a, h, d), qs in sorted(by_game.items(), key=lambda kv: kv[0][2]):
        sp = [q["spread_home"] for q in qs if q["spread_home"] is not None]
        to = [q["total"] for q in qs if q["total"] is not None]
        print(f"  {d}  {a:>3} @ {h:<3}  {len(qs)} books  "
              f"spread {min(sp):+.1f}..{max(sp):+.1f}" if sp else "",
              f" total {min(to):.1f}..{max(to):.1f}" if to else "")

    if dry:
        print("  --dry-run: nothing written")
        return
    if rows:
        supabase.table("preseason_lines").upsert(rows, on_conflict="game_id,book").execute()
        print(f"  wrote {len(rows)} quotes across {len(by_game)} games")


if __name__ == "__main__":
    main()
