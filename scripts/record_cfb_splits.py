"""
Record public betting splits for college games, captured by hand.

There is no automated source for these. Action Network's splits are a
proprietary product behind Cloudflare and JS rendering, and every free feed
checked (ESPN scoreboard, game summary, pickcenter) carries prices but no ticket
or money percentages. So the capture is a person reading the page; this script
is only the part that stops the numbers evaporating into chat scrollback.

Input is JSON -- a file path, or stdin -- shaped so a browsing session can emit
it directly:

    {
      "captured_at": "2026-09-03T18:00:00Z",     # optional, defaults to now
      "source": "action_network",                 # optional
      "games": [
        {"away": "TOLEDO", "home": "MICHIGAN_ST",
         "away_bets": 62, "home_bets": 38,
         "away_money": 71, "home_money": 29}
      ]
    }

Team names run through the same crosswalk as everything else, so
"Michigan State" works as well as "MICHIGAN_ST".

captured_at is the OBSERVATION time, not the write time, so a capture pairs with
the line snapshot that was live when it was taken. Re-running with the same
timestamp is idempotent.

A note on what this data is good for
------------------------------------
Capturing only the games you already like makes the sample selection-biased and
useless for testing whether any of these signals work. If that matters, capture
a pre-committed slate -- every game in a fixed window -- regardless of whether
you fancy any of them. The script prints the coverage it sees so the bias is
visible rather than assumed.

Usage:
    python record_cfb_splits.py splits.json
    cat splits.json | python record_cfb_splits.py -
    python record_cfb_splits.py splits.json --dry-run
"""

import json
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from cfb_teams import to_key, cfbd_to_key
from score_week import supabase

TOL = 3          # percentages are rounded on the page; allow a little slack


def key_for(name):
    if not name:
        return None
    for fn in (to_key, cfbd_to_key):
        try:
            k = fn(name)
        except Exception:
            k = None
        if k:
            return k
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print(__doc__)
        return
    raw = sys.stdin.read() if args[0] == "-" else open(args[0], encoding="utf-8").read()
    payload = json.loads(raw)

    captured = payload.get("captured_at")
    captured = (datetime.now(timezone.utc).isoformat() if not captured
                else datetime.fromisoformat(captured.replace("Z", "+00:00")).isoformat())
    source = payload.get("source", "action_network")
    bet_type = payload.get("bet_type", "spread")

    board = (supabase.table("cfb_predictions")
             .select("game_id, home_team, away_team, commence_time, open_line")
             .eq("bet_type", bet_type).execute().data or [])
    by_pair = {(g["home_team"], g["away_team"]): g for g in board}
    print(f"board: {len(board)} games")

    rows, problems = [], []
    for g in payload.get("games", []):
        h, a = key_for(g.get("home")), key_for(g.get("away"))
        if not h or not a:
            problems.append(f"unmapped team: {g.get('away')} @ {g.get('home')}")
            continue
        match = by_pair.get((h, a))
        if not match:
            problems.append(f"not on the board: {a} @ {h}")
            continue

        hb, ab = g.get("home_bets"), g.get("away_bets")
        hm, am = g.get("home_money"), g.get("away_money")
        # Transcription check. A pair that does not sum to ~100 means a column
        # was misread or the two sides came from different rows.
        for label, x, y in (("bets", hb, ab), ("money", hm, am)):
            if x is not None and y is not None and abs(x + y - 100) > TOL:
                problems.append(f"{a} @ {h}: {label} split sums to {x + y}, not 100")

        rows.append({"game_id": match["game_id"], "captured_at": captured,
                     "source": source, "bet_type": bet_type,
                     "home_bets_pct": hb, "away_bets_pct": ab,
                     "home_money_pct": hm, "away_money_pct": am,
                     "line_at_capture": g.get("line", match.get("open_line")),
                     "note": g.get("note")})

    for p in problems:
        print(f"  !! {p}")
    if problems and not rows:
        return
    if any("sums to" in p for p in problems):
        print("\n  refusing to write -- fix the transcription above first")
        return

    print(f"  {len(rows)} of {len(payload.get('games', []))} games matched"
          f"  ({len(rows) / max(len(board), 1) * 100:.0f}% of the board)")
    if len(rows) < len(board) * 0.5:
        print("  note: this is a partial capture. Fine for deciding tonight's")
        print("        plays, but a selective sample cannot test whether any")
        print("        of these signals actually work.")

    if dry:
        for r in rows[:8]:
            print(f"    {r['away_team'] if 'away_team' in r else ''}"
                  f"    bets {r['away_bets_pct']}/{r['home_bets_pct']}"
                  f"  money {r['away_money_pct']}/{r['home_money_pct']}")
        print("  --dry-run: nothing written")
        return

    supabase.table("cfb_public_splits").upsert(
        rows, on_conflict="game_id,bet_type,source,captured_at").execute()
    print(f"  wrote {len(rows)} split captures at {captured[:19]}")


if __name__ == "__main__":
    main()
