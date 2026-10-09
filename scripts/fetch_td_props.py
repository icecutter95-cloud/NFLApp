"""
Capture NFL touchdown-scorer prices from every book that quotes them.

Phase 0 of the touchdown work: capture only. No model, no screen, no pick.
The point is to own a clean, append-only price history BEFORE anything is
built on top of it -- the two bugs found in the college work this fortnight
(a live in-game line stored as the "close", and predictions re-stamped on
every run) both came from grading against a value a cron could rewrite.

What the board looks like, measured 2026-10-05 across 16 games and 8 books:
a full anytime board's implied probabilities sum to ~5.52 against a vig-free
~4.75, so the hold is ~16%, three to four times a -110 spread. But the best of
8 books beats the median book by 9.1% of payout at the median player and 24%
at the 90th percentile. The dispersion is the opportunity, and capturing it
accurately is the whole job here.

Best-of-N is biased upward by construction (see fetch_multibook_lines.py),
so this records every book's price and lets the analysis decide. It never
stores a "best price" as if it were the market.

Cost: 1 credit per market per event. Both markets for a 16-game slate is 32
credits, so an hourly job is ~5,400/week against the 51,000 we hold.

Writes only when a price has CHANGED since that player's last row at that
book -- otherwise an hourly job appends ~7,000 identical rows a day.

Usage:
    python fetch_td_props.py                  # both markets, this week
    python fetch_td_props.py --anytime        # anytime only (half the credits)
    python fetch_td_props.py --days 3         # only games inside 3 days
    python fetch_td_props.py --dry-run
"""

import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import dotenv_values

warnings.filterwarnings("ignore")

from config import DATA_DIR, CURRENT_SEASON as SEASON
from fetch_historical_lines import TEAM_NAME_TO_ABBR
from score_week import supabase, fetch_all

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = {"anytime": "player_anytime_td", "first": "player_1st_td"}

# Environment FIRST, then the .env file. dotenv_values() reads the file and
# nothing else -- it does not fall back to os.environ -- so on a GitHub runner,
# where the secret arrives as an environment variable and no .env exists, the
# key was always None. Combined with the clean `return` this used to do, the
# workflow ran every six hours from 2026-10-05 to 2026-10-08, reported success
# every time, and wrote not one row. That is the same shape as the three dead
# crons found the same week (refresh-public-betting, refresh-injuries, and the
# metrics job that never saw the current season); this one was self-inflicted.
KEY = (os.environ.get("ODDS_API_KEY")
       or dotenv_values(DATA_DIR.parent / ".env").get("ODDS_API_KEY"))


def roster_ids(season: int):
    """Two lookups: exact full name, and (team, surname) as a fallback.

    The books spell a player out ("Bijan Robinson"); the play-by-play
    abbreviates ("B.Robinson"), which collides constantly -- two Browns named
    B.Robinson would be indistinguishable. Matching on the gsis id removes the
    ambiguity, so the price and the touchdown join on the same key.

    Exact names alone are not enough. Books use the name on the broadcast and
    rosters use the legal one: "Joshua Palmer" is Josh Palmer, "Drew Ogletree"
    is Andrew Ogletree. The fallback matches on surname WITHIN THE TWO TEAMS
    PLAYING, and only when that is unique -- a surname is ambiguous league-wide
    (three Palmers) but almost never inside one game.

    Unmatched names are stored with a null id rather than dropped: a
    practice-squad call-up priced at +2500 is exactly the kind of row this
    table exists to keep.
    """
    try:
        import nfl_data_py as nfl
        r = nfl.import_seasonal_rosters([season])
    except Exception as e:
        print(f"  roster lookup unavailable ({type(e).__name__}); "
              f"storing prices without player ids")
        return {}, {}
    r = r.dropna(subset=["player_id", "player_name"])
    exact, by_team = {}, {}
    for _, p in r.iterrows():
        exact[norm(p.player_name)] = p.player_id
        parts = norm(p.player_name).split()
        by_team.setdefault((p.team, parts[-1]), set()).add(
            (p.player_id, parts[0] if len(parts) > 1 else ""))
    return exact, by_team


def match(player: str, home: str, away: str, exact: dict, by_team: dict):
    """gsis id, a synthetic id for non-player entries, or None."""
    n = norm(player)
    if n in exact:
        return exact[n]
    # "No Scorer" / "No Touchdown" are market entries, not people.
    if n.startswith("no ") and ("scorer" in n or "touchdown" in n):
        return "NO_SCORER"
    d = dst_id(player, home, away)
    if d:
        return d
    # Surname within the two teams playing, accepted only when unique. A
    # surname is ambiguous league-wide (three Palmers) but rarely inside a game.
    def by_surname(surname, first=""):
        hits = set()
        for team in (home, away):
            hits |= by_team.get((team, surname), set())
        if len(hits) > 1 and first:
            # Both teams can field the same surname -- Buffalo's Josh Palmer and
            # New England's Tejhaun Palmer are in the same game. A nickname is a
            # prefix of the legal name or the reverse (Josh/Joshua, Mike/Michael),
            # which separates them without a nickname dictionary.
            hits = {h for h in hits
                    if h[1].startswith(first) or first.startswith(h[1])}
        return hits.pop()[0] if len(hits) == 1 else None

    parts = n.split()
    hit = by_surname(parts[-1], parts[0] if len(parts) > 1 else "")
    if hit:
        return hit
    # Some books transpose the name: BetRivers and Fanatics both quote the
    # 49ers rookie back as "James Jordan", who is Jordan James. Try the
    # reversal, still team-scoped and still only when unique.
    if len(parts) == 2:
        rev = f"{parts[1]} {parts[0]}"
        if rev in exact:
            return exact[rev]
        return by_surname(parts[0], parts[1])
    return None


# Books also price the defence to score ("Atlanta Falcons D/ST", "Buffalo Bills
# Defense"). Those are real prices and are kept, but they are not players and
# will never match a roster id, so they get an unmistakable synthetic id rather
# than a null that reads like a failed match.
NICKNAME = {full.lower().split()[-1]: abbr
            for full, abbr in TEAM_NAME_TO_ABBR.items()}


def dst_id(name: str, home: str, away: str):
    n = str(name).lower()
    if not any(t in n for t in ("d/st", "defense", "defence")):
        return None
    for nick, abbr in NICKNAME.items():
        if nick in n and abbr in (home, away):
            return f"DST_{abbr}"
    return "DST"


def norm(name: str) -> str:
    """Strip the punctuation and suffixes books disagree about."""
    s = str(name).lower().replace(".", "").replace("'", "").replace("-", " ")
    for suf in (" jr", " sr", " ii", " iii", " iv", " v"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split())


def main():
    dry = "--dry-run" in sys.argv
    markets = ["anytime"] if "--anytime" in sys.argv else list(MARKETS)
    days = 8
    if "--days" in sys.argv:
        days = int(sys.argv[sys.argv.index("--days") + 1])
    cutoff = datetime.now(timezone.utc) + timedelta(days=days)

    if not KEY:
        # Hard failure, not a quiet return. A capture job that cannot reach the
        # feed has done nothing, and a green tick saying otherwise is worse
        # than a red one.
        sys.exit("ODDS_API_KEY not set (checked os.environ then .env) — "
                 "nothing captured")
    evs = requests.get(f"{API}/events", params={"apiKey": KEY}, timeout=30).json()
    evs = [e for e in evs
           if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) < cutoff]
    print(f"{len(evs)} games inside {days} days, markets {markets} "
          f"({len(evs) * len(markets)} credits)")
    if not evs:
        return

    exact, by_team = roster_ids(SEASON)
    print(f"  roster crosswalk: {len(exact)} players")

    rows, seen, remaining = [], 0, None
    for e in evs:
        r = requests.get(
            f"{API}/events/{e['id']}/odds",
            params={"apiKey": KEY, "regions": "us", "oddsFormat": "american",
                    "markets": ",".join(MARKETS[m] for m in markets)},
            timeout=40)
        remaining = r.headers.get("x-requests-remaining", remaining)
        if r.status_code != 200:
            print(f"  {e['away_team']} @ {e['home_team']}: {r.status_code} "
                  f"{r.text[:90]}")
            continue
        d = r.json()
        home = TEAM_NAME_TO_ABBR.get(e["home_team"], e["home_team"])
        away = TEAM_NAME_TO_ABBR.get(e["away_team"], e["away_team"])
        inv = {v: k for k, v in MARKETS.items()}
        for b in d.get("bookmakers", []):
            for m in b.get("markets", []):
                mk = inv.get(m["key"])
                if mk is None:
                    continue
                for o in m.get("outcomes", []):
                    # The player is in `description`; `name` is just "Yes".
                    player = o.get("description") or o.get("name")
                    if not player or player == "Yes":
                        continue
                    seen += 1
                    rows.append({
                        "event_id": e["id"], "season": SEASON,
                        "commence_time": e["commence_time"],
                        "home_team": home, "away_team": away,
                        "book": b["key"], "market": mk, "player": player,
                        "player_id": match(player, home, away, exact, by_team),
                        "price": int(o["price"])})

    print(f"  {seen} prices from {len({r['book'] for r in rows})} books, "
          f"{len({r['player'] for r in rows})} players "
          f"(credits left {remaining})")
    if not rows:
        return

    # Change-only. One read of the table, then compare in memory.
    prev = {}
    for r in fetch_all("nfl_td_props",
                       columns="event_id,market,book,player,price,captured_at"):
        k = (r["event_id"], r["market"], r["book"], r["player"])
        if k not in prev or r["captured_at"] > prev[k][1]:
            prev[k] = (r["price"], r["captured_at"])
    new = []
    for r in rows:
        k = (r["event_id"], r["market"], r["book"], r["player"])
        p = prev.get(k)
        if p is not None and p[0] == r["price"]:
            continue
        r["note"] = "open" if p is None else "move"
        new.append(r)

    unmatched = {r["player"] for r in new if r["player_id"] is None}
    if unmatched:
        print(f"  {len(unmatched)} players with no roster id, e.g. "
              f"{sorted(unmatched)[:4]}")
    if dry:
        for r in new[:8]:
            print(f"    {r['book']:14} {r['market']:7} {r['player']:22} "
                  f"{r['price']:+5d}  {r['note']}")
        print(f"  --dry-run: {len(new)} rows would be written")
        return
    for i in range(0, len(new), 500):
        supabase.table("nfl_td_props").insert(new[i:i + 500]).execute()
    opens = sum(1 for r in new if r["note"] == "open")
    print(f"  wrote {len(new)} rows ({opens} new, {len(new)-opens} moves)")


if __name__ == "__main__":
    main()
