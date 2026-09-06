"""
Shared reader for Action Network public-betting pages.

Both leagues use the identical payload shape, so the parsing lives here rather
than in two scripts. Conventions duplicated across files are how this project
has repeatedly ended up with two subtly different answers to the same question.

The page embeds its whole scoreboard in a __NEXT_DATA__ blob and serves it to an
unauthenticated request -- no login, no browser, no challenge.

Two constraints, both measured rather than assumed (2026-09-06):

  * NOT runnable on GitHub Actions. A residential connection returns every
    game; a runner returns a page with no __NEXT_DATA__ at all. Run it from a
    home machine.

  * The per-book breakout is not real. College: 194 game-markets where all eight
    book ids carry IDENTICAL percentages against 4 that differ, most of those
    being duplicate rows. NFL: 32 identical, 0 differing. It is one consensus
    figure under eight labels, so only book 15 is collected and nothing here may
    be presented as a particular book's numbers.

Moneyline is deliberately excluded. Extreme splits live there -- 94/6 on a game
whose spread was 47/53 -- so holding a moneyline figure against a spread table
manufactures a contradiction that was never there.
"""

import json
import re
from datetime import datetime, timedelta, timezone

import requests

URL = "https://www.actionnetwork.com/{league}/public-betting"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# Consensus. Every other id repeats it; see the note above.
CONSENSUS_BOOK = "15"


def fetch(league):
    """(games, books, observed_at, cdn_age) for 'nfl' or 'ncaaf'."""
    r = requests.get(URL.format(league=league), headers={"User-Agent": UA}, timeout=90)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  r.text, re.S)
    if not m:
        raise RuntimeError(
            "__NEXT_DATA__ not found. Almost certainly the IP, not the markup: "
            "this returns the full payload from a residential connection and "
            "nothing at all from a GitHub Actions runner, verified by running "
            "both within minutes of each other on 2026-09-06. Run it from a "
            "home machine. If it also fails locally, the page really did change.")
    data = json.loads(m.group(1))
    # The page is CDN-cached, so back the age out to recover roughly when these
    # numbers were generated. captured_at is the OBSERVATION time, which is what
    # pairs a capture with the line snapshot that was live beside it.
    age = int(r.headers.get("age") or 0)
    observed = datetime.now(timezone.utc) - timedelta(seconds=age)
    props = data["props"]["pageProps"]
    books = {str(k): v.get("display_name")
             for k, v in (props.get("allBooks") or {}).items()}
    return props["scoreboardResponse"]["games"], books, observed, age


def outcomes(game, book=CONSENSUS_BOOK):
    """{'spread': {...}, 'total': {...}} of (tickets, money, line) per side."""
    mk = (game.get("markets") or {}).get(book) or {}
    ev = mk.get("event") or {}
    out = {}
    for market in ("spread", "total"):
        pick = {}
        for o in ev.get(market) or []:
            side = o.get("side")
            if side not in ("home", "away", "over", "under"):
                continue
            bi = o.get("bet_info") or {}
            cand = ((bi.get("tickets") or {}).get("percent"),
                    (bi.get("money") or {}).get("percent"),
                    o.get("value"))
            # A side can appear twice, once as an empty (0, 0) placeholder.
            # Taking the last occurrence would silently store zeros.
            if side not in pick or (not pick[side][0] and cand[0]):
                pick[side] = cand
        if market == "spread":
            a, b = pick.get("home"), pick.get("away")
        else:
            a, b = pick.get("over"), pick.get("under")   # home column = over
        if a and b and (a[0] or b[0]):
            out[market] = {"home_bets_pct": a[0], "away_bets_pct": b[0],
                           "home_money_pct": a[1], "away_money_pct": b[1],
                           "line_at_capture": a[2]}
    return out


def kickoff(game):
    return datetime.fromisoformat(game["start_time"].replace("Z", "+00:00")).date()
