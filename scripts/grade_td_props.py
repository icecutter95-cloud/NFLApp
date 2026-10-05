"""
Ground truth for the touchdown markets, from play-by-play.

Every scoring play in nfl_data_py carries td_player_id -- a gsis id, the same
id the roster crosswalk in fetch_td_props.py attaches to each quoted price --
so grading needs no new data source and eight cached seasons can be graded for
free. That is the main reason this market is worth the effort: the spread model
gets 16 decisions a week, while one slate prices ~400 players.

scored_first is the first touchdown of the game in play order, which is what
the first-scorer market settles on. A game with no touchdowns has no first
scorer.

Props are joined to results on TEAM PAIR AND DATE, never on the Odds API event
id: that id is re-issued when a kickoff moves (it silently froze a college
game's line history last month), and clv_tracking already joins this way for
the same reason.

Usage:
    python grade_td_props.py                 # current season, played games
    python grade_td_props.py 2024            # a past season, from cache
    python grade_td_props.py --report        # grade, then score the captures
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, CURRENT_SEASON
from score_week import supabase, fetch_all

COLS = ["game_id", "season", "week", "home_team", "away_team",
        "touchdown", "td_player_id", "td_player_name", "td_team"]


def load_pbp(season: int) -> pd.DataFrame:
    """Cached parquet if present, else fetch and cache it."""
    f = DATA_DIR / f"pbp_{season}.parquet"
    if f.exists():
        return pd.read_parquet(f, columns=COLS)
    import nfl_data_py as nfl
    d = nfl.import_pbp_data([season], downcast=True, cache=False)
    d.to_parquet(f)
    return d[COLS]


def imp(american) -> np.ndarray:
    """American odds to implied probability."""
    a = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, 100.0 / (a + 100.0), -a / (-a + 100.0))


def grade(season: int) -> pd.DataFrame:
    d = load_pbp(season)
    td = d[(d.touchdown == 1) & d.td_player_id.notna()].copy()
    if td.empty:
        print(f"{season}: no touchdowns in the play-by-play yet")
        return pd.DataFrame()

    # Play order within a game is the file's own row order, so the first
    # touchdown row per game is the first touchdown of that game.
    first = td.groupby("game_id").head(1)[["game_id", "td_player_id"]].copy()
    first["scored_first"] = True

    g = (td.groupby(["game_id", "season", "week", "home_team", "away_team",
                     "td_player_id"], as_index=False)
           .agg(player=("td_player_name", "first"),
                team=("td_team", "first"),
                tds=("touchdown", "size")))
    g = g.merge(first, on=["game_id", "td_player_id"], how="left")
    g["scored_first"] = g.scored_first.fillna(False).astype(bool)
    g = g.rename(columns={"td_player_id": "player_id"})
    for c in ("season", "week", "tds"):
        g[c] = g[c].astype(int)
    return g


def report(season: int):
    """How good is the capture, and what does the board look like graded?"""
    props = pd.DataFrame(fetch_all("nfl_td_props"))
    if props.empty:
        print("\nno captures yet")
        return
    props["captured_at"] = pd.to_datetime(props.captured_at, utc=True,
                                          format="ISO8601")
    props["kick"] = pd.to_datetime(props.commence_time, utc=True,
                                   format="ISO8601")
    print(f"\n{len(props)} captured prices | {props.event_id.nunique()} games"
          f" | {props.book.nunique()} books | "
          f"{props.captured_at.min():%m-%d %H:%M} to "
          f"{props.captured_at.max():%m-%d %H:%M}")
    for m, d in props.groupby("market"):
        boards = d.groupby(["event_id", "book"]).size()
        print(f"  {m:8} {len(d):5d} prices, {d.player.nunique():3d} players, "
              f"{int((boards >= 20).sum())} full boards")

    props["p"] = imp(props.price)
    latest = (props.sort_values("captured_at")
              .groupby(["event_id", "market", "book", "player"]).tail(1))
    at = latest[latest.market == "anytime"].copy()
    if at.empty:
        return

    # Hold, on full boards only: a partial list understates the sum.
    n = at.groupby(["event_id", "book"]).player.transform("size")
    full = at[n >= 20]
    if not full.empty:
        s = full.groupby(["event_id", "book"]).p.sum()
        print(f"\nanytime overround, full boards only (vig-free sums to the"
              f" game's expected offensive TDs, ~4.75):")
        print(f"  mean {s.mean():.2f}  ->  hold ~{(s.mean() / 4.75 - 1) * 100:.0f}%")
        per = (full.groupby("book").p.sum()
               / full.groupby("book").event_id.nunique())
        print("  " + "  ".join(f"{b} {v:.2f}" for b, v in
                               per.round(2).sort_values().items()))

    # Cross-book dispersion -- what best-price capture is worth. Recorded as a
    # MEASUREMENT, never as a baseline: best-of-N is biased upward by
    # construction (see fetch_multibook_lines.py), so a model graded against it
    # manufactures edge out of the bias.
    disp = (at.groupby(["event_id", "player"])
              .agg(n=("book", "size"), best=("price", "max"),
                   med=("price", "median")).reset_index())
    disp = disp[disp.n >= 5]
    if not disp.empty:
        gain = (imp(disp.med) / imp(disp.best) - 1) * 100
        print(f"\nbest of {at.book.nunique()} books vs the median book, "
              f"{len(disp)} players priced at 5+ books:")
        print(f"  median +{np.median(gain):.1f}% payout, "
              f"75th +{np.percentile(gain, 75):.1f}%, "
              f"90th +{np.percentile(gain, 90):.1f}%")
        holder = (at.loc[at.groupby(["event_id", "player"]).price.idxmax()]
                    .book.value_counts().head(4))
        print("  best price held by: "
              + ", ".join(f"{b} {c}" for b, c in holder.items()))

    res = pd.DataFrame(fetch_all("nfl_td_results"))
    if res.empty:
        print("\nnothing graded yet")
        return

    # Team pair + date, never the event id.
    res["key"] = res.home_team + "_" + res.away_team
    at["key"] = at.home_team + "_" + at.away_team
    sched = res.groupby(["key", "game_id"], as_index=False).week.first()
    m = at.merge(sched, on="key", how="inner")
    if m.empty:
        print("\nno captured game has finished yet")
        return
    scorers = set(zip(res.game_id, res.player_id))
    m["hit"] = [(g, p) in scorers for g, p in zip(m.game_id, m.player_id)]
    best = m.loc[m.groupby(["game_id", "player"]).price.idxmax()]
    pay = np.where(best.price > 0, best.price / 100.0, 100.0 / -best.price)
    pnl = np.where(best.hit, pay, -1.0)
    print(f"\n{len(best)} graded player-games, best available price, flat 1u:")
    print(f"  {int(best.hit.sum())}-{int((~best.hit).sum())}, "
          f"P/L {pnl.sum():+.2f}u on {len(best)}u "
          f"({pnl.sum() / len(best) * 100:+.1f}%)")
    print("  NOTE: best-of-N is biased upward. This measures the capture, not")
    print("        a strategy -- betting every player on the board loses.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    season = int(args[0]) if args else CURRENT_SEASON
    g = grade(season)
    if not g.empty:
        print(f"{season}: {len(g)} player-games with a touchdown across "
              f"{g.game_id.nunique()} games "
              f"({int(g.scored_first.sum())} first scorers)")
        recs = g.to_dict("records")
        for i in range(0, len(recs), 500):
            supabase.table("nfl_td_results").upsert(
                recs[i:i + 500], on_conflict="game_id,player_id").execute()
        print(f"  wrote {len(recs)} result rows")
    if "--report" in sys.argv:
        report(season)


if __name__ == "__main__":
    main()
