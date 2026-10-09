"""
The four touchdown-prop questions, in one place, runnable weekly.

Every section prints its sample size first and refuses to conclude below a
stated minimum. Most of them will say "not enough yet" for several weeks --
that is the honest output of a harness built before the data, and it is built
first so the collection it depends on starts now rather than after someone
thinks of it.

  1  DE-VIG          multiplicative vs power vs Shin. How much do they even
                     disagree, and which one calibrates once results exist?
  2  CALIBRATION     de-vigged probability against realised hit rate. The acid
                     test: if 10% players hit 7% of the time, every EV on the
                     board is overstated by three points and nothing else here
                     matters.
  3  MOVEMENT        does a shortening price predict scoring, beyond what its
                     own closing price already says? Same question the CLV work
                     asks about spreads, on a market with ~25x the sample.
  4  REPRICING       when a player's injury status changes, how long until each
                     book moves his team-mates' prices? The one plausible edge
                     here that rests on speed rather than forecasting.

Needs nfl_td_props (append-only prices), nfl_td_results (play-by-play truth)
and nfl_injury_log (append-only status changes, which only began on
2026-10-08 -- section 4 cannot see further back than that).

Usage:
    python scripts/td_props_analysis.py
    python scripts/td_props_analysis.py --market first
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

from score_week import fetch_all

# Below these, a section reports its sample and stops.
MIN_GRADED = 200          # calibration
MIN_MOVES = 150           # movement
MIN_EVENTS = 15           # repricing

# Measured on 2,214 games: scorers = -0.101 + 0.0998 x total.
SCORERS_A, SCORERS_B, SCORERS_MEAN = -0.101, 0.0998, 4.43


def implied(american):
    a = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, 100.0 / (a + 100.0), -a / (-a + 100.0))


def payout(american):
    a = np.asarray(american, dtype=float)
    return np.where(a > 0, a / 100.0, 100.0 / -a)


# ---------------------------------------------------------------------------
# 1. De-vig methods
# ---------------------------------------------------------------------------

def devig_multiplicative(q, target):
    """Scale every probability by the same factor. Overstates longshots."""
    s = q.sum()
    return q * target / s if s > 0 else q


def devig_power(q, target):
    """Find k with sum(q^k) = target.

    Raising to a power > 1 shrinks small probabilities proportionally more than
    large ones, which is the direction the favourite-longshot bias needs: books
    hold more on a +3000 player than on a -200 one, so his raw price overstates
    him by more.
    """
    q = np.clip(q, 1e-9, 1 - 1e-9)
    if q.sum() <= target:
        return devig_multiplicative(q, target)
    try:
        k = brentq(lambda k: (q ** k).sum() - target, 1.0, 25.0, xtol=1e-8)
    except ValueError:
        return devig_multiplicative(q, target)
    return q ** k


def devig_shin(q, target=1.0):
    """Shin's model: price includes a share z of better-informed money.

    Defined for a complete market whose true probabilities sum to 1, so it is
    applied to FIRST TD only. Anytime is not a complete market -- each player
    is his own binary -- and pretending otherwise would be a worse error than
    the longshot bias it is meant to fix.
    """
    s = q.sum()
    if s <= 0 or abs(target - 1.0) > 1e-6:
        return None

    # RAW probabilities and their raw sum, not normalised ones. The overround
    # IS the quantity Shin attributes to informed money, so dividing it out
    # first destroys the thing being solved for: z collapses to zero and the
    # method silently degenerates into the multiplicative de-vig it was meant
    # to improve on. (It did exactly that until this comment existed -- the two
    # columns printed identical numbers to four decimals.)
    def shin_p(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: shin_p(z).sum() - 1.0, 1e-9, 0.6, xtol=1e-10)
    except ValueError:
        return devig_multiplicative(q, target)
    return shin_p(z)


def board_with_devigs(props: pd.DataFrame, totals: dict) -> pd.DataFrame:
    """Latest pre-kickoff price per book -> per-player consensus + 3 de-vigs."""
    p = props.copy()
    p["captured_at"] = pd.to_datetime(p.captured_at, utc=True, format="ISO8601")
    p["kick"] = pd.to_datetime(p.commence_time, utc=True, format="ISO8601")
    p = p[p.captured_at < p.kick]
    if p.empty:
        return pd.DataFrame()
    p["q"] = implied(p.price)
    latest = (p.sort_values("captured_at")
              .groupby(["event_id", "market", "book", "player"]).tail(1))
    per = (latest.groupby(["event_id", "market", "player", "player_id",
                           "home_team", "away_team", "kick"], as_index=False)
           .agg(n_books=("book", "size"), best_price=("price", "max"),
                q_med=("q", "median")))

    out = []
    for (ev, mkt), d in per.groupby(["event_id", "market"]):
        tot = totals.get(ev)
        target = (1.0 if mkt == "first"
                  else (SCORERS_A + SCORERS_B * tot if tot else SCORERS_MEAN))
        q = d.q_med.to_numpy()
        d = d.copy()
        d["target"] = target
        d["p_mult"] = devig_multiplicative(q, target)
        d["p_power"] = devig_power(q, target)
        sh = devig_shin(q, target)
        d["p_shin"] = sh if sh is not None else np.nan
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

def attach_results(board: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join on team pair, never the Odds API event id (it gets re-issued)."""
    if board.empty or results.empty:
        return board.assign(hit=np.nan)
    res = results.copy()
    res["key"] = res.home_team + "_" + res.away_team
    sched = res.groupby(["key", "game_id"], as_index=False).week.first()
    b = board.copy()
    b["key"] = b.home_team + "_" + b.away_team
    b = b.merge(sched, on="key", how="left")
    scored = set(zip(res.game_id, res.player_id))
    firsts = set(zip(res[res.scored_first].game_id, res[res.scored_first].player_id))
    def mark(r):
        if pd.isna(r.game_id) or not r.player_id:
            return np.nan
        tgt = firsts if r.market == "first" else scored
        return float((r.game_id, r.player_id) in tgt)
    b["hit"] = b.apply(mark, axis=1)
    return b


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_devig(board):
    print("\n" + "=" * 70)
    print("1. DE-VIG — how much do the three methods disagree?")
    print("=" * 70)
    if board.empty:
        print("  no pre-kickoff prices"); return
    for mkt, d in board.groupby("market"):
        print(f"\n  {mkt} ({len(d)} player-markets, {d.event_id.nunique()} games)")
        print(f"    raw implied sums to {d.q_med.sum() / d.event_id.nunique():.2f} "
              f"per game against a target of {d.target.mean():.2f}")
        short = d[d.best_price <= 300]
        long_ = d[d.best_price >= 1500]
        for lab, sub in (("short (<= +300)", short), ("long (>= +1500)", long_)):
            if sub.empty:
                continue
            m, p = sub.p_mult.mean(), sub.p_power.mean()
            extra = (f", shin {sub.p_shin.mean():.4f}"
                     if sub.p_shin.notna().any() else "")
            print(f"    {lab:16} mult {m:.4f}  power {p:.4f}{extra}"
                  f"   power is {100 * (p / m - 1):+.1f}% vs mult")
        print("    power pulls longshots down and favourites up, which is the")
        print("    direction the favourite-longshot bias requires.")
    print("\n  Which is RIGHT is decided by section 2, not by theory.")


def section_calibration(board):
    print("\n" + "=" * 70)
    print("2. CALIBRATION — does a de-vigged probability mean what it says?")
    print("=" * 70)
    g = board[board.hit.notna()]
    print(f"  graded player-markets: {len(g)} (need {MIN_GRADED})")
    if len(g) < MIN_GRADED:
        print("  Not enough yet. Every captured game is still ahead of kickoff,")
        print("  or too few have been graded. This is the section that decides")
        print("  whether any EV on the TDs tab can be trusted, so it is the one")
        print("  to re-run first each week.")
        return
    for col in ("p_mult", "p_power", "p_shin"):
        d = g[g[col].notna()]
        if d.empty:
            continue
        d = d.assign(b=pd.cut(d[col], [0, .05, .10, .15, .25, .40, 1.0]))
        print(f"\n  {col}")
        tot_err = 0.0
        for bucket, sub in d.groupby("b", observed=True):
            exp, act, n = sub[col].mean(), sub.hit.mean(), len(sub)
            tot_err += abs(exp - act) * n
            print(f"    {str(bucket):14} n={n:4d}  predicted {exp:5.1%}  "
                  f"actual {act:5.1%}  ({act - exp:+.1%})")
        print(f"    weighted absolute error {tot_err / len(d):.3%}")


def section_movement(props):
    print("\n" + "=" * 70)
    print("3. MOVEMENT — does a shortening price predict scoring?")
    print("=" * 70)
    p = props.copy()
    p["captured_at"] = pd.to_datetime(p.captured_at, utc=True, format="ISO8601")
    moves = p[p.note == "move"]
    print(f"  price changes logged: {len(moves)} across "
          f"{moves.player.nunique()} players, {moves.book.nunique()} books")
    if len(moves) < MIN_MOVES:
        print(f"  Need {MIN_MOVES} before asking anything of them. The capture")
        print("  writes one row per CHANGE, so this grows every few hours.")
        return
    first = (p.sort_values("captured_at")
             .groupby(["event_id", "market", "book", "player"]).head(1)
             [["event_id", "market", "book", "player", "price"]]
             .rename(columns={"price": "open_price"}))
    last = (p.sort_values("captured_at")
            .groupby(["event_id", "market", "book", "player"]).tail(1)
            [["event_id", "market", "book", "player", "price"]]
            .rename(columns={"price": "last_price"}))
    m = first.merge(last, on=["event_id", "market", "book", "player"])
    m["drift"] = implied(m.last_price) - implied(m.open_price)
    movers = m[m.drift.abs() > 1e-9]
    big = m[m.drift.abs() > 0.01]
    print(f"  {len(movers)} of {len(m)} player-books moved at all; "
          f"{len(big)} by more than 1 point of implied probability")
    if len(movers):
        print(f"  among movers: median absolute drift "
              f"{movers.drift.abs().median():.2%}, largest "
              f"{movers.drift.abs().max():.2%}")
    print("  Outcome test needs section 2's graded sample — rerun once games")
    print("  that were captured have been played.")


def section_repricing(props, injuries):
    print("\n" + "=" * 70)
    print("4. REPRICING — how fast does each book react to injury news?")
    print("=" * 70)
    if injuries.empty:
        print("  nfl_injury_log is empty."); return
    inj = injuries.copy()
    inj["observed_at"] = pd.to_datetime(inj.observed_at, utc=True, format="ISO8601")
    material = inj[(inj.status.isin(["out", "doubtful"]))
                   & (inj.prev_status.isna() | ~inj.prev_status.isin(["out", "doubtful"]))
                   & inj.position.isin(["QB", "RB", "WR", "TE"])]
    print(f"  log spans {inj.observed_at.min():%Y-%m-%d} to "
          f"{inj.observed_at.max():%Y-%m-%d}, {len(inj)} changes")
    print(f"  material ones (a skill starter newly out/doubtful): {len(material)} "
          f"(need {MIN_EVENTS})")
    if len(material) < MIN_EVENTS:
        print("  Not enough yet. The log began 2026-10-08 and records only")
        print("  CHANGES, so this fills at the pace real news happens -- a few")
        print("  a week. It cannot be backfilled: injury_flags was a snapshot")
        print("  with no history, which is why the log exists at all.")
        return
    p = props.copy()
    p["captured_at"] = pd.to_datetime(p.captured_at, utc=True, format="ISO8601")
    rows = []
    for ev in material.itertuples():
        team = ev.team
        after = p[(p.captured_at > ev.observed_at)
                  & (p.captured_at < ev.observed_at + pd.Timedelta(days=3))
                  & ((p.home_team == team) | (p.away_team == team))
                  & (p.note == "move")]
        for bk, d in after.groupby("book"):
            rows.append({"book": bk, "player_out": ev.player_name,
                         "hours": (d.captured_at.min() - ev.observed_at)
                                  .total_seconds() / 3600})
    if not rows:
        print("  no price moves found after any of those events")
        return
    r = pd.DataFrame(rows)
    print("\n  hours from designation change to that book's first move:")
    print(r.groupby("book").hours.agg(["count", "median", "min"]).round(1).to_string())
    print("\n  A book that is consistently slow is where a stale number sits.")


def main():
    mkt = None
    if "--market" in sys.argv:
        mkt = sys.argv[sys.argv.index("--market") + 1]

    props = pd.DataFrame(fetch_all("nfl_td_props"))
    results = pd.DataFrame(fetch_all("nfl_td_results"))
    injuries = pd.DataFrame(fetch_all("nfl_injury_log"))
    if props.empty:
        print("nfl_td_props is empty — run fetch_td_props.py")
        return
    if mkt:
        props = props[props.market == mkt]

    lh = pd.DataFrame(fetch_all("line_history", columns="game_id,total,recorded_at"))
    totals = {}
    if not lh.empty:
        lh = lh.dropna(subset=["total"]).sort_values("recorded_at")
        totals = dict(zip(lh.game_id, lh.total))

    print(f"{len(props)} price rows | {props.event_id.nunique()} games | "
          f"{len(results)} graded player-games | {len(injuries)} injury changes")

    board = board_with_devigs(props, totals)
    board = attach_results(board, results)

    section_devig(board)
    section_calibration(board)
    section_movement(props)
    section_repricing(props, injuries)
    print()


if __name__ == "__main__":
    main()
