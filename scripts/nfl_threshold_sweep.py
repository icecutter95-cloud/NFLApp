"""
Can the NFL gates be loosened for more volume? Swept, then priced for the sweep.

THE QUESTION
The two gates leave only ~16 NFL picks a season. Asked 2026-11-12: are there
looser cells -- 1 point of movement instead of 1.5, a different ticket bar --
that keep the hit rate but double the volume? A 42-cell grid will always
produce some elevated cells on 88 picks, so the sweep alone cannot answer it.
Two things are needed beside it: a permutation test that prices in having
searched the grid, and an out-of-sample check on whatever dimension has one.
The permutation is seeded, so every number in this docstring reproduces.

WHAT THE GRID SHOWED (2026 NFL qualifying picks, n=88)
Smooth and monotone in both directions, which normally argues against noise:

    move >=      any tix   <60     <55     <50     <45
    any           50%(88)  54%(50) 52%(46) 49%(39) 46%(28)
    >=0           55%(58)  61%(33) 61%(31) 59%(27) 56%(18)
    >=1.0         53%(36)  65%(20) 63%(19) 67%(18) 54%(13)
    >=1.5         57%(30)  72%(18) 71%(17) 75%(16) 64%(11)
    >=2.5         59%(22)  77%(13) 75%(12) 82%(11) 71% (7)

So yes, looser cells exist: move>=0/tix<60 is 61% on 33 picks, double the
volume of the current gate, and move>=1.0/tix<50 is 67% on 18.

WHY NEITHER IS WORTH TAKING
[1] Permutation over the grid, p = 0.048. Shuffle the win/loss labels, re-run
    the same search, take the best cell: that procedure averages 60.8% and
    reaches the observed 75.0% about 5% of the time. The headline cell sits
    at the 95th percentile of its own null (73.7%). Its naive binomial p of
    0.077 describes one cell chosen in advance, which is not what happened.
    42 cells in the grid, 28 of them big enough to be candidates.

[2] The movement dimension is flat out of sample at EVERY bar. On
    movement_history's select/holdout split the holdout sits at 51-56%
    against a 55.6% base rate, while the select period climbs 61.6 -> 72.1.
    That is the signature of a surface fitted to its own period, across the
    whole range. 1.5 was not a bad choice of threshold; no threshold works.

WHAT THAT LEAVES
The ungated NFL model hits 55.6% on the honest holdout (n=142), already clear
of the 52.38% break-even, and 2026's 50.0% is consistent with it --
P(<=44 of 88 | 55.6%) = 0.169. Per 88-pick season at -110:

    bet all 88 at the combined 53.5%             +1.8u
    bet all 88 at the holdout 55.6%              +5.5u
    gate to 16, gate is real (70%)               +5.4u
    gate to 16, gate adds nothing (53.5%)        +0.3u

Gating only matches betting everything IF the gate is real, and its one
honest test says it is not. The volume answer is "all of them", not a looser
cell -- with the gate used to SIZE rather than to select.

WHAT IS GENUINELY UNTESTED
Rule 2 is not falsified, it is unmeasurable: nfl_public_splits begins
2026-09-06, so no holdout for it exists or can be built. Rule 1 is the half
that IS falsified. The most interesting untested direction is money-tickets
divergence rather than the ticket level itself -- money>=tickets+10 is 6-1,
which is 7 picks and proves nothing, but it is a different claim from "the
public is on the other side" and deserves its own season.

See two_rule_filter_check.py for whether the gates are real at all, and
season_record.py for the plain record. This file is only about whether the
thresholds can be moved.

Usage:
    python scripts/nfl_threshold_sweep.py
    python scripts/nfl_threshold_sweep.py --shuffles 50000
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

from score_week import fetch_all, ts

MOVE_BARS = [None, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
TICKET_BARS = [None, 60.0, 55.0, 50.0, 45.0, 40.0]
MIN_CELL = 15          # cells smaller than this are not candidates in the test
SHUFFLES = 20000
VIG_RETURN = 100 / 110
BREAK_EVEN = 110 / 210
RNG = np.random.default_rng(20261112)


def rate(x):
    """(wins, losses, rate) over win/loss rows only."""
    w = int((x.result == "win").sum())
    l = int((x.result == "loss").sum())
    return w, l, (w / (w + l) if w + l else None)


def show(x, label, indent=2):
    w, l, r = rate(x)
    pad = " " * indent
    if r is None:
        print(f"{pad}{label:<34} --")
        return
    n = w + l
    units = w * VIG_RETURN - l
    print(f"{pad}{label:<34} {w:3d}-{l:<3d} {r:6.1%}  {units:+6.1f}u  "
          f"p={binomtest(w, n).pvalue:.3f}")


def nfl_frame():
    """2026 qualifying NFL picks with the last PRE-kickoff split attached.

    Money percentages are carried as well as tickets -- season_record's
    loader drops them, and the divergence gate below is the one alternative
    second rule that is not just a different ticket level.
    """
    t = pd.DataFrame(fetch_all("clv_tracking"))
    t = t[t.bet_type.isin(["spread", "total"])].copy()
    t["kick"] = ts(t.commence_time)

    sp = pd.DataFrame(fetch_all("nfl_public_splits"))
    sp["c"] = ts(sp.captured_at)
    m = sp.merge(t[["game_id", "bet_type", "kick"]],
                 on=["game_id", "bet_type"], how="inner")
    m = m[m.c < m.kick]
    cols = ["home_bets_pct", "away_bets_pct", "home_money_pct",
            "away_money_pct"]
    last = (m.sort_values("c").groupby(["game_id", "bet_type"]).tail(1)
            [["game_id", "bet_type"] + cols])

    d = t.merge(last, on=["game_id", "bet_type"], how="left")
    home = d.predicted_side.isin(["home", "over"])
    d["our_tix"] = np.where(home, d.home_bets_pct, d.away_bets_pct)
    d["our_money"] = np.where(home, d.home_money_pct, d.away_money_pct)
    d["divergence"] = d.our_money - d.our_tix
    return d[d.result.isin(["win", "loss"]) & d.predicted_side.notna()
             & (d.qualifies == True)]            # noqa: E712


def mask(d, move, tix):
    m = pd.Series(True, index=d.index)
    if move is not None:
        m &= d.clv_points >= move
    if tix is not None:
        m &= d.our_tix < tix
    return m


def grid(d):
    """The sweep itself. Prints rate and n for every cell."""
    print("\n" + "=" * 74)
    print(f"GRID -- {len(d)} graded NFL qualifying picks, 2026")
    print("=" * 74)
    head = "".join(f"{('any' if t is None else '<' + str(int(t))):>12}"
                   for t in TICKET_BARS)
    print(f"  {'move >=':<9}{head}")
    for mv in MOVE_BARS:
        row = f"  {('any' if mv is None else str(mv)):<9}"
        for tx in TICKET_BARS:
            w, l, r = rate(d[mask(d, mv, tx)])
            cell = "--" if r is None else f"{r:.0%}"
            row += f"{cell + ' (' + str(w + l) + ')':>12}"
        print(row)
    print(f"\n  cells with n < {MIN_CELL} are shown but are not treated as")
    print("  candidates in the permutation test below")


def permutation(d, shuffles):
    """Price in having searched the grid.

    The question a naive binomial answers is "how often does THIS cell beat
    chance". The question that was actually asked is "how often does the BEST
    OF 42 NESTED CELLS look this good", which is a different and much larger
    number. Shuffling the outcomes while holding the gate values fixed
    answers the second one: it destroys any real relationship but preserves
    the grid's shape, its nesting, and every cell's n.
    """
    print("\n" + "=" * 74)
    print(f"PERMUTATION TEST -- best cell against its own search, "
          f"{shuffles:,} shuffles")
    print("=" * 74)

    cells, obs_best, obs_label = [], -1.0, None
    for mv in MOVE_BARS:
        for tx in TICKET_BARS:
            idx = np.where(mask(d, mv, tx).values)[0]
            if len(idx) < MIN_CELL:
                continue
            cells.append(idx)
            r = (d.result.values[idx] == "win").mean()
            if r > obs_best:
                obs_best, obs_label = r, (mv, tx)

    wins = (d.result.values == "win")
    best = np.empty(shuffles)
    for i in range(shuffles):
        sh = RNG.permutation(wins)
        best[i] = max(sh[idx].mean() for idx in cells)

    p = float((best >= obs_best).mean())
    mv, tx = obs_label
    print(f"  candidate cells (n >= {MIN_CELL}): {len(cells)}")
    print(f"  best observed: {obs_best:.1%} at move >= {mv}, "
          f"tickets < {tx if tx else 'any'}")
    print(f"  under shuffled outcomes the best cell averages "
          f"{best.mean():.1%}")
    print(f"    50th pct {np.percentile(best, 50):.1%}   "
          f"95th pct {np.percentile(best, 95):.1%}   "
          f"99th pct {np.percentile(best, 99):.1%}")
    print(f"\n  p = {p:.3f}  -- a cell this good appears by chance at this "
          f"rate")
    if p > 0.01:
        print("  Not evidence the surface is real. The same grid search on")
        print("  noise finds something comparable about this often.")


def veto(d):
    """The other shape of the rule: drop the worst, keep everything else.

    A gate that keeps 16 of 88 picks is a selection rule. A gate that drops
    the picks the line ran AWAY from is a veto, and it preserves the volume
    the selection rule destroys -- so if the movement signal is real in any
    form, this is the form worth having.
    """
    print("\n" + "=" * 74)
    print("VETO FORM -- drop the worst, keep the rest")
    print("=" * 74)
    for bar in (-1.0, -2.0, -3.0):
        kept, dropped = d[d.clv_points > bar], d[d.clv_points <= bar]
        w, l, _ = rate(kept)
        show(kept, f"keep clv > {bar}  ({w + l}/{len(d)} picks)")
        show(dropped, "  the dropped set", indent=4)


def second_gates(d, move=1.5):
    """Alternatives to the ticket bar, at the current movement bar."""
    print("\n" + "=" * 74)
    print(f"ALTERNATIVE SECOND GATES -- all at move >= {move}")
    print("=" * 74)
    g = d[d.clv_points >= move]
    show(g, "no second gate")
    show(g[g.our_tix < 50], "tickets < 50  (current rule 2)")
    show(g[g.our_tix < 55], "tickets < 55")
    show(g[g.our_money < 50], "money < 50")
    show(g[g.divergence >= 5], "money - tickets >= +5")
    show(g[g.divergence >= 10], "money - tickets >= +10")
    print("\n  Divergence is a different claim from the ticket level: not")
    print("  'the public is elsewhere' but 'the money disagrees with the")
    print("  tickets'. On this n it is a hypothesis, not a result.")


def oos_movement():
    """The movement dimension swept across movement_history's honest split.

    This is the only out-of-sample test that exists for either gate. If the
    movement surface were real, the holdout column would rise with the bar
    the way the select column does.
    """
    print("\n" + "=" * 74)
    print("OUT OF SAMPLE -- movement bar swept on movement_history (2023-25)")
    print("=" * 74)
    m = pd.DataFrame(fetch_all("movement_history"))
    m = m[(m.qualifies == True) & m.result.isin(["win", "loss"])  # noqa: E712
          & m.clv_points.notna()].copy()

    print(f"  {'move >=':<10}{'select':>20}{'holdout':>20}")
    for bar in MOVE_BARS:
        cells = []
        for period in ("select", "holdout"):
            s = m[m.period == period]
            if bar is not None:
                s = s[s.clv_points >= bar]
            w, l, r = rate(s)
            cells.append("--" if r is None else f"{r:.1%} (n={w + l})")
        label = "any" if bar is None else str(bar)
        print(f"  {label:<10}{cells[0]:>20}{cells[1]:>20}")

    _, _, base = rate(m[m.period == "holdout"])
    print(f"\n  holdout base rate {base:.1%}. The select column rises with")
    print("  the bar and the holdout column does not move at any bar, which")
    print("  is overfitting across the whole dimension -- not a badly chosen")
    print("  1.5. Rule 2 cannot be tested here at all: nfl_public_splits")
    print("  begins 2026-09-06, so no pre-2026 ticket data exists.")
    return m


def framing(d, m):
    """What the ungated model is worth, which is the real answer on volume."""
    print("\n" + "=" * 74)
    print("SO WHAT IS THE VOLUME ACTUALLY WORTH")
    print("=" * 74)
    hw, hl, hr = rate(m[m.period == "holdout"])
    lw, ll, _ = rate(d)
    cw, cl = hw + lw, hl + ll
    cr = cw / (cw + cl)
    show(m[m.period == "holdout"], "2023-25 holdout, ungated")
    show(d, "2026 live, ungated")
    print(f"  {'combined':<34} {cw:3d}-{cl:<3d} {cr:6.1%}  "
          f"{cw * VIG_RETURN - cl:+6.1f}u  "
          f"p={binomtest(cw, cw + cl).pvalue:.3f}")

    n = lw + ll
    tail = binomtest(lw, n, hr, alternative="less").pvalue
    print(f"\n  P(<= {lw} wins of {n} | true rate {hr:.1%}) = {tail:.3f}")
    print("  2026 is a below-average stretch of the holdout model, not a")
    print("  different model.")

    gated = int(mask(d, 1.5, 50.0).sum())
    print(f"\n  expected units per {n}-pick season at -110")
    for label, k, r in (("bet all, combined rate", n, cr),
                        ("bet all, holdout rate", n, hr),
                        (f"gate to {gated}, gate real (70%)", gated, 0.70),
                        (f"gate to {gated}, gate adds nothing", gated, cr)):
        u = k * (r * VIG_RETURN - (1 - r))
        print(f"    {label:<34} {u:+6.1f}u")
    print("\n  Gating only TIES betting everything, and only if the gate is")
    print("  real. If it is not, it costs most of the edge. That asymmetry,")
    print("  not the sweep, is the argument for sizing instead of selecting.")


def main():
    shuffles = SHUFFLES
    if "--shuffles" in sys.argv:
        shuffles = int(sys.argv[sys.argv.index("--shuffles") + 1])
    print(f"break-even at -110 is {BREAK_EVEN:.2%}; "
          f"everything is graded at the OPENER")
    d = nfl_frame()
    grid(d)
    permutation(d, shuffles)
    veto(d)
    second_gates(d)
    m = oos_movement()
    framing(d, m)


if __name__ == "__main__":
    main()
