"""
Do specialised models beat one global model?

The mixture-of-experts idea is that one model cannot capture every game state --
early season differs from late, short spreads from long, stable quarterback
situations from uncertain ones. Fit an expert per regime, let a gate decide the
weights.

Before building a gate, test the cheap version of the same hypothesis: fit
SEPARATE residual models per regime and see whether they beat a single global
model on identical out-of-sample games. If hard specialisation does not help,
soft specialisation will not either, and a gating network only adds parameters.

Our own results predict failure:
  * subset discovery found nothing surviving max-statistic permutation (p=0.22)
  * ensemble dispersion (3.01) exceeds mean prediction (2.83), so per-game
    confidence -- which a gate must estimate -- does not exist
  * the residual model's win rate is flat from 55.2% to 57.9% across every
    threshold, the signature of a UNIFORM edge

Stating that in advance so the test is a check on a prediction rather than a
search for a story afterwards.

Regimes tested (each partitions the games, so volume is comparable):
    week     1-4 / 5-12 / 13+          information accumulates through a season
    spread   <3 / 3-7 / 7+             different game types entirely
    total    low / high                scoring environment

Usage:
    python test_experts.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import SPREAD_FEATURES
from train_models import train_model

OPEN = "week_open_spread_home"
BAR = 1.0


def load():
    from model_line_movement import load_joined
    d = load_joined().dropna(subset=[OPEN, "home_margin"]).copy()
    d["target"] = d["home_margin"] + d[OPEN]
    return d, [c for c in SPREAD_FEATURES if c in d.columns]


def regime_week(d):
    return np.where(d["week"] <= 4, "early", np.where(d["week"] <= 12, "mid", "late"))


def regime_spread(d):
    a = d[OPEN].abs()
    return np.where(a < 3, "short", np.where(a <= 7, "mid", "long"))


def regime_total(d):
    if "week_open_total" not in d:
        return np.array(["all"] * len(d))
    med = d["week_open_total"].median()
    return np.where(d["week_open_total"] <= med, "lowtot", "hightot")


def grade(s):
    t = s["target"].values
    live = np.abs(t) > 1e-9
    won = np.where(s["pred"].values > 0, t > 0, t < 0)[live]
    return int(won.sum()), int((~won).sum())


def walk_global(df, feats):
    f = feats + [OPEN]
    out = []
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        m = train_model(tr[f], tr["target"], tr, f"g_{S}")
        t = te.copy()
        t["pred"] = m.predict(te[f].fillna(0))
        out.append(t)
    return pd.concat(out, ignore_index=True)


def walk_experts(df, feats, regime_fn, name):
    """One model per regime, trained only on that regime's history."""
    f = feats + [OPEN]
    df = df.copy()
    df["_reg"] = regime_fn(df)
    out = []
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr_all, te_all = df[df.season < S], df[df.season == S]
        if len(tr_all) < 300:
            continue
        for r in sorted(df["_reg"].unique()):
            tr, te = tr_all[tr_all._reg == r], te_all[te_all._reg == r]
            # An expert needs enough history to be an expert.
            if len(tr) < 150 or len(te) == 0:
                continue
            m = train_model(tr[f], tr["target"], tr, f"e_{name}_{r}_{S}")
            t = te.copy()
            t["pred"] = m.predict(te[f].fillna(0))
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(label, ev, restrict_index=None):
    if restrict_index is not None:
        ev = ev.loc[ev.index.intersection(restrict_index)]
    q = ev["pred"].abs() >= BAR
    s = ev[q]
    w, l = grade(s)
    n = max(w + l, 1)
    print(f"  {label:<34}{w+l:>6}{f'{w}-{l}':>12}{w/n*100:>7.1f}%"
          f"{(w*(100/110)-l)/n*100:>+8.1f}%")
    return w, l


def main():
    df, feats = load()
    print(f"NFL {len(df)} games, {len(feats)} features, bar |pred| >= {BAR}")

    g = walk_global(df, feats)
    print(f"\n  {'model':<34}{'n':>6}{'W-L':>12}{'win%':>7}{'ROI':>9}")
    gw, gl = report("GLOBAL (one model)", g)

    for fn, name in [(regime_week, "week"), (regime_spread, "spread"),
                     (regime_total, "total")]:
        e = walk_experts(df, feats, fn, name)
        if e.empty:
            continue
        report(f"experts by {name}", e)

    # Per-regime detail, to see whether ANY single expert beats the global model
    # on its own slice -- specialisation could help in one place and hurt overall.
    print("\n  per-regime, expert vs global on the SAME games:")
    for fn, name in [(regime_week, "week"), (regime_spread, "spread")]:
        e = walk_experts(df, feats, fn, name)
        if e.empty:
            continue
        e["_reg"] = fn(e)
        g2 = g.copy()
        g2["_reg"] = fn(g2)
        for r in sorted(e["_reg"].unique()):
            es, gs = e[e._reg == r], g2[g2._reg == r]
            ew, el = grade(es[es["pred"].abs() >= BAR])
            gw2, gl2 = grade(gs[gs["pred"].abs() >= BAR])
            en, gn = max(ew + el, 1), max(gw2 + gl2, 1)
            print(f"    {name}={r:<8} expert {ew/en*100:>5.1f}% (n={ew+el:>4})   "
                  f"global {gw2/gn*100:>5.1f}% (n={gw2+gl2:>4})   "
                  f"{ew/en*100 - gw2/gn*100:+.1f}")


if __name__ == "__main__":
    main()
