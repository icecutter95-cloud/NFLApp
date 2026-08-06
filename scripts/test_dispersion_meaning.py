"""
Does our model's dispersion measure GAME uncertainty, or just our own noise?

A previous test asked whether ensemble agreement ranks bets better than
prediction magnitude. It does not -- agreement correlates +0.85 with magnitude
and its incremental effect flips sign across bands. That killed dispersion as a
BET-RANKING signal.

This asks something different and, if it holds, more useful: is dispersion
measuring anything real about the game at all? There are three external checks
it should pass if it is:

  1. BOOK DISAGREEMENT   do the sportsbooks also find these games hard? We have
                         9 books quoted at the weekly opener for 2020-2025, so
                         their spread of opinion is observable. If our models
                         and the market are confused by the same games,
                         dispersion tracks something in the world.
  2. LINE MOVEMENT       uncertain games should move more between open and
                         close, as information arrives and gets priced.
  3. OUTCOME VARIANCE    uncertain games should have larger |opener error|,
                         i.e. be harder for anyone to call.

If dispersion fails all three, it is estimation noise -- an artefact of
resampling 1,300 training games -- and there is nothing more to chase. Mean
dispersion (3.01) already EXCEEDS mean prediction (2.83), which is what pure
noise would look like.

The practical payoff if it passes check 2: dispersion becomes an input to the
MOVEMENT model rather than a bet filter, which is a different job than the one
it already failed at.

Usage:
    python test_dispersion_meaning.py [n_models]
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model

OPEN = "week_open_spread_home"
MOVE = "week_spread_movement"
PANEL = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers",
         "pinnacle", "betonlineag", "lowvig", "bovada"]


def load():
    from model_line_movement import load_joined
    d = load_joined().dropna(subset=[OPEN, MOVE, "home_margin"]).copy()
    d["resid_open"] = d["home_margin"] + d[OPEN]
    return d, [c for c in SPREAD_FEATURES if c in d.columns]


def book_dispersion() -> pd.DataFrame:
    """Spread of opinion across sportsbooks at the weekly opener."""
    frames = []
    for s in range(2020, 2026):
        p = DATA_DIR / f"multibook_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).dropna(subset=["spread_home"])
    d = d[d.book.isin(PANEL)]
    d["req"] = pd.to_datetime(d["requested_at"], utc=True)
    g = (d.groupby(["req", "home_team", "away_team"])
         .agg(book_disp=("spread_home", "std"), n_books=("book", "nunique"))
         .reset_index())
    g = g[g.n_books >= 5]

    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    oc["req"] = pd.to_datetime(oc["week_opened_at"], utc=True)
    ct = pd.to_datetime(oc["commence_time"], utc=True)
    oc["season"] = np.where(ct.dt.month >= 3, ct.dt.year, ct.dt.year - 1)
    k = oc[["season", "home_team", "away_team", "req"]].merge(
        g, on=["home_team", "away_team", "req"], how="inner")
    return k.drop(columns=["req"]).drop_duplicates(
        subset=["season", "home_team", "away_team"])


def ensemble(df, feats, n_models=20, seed=9):
    """Walk-forward residual ensemble; keep the spread of predictions."""
    rng = np.random.default_rng(seed)
    out = []
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        preds = np.zeros((len(te), n_models))
        for i in range(n_models):
            rows = rng.integers(0, len(tr), len(tr))
            k = max(8, int(len(feats) * 0.7))
            sub = list(rng.choice(feats, size=k, replace=False)) + [OPEN]
            b = tr.iloc[rows]
            m = train_model(b[sub], b["resid_open"], b, f"dm_{S}_{i}")
            preds[:, i] = m.predict(te[sub].fillna(0))
        t = te.copy()
        t["mean_pred"] = preds.mean(axis=1)
        t["disp"] = preds.std(axis=1)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def band_table(ev, col, label, targets):
    print(f"\n  by {label} (quartiles)")
    q = pd.qcut(ev[col], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    hdr = "".join(f"{t:>18}" for t, _ in targets)
    print(f"    {'band':<10}{'n':>6}{hdr}")
    for b in ["Q1 low", "Q2", "Q3", "Q4 high"]:
        s = ev[q == b]
        cells = "".join(f"{fn(s):>18.2f}" for _, fn in targets)
        print(f"    {b:<10}{len(s):>6}{cells}")


def main():
    n_models = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    df, feats = load()
    ev = ensemble(df, feats, n_models)
    print(f"NFL {len(ev)} test games | ensemble of {n_models} residual models")
    print(f"  mean dispersion {ev.disp.mean():.2f}   "
          f"mean |prediction| {ev.mean_pred.abs().mean():.2f}")

    bd = book_dispersion()
    ev = ev.merge(bd, on=["season", "home_team", "away_team"], how="left")
    have = ev.book_disp.notna()
    print(f"  book dispersion available for {have.sum()}/{len(ev)} games")

    print("\nTHREE EXTERNAL CHECKS — does dispersion track anything real?")
    m = ev[have]
    c1 = np.corrcoef(m.disp, m.book_disp)[0, 1]
    c2 = np.corrcoef(ev.disp, ev[MOVE].abs())[0, 1]
    c3 = np.corrcoef(ev.disp, ev.resid_open.abs())[0, 1]
    print(f"  1. corr(model dispersion, BOOK dispersion)   {c1:+.3f}   n={len(m)}")
    print(f"  2. corr(model dispersion, |line movement|)   {c2:+.3f}")
    print(f"  3. corr(model dispersion, |opener error|)    {c3:+.3f}")

    # Control: is any of this just spread size in disguise?
    print(f"\n  control — corr(|spread|, each of the above):")
    a = ev[OPEN].abs()
    print(f"     |spread| vs model dispersion  "
          f"{np.corrcoef(a, ev.disp)[0,1]:+.3f}")
    print(f"     |spread| vs |line movement|   "
          f"{np.corrcoef(a, ev[MOVE].abs())[0,1]:+.3f}")
    if have.sum() > 50:
        print(f"     |spread| vs book dispersion   "
              f"{np.corrcoef(m[OPEN].abs(), m.book_disp)[0,1]:+.3f}")

    band_table(ev, "disp", "MODEL dispersion",
               [("|movement|", lambda s: s[MOVE].abs().mean()),
                ("|opener err|", lambda s: s.resid_open.abs().mean()),
                ("|spread|", lambda s: s[OPEN].abs().mean())])
    if have.sum() > 200:
        band_table(ev[have], "book_disp", "BOOK dispersion",
                   [("|movement|", lambda s: s[MOVE].abs().mean()),
                    ("model disp", lambda s: s.disp.mean()),
                    ("|spread|", lambda s: s[OPEN].abs().mean())])


if __name__ == "__main__":
    main()
