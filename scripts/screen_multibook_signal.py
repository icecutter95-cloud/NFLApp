"""
Does the cross-book panel add anything BEYOND the movement model we ship?

Correlating with line movement is not the bar. The disagreement signal for
totals correlates with movement at +0.32 and still loses money -- screen_totals_
signal.py shows it inverting on the holdout. The only question worth asking is
whether the panel improves the bets we would actually place.

Two distinct ways it could help, tested separately:

  A. As a MODEL FEATURE   -- consensus gap and dispersion added to the movement
                             model's inputs.
  B. As a FILTER          -- an extra condition on the existing qualifying rule:
                             only bet when the book panel leans the same way.

Both are judged on the betting metric under the same strict splits used
everywhere else: train 2020-2022, select 2023-2024, holdout 2025.

A note on the panel: book coverage grows over the sample (18 books in 2020, 27
by 2022), so consensus is computed over a FIXED list of books rather than
whatever happened to be quoting. Otherwise the feature partly measures which
books existed that year.

Usage:
    python screen_multibook_signal.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN, SELECT, HOLDOUT

# Books that are both liquid and consistently present. Offshore books with
# stale quotes blew the measured line-shopping value up to +1.28 pts -- pure
# max-of-N bias -- before this list was applied.
PANEL = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers",
         "pinnacle", "betonlineag", "lowvig", "bovada"]
MIN_BOOKS = 5


def build_panel() -> pd.DataFrame:
    frames = []
    for s in range(2020, 2026):
        p = DATA_DIR / f"multibook_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True).dropna(subset=["spread_home"])
    d = d[d.book.isin(PANEL)]
    d["req"] = pd.to_datetime(d["requested_at"], utc=True)

    g = d.groupby(["req", "home_team", "away_team"])
    p = g.agg(consensus=("spread_home", "median"),
              dispersion=("spread_home", "std"),
              n_books=("book", "nunique"),
              best_home=("spread_home", "max"),
              best_away=("spread_home", "min")).reset_index()
    dk = d[d.book == "draftkings"].set_index(["req", "home_team", "away_team"])["spread_home"]
    p = p.join(dk.rename("dk_panel"), on=["req", "home_team", "away_team"])
    p = p.dropna(subset=["dk_panel"])
    p = p[p.n_books >= MIN_BOOKS]
    # >0 means DK hangs a number above the field, so DK should drift DOWN.
    p["dk_vs_consensus"] = p["dk_panel"] - p["consensus"]
    return p


def grade(sel: pd.DataFrame, bet_home: np.ndarray) -> dict:
    surplus = sel["home_margin"].values + sel["week_open_spread_home"].values
    live = np.abs(surplus) > 1e-9
    won = np.where(bet_home, surplus > 0, surplus < 0)[live]
    w, l = int(won.sum()), int((~won).sum())
    n = max(w + l, 1)
    clv = np.where(bet_home, -sel["week_spread_movement"], sel["week_spread_movement"])
    return {"n": len(sel), "w": w, "l": l, "wr": w / n * 100,
            "roi": (w * (100 / 110) - l) / n * 100, "clv": float(np.mean(clv))}


def row(tag, r):
    print(f"    {tag:<34} n={r['n']:>3}  {r['w']:>3}-{r['l']:<3} "
          f"{r['wr']:>5.1f}%  ROI {r['roi']:>+6.1f}%  CLV {r['clv']:>+5.2f}")


def main():
    df = load_joined()
    panel = build_panel()

    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    oc["req"] = pd.to_datetime(oc["week_opened_at"], utc=True)
    ct = pd.to_datetime(oc["commence_time"], utc=True)
    # NFL seasons span the new year: anything before March belongs to the prior one.
    oc["season"] = np.where(ct.dt.month >= 3, ct.dt.year, ct.dt.year - 1)
    key = oc[["season", "home_team", "away_team", "req"]].merge(
        panel, on=["home_team", "away_team", "req"], how="inner")

    # Season MUST be part of the join. Merging on the team pair alone matches
    # every season's panel to every season's game with the same two teams --
    # it fanned 1,640 rows out to 4,656 and manufactured an 81% holdout.
    key = key.drop(columns=["req"]).drop_duplicates(subset=["season", "home_team", "away_team"])
    before = len(df)
    df = df.merge(key, on=["season", "home_team", "away_team"], how="left")
    assert len(df) == before, f"merge fanned out: {before} -> {len(df)}"
    have = df["dk_vs_consensus"].notna()
    print(f"games: {len(df)} | with a usable book panel: {have.sum()}")
    for s in sorted(df.season.unique()):
        m = df[df.season == s]
        print(f"  {s}: {m['dk_vs_consensus'].notna().sum():>3}/{len(m):>3}")

    df = df[have].copy()
    OPEN, TGT = "week_open_spread_home", "week_spread_movement"
    team = [c for c in SPREAD_FEATURES if c in df.columns]

    tr = df[df.season.isin(TRAIN)]
    base_f = team + [OPEN]
    new_f = base_f + ["dk_vs_consensus", "dispersion"]
    base = train_model(tr[base_f], tr[TGT], tr, "mv_base")
    new = train_model(tr[new_f], tr[TGT], tr, "mv_panel")

    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    mf = [c for c in SPREAD_FEATURES if c in full.columns]
    mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
    mm = train_model(mfit[mf], mfit["home_margin"], mfit, "margin")

    for tag, seasons in [("SELECT 2023-24", SELECT), ("HOLDOUT 2025", HOLDOUT)]:
        ev = df[df.season.isin(seasons)].copy()
        ev["mv_base"] = base.predict(ev[base_f].fillna(0))
        ev["mv_new"] = new.predict(ev[new_f].fillna(0))
        ev["dis"] = mm.predict(ev[mf].fillna(0)) + ev[OPEN].values

        print(f"\n{tag}   (n={len(ev)})")
        for nm in ["mv_base", "mv_new"]:
            c = np.corrcoef(ev[nm], ev[TGT])[0, 1]
            print(f"  corr({nm:<8}, movement) = {c:+.3f}")

        def qual(mvcol):
            return ((ev["dis"].abs() >= 3.0) & (ev[mvcol].abs() >= 0.5)
                    & ((ev["dis"] > 0) == (ev[mvcol] < 0)))

        # A: does the panel help as a model input?
        q = qual("mv_base"); row("current filter (baseline)", grade(ev[q], (ev[q].mv_base < 0).values))
        q = qual("mv_new");  row("A: panel as model feature", grade(ev[q], (ev[q].mv_new < 0).values))

        # B: does the panel help as an extra filter on the current model?
        base_q = qual("mv_base")
        for thr in [0.01, 0.25, 0.5]:
            # Panel must lean the same way the model expects the line to move.
            agree = (ev["dk_vs_consensus"].abs() >= thr) & \
                    ((ev["dk_vs_consensus"] > 0) == (ev["mv_base"] < 0))
            s = ev[base_q & agree]
            if len(s) >= 15:
                row(f"B: + panel agrees (gap>={thr})", grade(s, (s.mv_base < 0).values))


if __name__ == "__main__":
    main()
