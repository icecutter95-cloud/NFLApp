"""
Predict LINE MOVEMENT rather than game outcomes.

The premise
----------
Three tiers of feature work established that we cannot beat the closing line at
predicting football: our best margin model reaches corr 0.383 against the
market's 0.477. But market movement is a different target. It is driven by
mechanical, partly-observable forces -- sharp money reacting to the same public
data we have, injury news, the public piling onto favourites. Predicting people
may be easier than predicting football.

Target: week_spread_movement = closing_spread_home - week_open_spread_home
        (spreads use the standard convention, negative = home favored, so
         movement < 0 means the line moved TOWARD the home team)

Design notes
------------
The opener IS a feature here, and that is correct rather than circular: the
target is movement, and `close` never appears among the inputs. But it does mean
a naive "fade extreme openers" mean-reversion strategy could explain any result,
so the model is benchmarked against exactly that:

    A  predict zero              -- no skill at all
    B  opener only               -- pure mean reversion
    C  opener + team features    -- the actual hypothesis

If C only matches B, our football features add nothing and the whole effect is
regression to the mean.

Evaluated primarily on CLOSING LINE VALUE, not win rate. CLV is far lower
variance -- it resolves in weeks rather than a season -- and it is what actually
determines long-run profitability: if you consistently get a better number than
the close, and the close is efficient (we showed it is), you are +EV.

Splits are strict: train 2020-2022, select on 2023-2024, hold out 2025.

Usage:
    python model_line_movement.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model

TRAIN = [2020, 2021, 2022]
SELECT = [2023, 2024]
HOLDOUT = [2025]


def load_joined() -> pd.DataFrame:
    """Team features joined to opening/closing lines, one row per game."""
    feats = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    oc["ct"] = pd.to_datetime(oc["commence_time"], utc=True)
    # NFL seasons span the new year, so anything before March belongs to the
    # previous season.
    oc["season"] = np.where(oc["ct"].dt.month >= 3, oc["ct"].dt.year, oc["ct"].dt.year - 1)
    oc = oc[oc["n_snapshots_week"] >= 2]

    keep = ["season", "home_team", "away_team", "week_open_spread_home",
            "closing_spread_home", "week_spread_movement"]
    j = feats.merge(oc[keep], on=["season", "home_team", "away_team"],
                    how="inner", suffixes=("", "_oc"))
    return j.dropna(subset=["week_open_spread_home", "week_spread_movement"])


def evaluate(name: str, pred: np.ndarray, actual: np.ndarray,
             opener: np.ndarray, margin: np.ndarray) -> dict:
    """Accuracy of the movement prediction, plus what it would be worth."""
    corr = float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 1e-9 else float("nan")
    mae = float(mean_absolute_error(actual, pred))

    # Bet the side the line is predicted to move toward. Predicted movement < 0
    # means the line is heading toward home, so we take home at the opener.
    bet_home = pred < 0
    moved = np.abs(actual) >= 0.5
    dir_acc = float(((pred < 0) == (actual < 0))[moved].mean()) if moved.any() else float("nan")

    # CLV in points: how much better our number is than the close.
    #   home bet -> open - close ;  away bet -> close - open
    clv = np.where(bet_home, -actual, actual)

    # Did the bet actually cover the number we took it at?
    surplus = margin + opener          # >0 means home covered the opener
    won = np.where(bet_home, surplus > 0, surplus < 0)
    live = np.abs(surplus) >= 0.01     # drop pushes
    w = int(won[live].sum()); l = int((~won[live]).sum())
    wr = w / max(w + l, 1)
    roi = (w * (100 / 110) - l) / max(w + l, 1) * 100

    return {"name": name, "corr": corr, "mae": mae, "dir_acc": dir_acc,
            "clv": float(clv.mean()), "clv_pos": float((clv > 0).mean()),
            "w": w, "l": l, "wr": wr * 100, "roi": roi}


def show(rows: list, title: str):
    print(f"\n{title}")
    print(f"  {'model':<26}{'corr':>7}{'MAE':>7}{'dir%':>7}{'CLV':>8}{'CLV+%':>7}"
          f"{'W-L':>11}{'win%':>7}{'ROI':>8}")
    for r in rows:
        print(f"  {r['name']:<26}{r['corr']:>+7.3f}{r['mae']:>7.2f}{r['dir_acc']*100:>6.1f}%"
              f"{r['clv']:>+8.2f}{r['clv_pos']*100:>6.1f}%"
              f"{r['w']:>5}-{r['l']:<5}{r['wr']:>6.1f}%{r['roi']:>+7.1f}%")


def train_and_save_production(df: pd.DataFrame):
    """Train on every season with line coverage and persist for live use.

    Evaluation already happened on strict splits above; the shipped artifact is
    trained on all available data because more history is better once the
    methodology is settled. Same pattern as the spread/total models.
    """
    import joblib
    from config import MODELS_DIR

    feats = [c for c in SPREAD_FEATURES if c in df.columns] + ["week_open_spread_home"]
    fit = df.dropna(subset=["week_spread_movement"])
    model = train_model(fit[feats], fit["week_spread_movement"], fit, "movement_production")

    path = MODELS_DIR / "movement_model.joblib"
    joblib.dump(model, path)
    # Persist the feature order too: the live scorer must build columns in
    # exactly this order, and silently mismatched columns is the failure mode
    # this pipeline has hit three times.
    joblib.dump(feats, MODELS_DIR / "movement_features.joblib")
    print(f"\nProduction movement model trained on {len(fit)} games "
          f"({sorted(fit.season.unique())})")
    print(f"  saved -> {path.name} ({len(feats)} features)")


def main():
    df = load_joined()
    print(f"joined games with line data: {len(df)}")
    for tag, seasons in [("train", TRAIN), ("select", SELECT), ("holdout", HOLDOUT)]:
        print(f"  {tag:<8} {seasons} -> {len(df[df.season.isin(seasons)])} games")

    team_feats = [c for c in SPREAD_FEATURES if c in df.columns]
    OPENER = "week_open_spread_home"
    TARGET = "week_spread_movement"

    tr = df[df.season.isin(TRAIN)]
    models = {}
    models["B opener only"] = (train_model(tr[[OPENER]], tr[TARGET], tr, "mv_open"), [OPENER])
    feats_c = team_feats + [OPENER]
    models["C opener + features"] = (train_model(tr[feats_c], tr[TARGET], tr, "mv_full"), feats_c)

    for tag, seasons in [("SELECT (2023-2024)", SELECT), ("HOLDOUT (2025)", HOLDOUT)]:
        ev = df[df.season.isin(seasons)]
        actual = ev[TARGET].values
        opener = ev[OPENER].values
        margin = ev["home_margin"].values
        rows = [evaluate("A predict zero", np.zeros(len(ev)), actual, opener, margin)]
        for name, (m, f) in models.items():
            rows.append(evaluate(name, m.predict(ev[f].fillna(0)), actual, opener, margin))
        show(rows, tag)

    # Does conviction help? Only meaningful if C beat the baselines above.
    m, f = models["C opener + features"]
    print("\nHOLDOUT by conviction (|predicted movement|):")
    ev = df[df.season.isin(HOLDOUT)]
    pred = m.predict(ev[f].fillna(0))
    for thr in [0, 0.5, 1.0, 1.5]:
        s = np.abs(pred) >= thr
        if s.sum() < 30:
            continue
        r = evaluate(f"  |pred|>={thr}", pred[s], ev[TARGET].values[s],
                     ev[OPENER].values[s], ev["home_margin"].values[s])
        print(f"    |pred|>={thr:<4} n={int(s.sum()):>3}  CLV {r['clv']:+.2f} pts  "
              f"CLV+ {r['clv_pos']*100:.0f}%  {r['w']}-{r['l']} = {r['wr']:.1f}%  ROI {r['roi']:+.1f}%")

    if "--save" in sys.argv:
        train_and_save_production(df)


if __name__ == "__main__":
    main()
