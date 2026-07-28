"""
Test whether shrinking the prior-season metrics toward league average --
based on offseason roster turnover and coaching change -- improves early-season
predictions beyond the flat k=3 blend.

Motivation
----------
calibrate_metric_blend.py fitted a blend that leans on last season's metrics
early in the year. But it treats every team's prior as equally trustworthy.
A team returning 91% of its snaps under the same coach (DEN) deserves more
faith in last year's identity than one returning 50% (MIA).

    prior_adj = lam * prior + (1 - lam) * league_mean
    lam       = (1 - alpha * (1 - continuity)) * (1 - beta * new_coach)

    alpha = 0, beta = 0  ->  lam = 1  ->  no shrinkage (current behaviour)

alpha and beta are swept, not guessed. Selection uses 2023-2024 only; 2025 is
held out to confirm, because a shrinkage parameter fit on thin early-season
data is exactly the kind of thing that can look good and mean nothing.

Continuity is computed by NAME join (snap counts <-> rosters): the pfr_id and
gsis crosswalk paths only match ~54% of snaps even against the SAME season's
roster, i.e. they are broken rather than measuring attrition. The name join
sanity-checks at 98.5%.

Usage:
    python calibrate_prior_trust.py
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from config import (DATA_DIR, SPREAD_FEATURES, VALIDATE_SEASONS, TEST_SEASON,
                    METRIC_BLEND_PSEUDO_COUNT, _TEAM_METRIC_COLS)
from train_models import load_split, train_model, ats_roi

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]   # roster-turnover shrinkage strength
BETA_GRID = [0.0, 0.1, 0.2, 0.3]           # extra shrinkage for a new head coach

CACHE = DATA_DIR / "prior_trust_cache.parquet"


def _norm_name(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower()
             .str.replace(r"[.']", "", regex=True)
             .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
             .str.strip())


def build_trust_inputs(seasons: list) -> pd.DataFrame:
    """continuity + new_coach per (team, season), cached to parquet."""
    if CACHE.exists():
        cached = pd.read_parquet(CACHE)
        if set(seasons).issubset(set(cached["season"].unique())):
            return cached

    rows = []
    for season in seasons:
        sc = nfl.import_snap_counts([season - 1])
        sc = sc[sc["game_type"] == "REG"].copy()
        sc["k"] = _norm_name(sc["player"])

        r = nfl.import_seasonal_rosters([season]).copy()
        namecol = "player_name" if "player_name" in r.columns else "football_name"
        r["k"] = _norm_name(r[namecol])
        roster = r.groupby("team")["k"].apply(set).to_dict()

        parts = {}
        for side, col in [("off", "offense_snaps"), ("def", "defense_snaps")]:
            g = sc.groupby(["team", "k"], as_index=False)[col].sum()
            g = g[g[col] > 0]
            for team, sub in g.groupby("team"):
                keep = roster.get(team, set())
                share = sub[sub["k"].isin(keep)][col].sum() / max(sub[col].sum(), 1e-9)
                parts.setdefault(team, {})[side] = share

        for team, d in parts.items():
            rows.append({"team": team, "season": season,
                         "continuity": float(np.mean(list(d.values())))})

    cont = pd.DataFrame(rows)

    # Coaching change: coach in week 1 of season S vs final week of season S-1
    sched = nfl.import_schedules(sorted(set(seasons) | {s - 1 for s in seasons}))
    sched = sched[sched["game_type"] == "REG"]
    long = pd.concat([
        sched[["season", "week", "home_team", "home_coach"]]
            .rename(columns={"home_team": "team", "home_coach": "coach"}),
        sched[["season", "week", "away_team", "away_coach"]]
            .rename(columns={"away_team": "team", "away_coach": "coach"}),
    ]).dropna(subset=["coach"])

    first = long.sort_values("week").groupby(["team", "season"], as_index=False).first()
    last = long.sort_values("week").groupby(["team", "season"], as_index=False).last()
    last = last.rename(columns={"coach": "prev_coach"})
    last["season"] = last["season"] + 1

    coach = first.merge(last[["team", "season", "prev_coach"]], on=["team", "season"], how="left")
    coach["new_coach"] = (coach["coach"] != coach["prev_coach"]).astype(int)
    coach.loc[coach["prev_coach"].isna(), "new_coach"] = 0

    out = cont.merge(coach[["team", "season", "new_coach"]], on=["team", "season"], how="left")
    out["new_coach"] = out["new_coach"].fillna(0).astype(int)
    out.to_parquet(CACHE, index=False)
    return out


def prior_season_end(metrics: pd.DataFrame) -> pd.DataFrame:
    idx = metrics.groupby(["team", "season"])["week"].idxmax()
    end = metrics.loc[idx].copy()
    end["season"] = end["season"].astype(int) + 1   # applies TO the next season
    return end


def shrink_prior(prior: pd.DataFrame, trust: pd.DataFrame,
                 alpha: float, beta: float) -> pd.DataFrame:
    """prior_adj = lam*prior + (1-lam)*league_mean, per season."""
    out = prior.merge(trust, on=["team", "season"], how="left")
    out["continuity"] = out["continuity"].fillna(out["continuity"].mean() if out["continuity"].notna().any() else 1.0)
    out["new_coach"] = out["new_coach"].fillna(0)

    lam = (1.0 - alpha * (1.0 - out["continuity"])) * (1.0 - beta * out["new_coach"])
    lam = lam.clip(0.0, 1.0)

    bases = [b for b in _TEAM_METRIC_COLS if b in out.columns]
    for season, grp in out.groupby("season"):
        means = grp[bases].mean()
        for b in bases:
            out.loc[grp.index, b] = lam[grp.index] * grp[b] + (1 - lam[grp.index]) * means[b]

    return out.drop(columns=["continuity", "new_coach"])


def blend_with_prior(df: pd.DataFrame, prior: pd.DataFrame, k: float) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    n = (out["week"].astype(float) - 1.0).clip(lower=0.0)

    for side in ("home", "away"):
        bases = [b for b in _TEAM_METRIC_COLS
                 if f"{b}_{side}" in out.columns and b in prior.columns]
        if not bases:
            continue
        pdf = prior[["team", "season"] + bases].rename(
            columns={"team": f"{side}_team", **{b: f"__p_{b}" for b in bases}})
        out = out.merge(pdf, on=[f"{side}_team", "season"], how="left")

        for b in bases:
            col, pcol = f"{b}_{side}", f"__p_{b}"
            cur, pri = out[col], out[pcol]
            denom = n + k
            raw = (n * cur.fillna(0.0) + k * pri.fillna(0.0)) / denom.replace(0.0, np.nan)
            out[col] = pd.Series(
                np.where(cur.isna() & pri.notna(), pri,
                         np.where(pri.isna() & cur.notna(), cur, raw)), index=out.index)
            out.drop(columns=[pcol], inplace=True)
    return out


def score(preds, actuals, edge_threshold: float = 1.5) -> dict:
    preds, actuals = np.asarray(preds, float), np.asarray(actuals, float)
    m = (~np.isnan(preds)) & (~np.isnan(actuals))
    preds, actuals = preds[m], actuals[m]
    if len(preds) < 2:
        return {"n": 0, "corr": np.nan, "wr": np.nan, "roi": np.nan, "bets": 0}
    r = ats_roi(preds, actuals, edge_threshold=edge_threshold)
    w, l = r["wins"], r["losses"]
    return {"n": len(preds), "corr": np.corrcoef(preds, actuals)[0, 1],
            "wr": w / max(w + l, 1) * 100, "roi": r["roi_pct"], "bets": w + l}


def main():
    eval_seasons = VALIDATE_SEASONS + [TEST_SEASON]
    train_seasons = list(range(2018, min(eval_seasons)))
    k = METRIC_BLEND_PSEUDO_COUNT

    print("=" * 84)
    print("PRIOR-TRUST CALIBRATION (roster continuity + coaching change)")
    print(f"  blend k={k:g} held fixed;  train={train_seasons}  eval={eval_seasons}")
    print("=" * 84)

    print("\nBuilding continuity / coach-change inputs (first run downloads + caches)...")
    trust = build_trust_inputs(eval_seasons)
    print(f"  {len(trust)} team-seasons  |  new coaches: {int(trust['new_coach'].sum())}")
    print(f"  continuity: min={trust['continuity'].min():.3f} "
          f"max={trust['continuity'].max():.3f} std={trust['continuity'].std():.3f}")

    metrics = pd.read_parquet(DATA_DIR / "team_metrics_all.parquet")
    prior_raw = prior_season_end(metrics)
    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")

    X_tr, y_tr, df_tr = load_split(train_seasons, SPREAD_FEATURES, "home_cover_surplus")
    model = train_model(X_tr, y_tr, df_tr, "prior_trust")
    fc = model.get_booster().feature_names

    def subset(seasons):
        s = full[full["season"].isin(seasons)].copy()
        return s[s["home_cover_surplus"].notna()].reset_index(drop=True)

    def eval_combo(df, alpha, beta, weeks):
        pri = shrink_prior(prior_raw, trust, alpha, beta)
        b = blend_with_prior(df, pri, k)
        p = model.predict(b[fc].fillna(0))
        m = b["week"].isin(weeks).values
        return score(p[m], b["home_cover_surplus"].values[m])

    # ---- select on 2023-2024 only ----
    sel = subset([2023, 2024])
    early = [1, 2, 3, 4]
    print("\nSELECTION on 2023-2024 (weeks 1-4 ROI); alpha=roster, beta=coach")
    print(f"  {'alpha':>6} | " + " | ".join(f"beta={b:<4g}" for b in BETA_GRID))
    print("  " + "-" * 52)
    best = (-1e9, None)
    for a in ALPHA_GRID:
        cells = []
        for bta in BETA_GRID:
            s = eval_combo(sel, a, bta, early)
            cells.append(f"{s['roi']:+6.1f}%  ")
            if s["roi"] > best[0]:
                best = (s["roi"], (a, bta))
        tag = "  <- current" if a == 0 else ""
        print(f"  {a:>6g} | " + " | ".join(cells) + tag)

    ba, bb = best[1]
    print(f"\n-> selected alpha={ba:g}, beta={bb:g}  (2023-24 wk1-4 ROI {best[0]:+.1f}%)")

    # ---- confirm on 2025 holdout ----
    ho = subset([2025])
    print("\nCONFIRM on 2025 holdout (never used for selection):")
    for label, (a, bta) in [("baseline (alpha=0,beta=0)", (0.0, 0.0)),
                            (f"tuned (alpha={ba:g},beta={bb:g})", (ba, bb))]:
        for wlabel, weeks in [("wk1-4", early), ("ALL", list(range(1, 23)))]:
            s = eval_combo(ho, a, bta, weeks)
            print(f"   {label:<28} {wlabel:<6} n={s['n']:>3}  corr={s['corr']:+.3f}  "
                  f"win={s['wr']:5.1f}%  ROI={s['roi']:+6.1f}%  bets={s['bets']}")


if __name__ == "__main__":
    main()
