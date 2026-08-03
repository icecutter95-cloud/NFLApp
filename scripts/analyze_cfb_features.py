"""
Audit the CFB feature mix: what earns its place, and what is dead weight?

Adding 46 form features improved nothing measurable -- movement correlation
actually fell slightly. That is the signature of redundancy or noise rather
than information, so before reaching for MORE features it is worth asking
which of the current ones do any work.

Three checks:
  1. Gain-based importance, aggregated to FAMILIES rather than columns, since
     46 columns come from 23 base metrics x 2 rolling windows and reading them
     individually is noise.
  2. Redundancy -- L4 against L8 of the same metric is close to a duplicate,
     and duplicated inputs dilute a tree model's splits without adding signal.
  3. Ablation. Drop a family, retrain, and see whether the margin model gets
     worse. A family that costs nothing when removed is not paying rent.

Judged on margin prediction, because that is what the previous test showed we
are worst at relative to the market.

Usage:
    python analyze_cfb_features.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR
from train_models import train_model
from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS

TRAIN = [2020, 2021, 2022, 2023]
TEST = [2024, 2025]

FAMILIES = {
    "core_epa":    ["off_ppa_adj", "def_ppa_adj", "off_sr_adj", "def_sr_adj"],
    "explosive":   ["off_expl", "def_expl"],
    "trench":      ["off_power", "def_power", "off_stuff", "def_stuff",
                    "off_line_yds", "def_line_yds", "off_2nd_lvl", "off_open_field"],
    "situational": ["off_sd_ppa", "off_pd_ppa", "def_sd_ppa", "def_pd_ppa",
                    "off_sd_sr", "off_pd_sr"],
    "tendency":    ["off_rush_ppa", "off_pass_ppa", "off_plays"],
    "preseason":   PRESEASON_COLS,
    "context":     ["games_played", "rest_days"],
}
STANDALONE = ["neutral_site", "conference_game", "travel_miles", "elev_change", "is_dome"]


def cols_for(family_keys):
    out = []
    for base in family_keys:
        if base in PRESEASON_COLS or base in ("games_played", "rest_days"):
            out.append(f"diff_{base}")
        else:
            out += [f"diff_{base}_L{n}" for n in (4, 8)]
    return out


def main():
    df = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    allf = [f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS
            + ["games_played", "rest_days"]] + STANDALONE
    allf = [c for c in allf if c in df.columns]
    tr, te = df[df.season.isin(TRAIN)], df[df.season.isin(TEST)]
    print(f"train {len(tr)} / test {len(te)} games, {len(allf)} features")

    base = train_model(tr[allf], tr["home_margin"], tr, "fa_all")
    p = base.predict(te[allf].fillna(0))
    base_corr = np.corrcoef(p, te["home_margin"])[0, 1]
    print(f"baseline corr {base_corr:+.4f}  (market {np.corrcoef(-te.closing_spread_home, te.home_margin)[0,1]:+.4f})")

    # ---- 1. importance by family
    imp = getattr(base, "feature_importances_", None)
    if imp is not None:
        s = pd.Series(imp, index=allf)
        print("\n1. GAIN SHARE BY FAMILY")
        rows = []
        for fam, keys in FAMILIES.items():
            cs = [c for c in cols_for(keys) if c in s.index]
            rows.append((fam, s[cs].sum() * 100, len(cs)))
        for c in STANDALONE:
            if c in s.index:
                rows.append((c, s[c] * 100, 1))
        for fam, pct, n in sorted(rows, key=lambda r: -r[1]):
            bar = "#" * int(pct / 2)
            print(f"  {fam:<14}{n:>3} cols{pct:>7.1f}%  {bar}")

    # ---- 2. redundancy: L4 vs L8
    print("\n2. REDUNDANCY (corr between L4 and L8 of the same metric)")
    hi = []
    for base_col in FEATURE_COLS:
        if base_col.endswith("_L4"):
            l8 = base_col[:-3] + "_L8"
            a, b = f"diff_{base_col}", f"diff_{l8}"
            if a in df and b in df:
                r = df[[a, b]].corr().iloc[0, 1]
                hi.append((base_col[:-3], r))
    hi.sort(key=lambda x: -abs(x[1]))
    for nm, r in hi[:6]:
        print(f"  {nm:<18}{r:+.3f}")
    print(f"  median across {len(hi)} metrics: {np.median([abs(r) for _, r in hi]):.3f}")

    # ---- 3. ablation
    print("\n3. ABLATION (drop a family, retrain; negative delta = family helps)")
    print(f"  {'family':<14}{'corr':>9}{'delta':>9}")
    for fam, keys in FAMILIES.items():
        drop = set(cols_for(keys))
        keep = [c for c in allf if c not in drop]
        if len(keep) < 5:
            continue
        m = train_model(tr[keep], tr["home_margin"], tr, f"fa_no_{fam}")
        cr = np.corrcoef(m.predict(te[keep].fillna(0)), te["home_margin"])[0, 1]
        print(f"  -{fam:<13}{cr:>+9.4f}{cr - base_corr:>+9.4f}"
              f"{'   <- removing HELPS' if cr > base_corr else ''}")

    # ---- 4. a deliberately small model, as a check on bloat
    lean = cols_for(FAMILIES["core_epa"]) + cols_for(FAMILIES["preseason"]) + STANDALONE
    lean = [c for c in lean if c in df.columns]
    m = train_model(tr[lean], tr["home_margin"], tr, "fa_lean")
    cr = np.corrcoef(m.predict(te[lean].fillna(0)), te["home_margin"])[0, 1]
    print(f"\n4. LEAN MODEL ({len(lean)} features: core EPA + preseason + context)")
    print(f"   corr {cr:+.4f} vs full {base_corr:+.4f} ({cr - base_corr:+.4f})")


if __name__ == "__main__":
    main()
