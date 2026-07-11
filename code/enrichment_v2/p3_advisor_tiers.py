"""P3 — Advisor-aligned tiers (crop-specific RS) + per-crop selection + ineffective-feature finding.

Tiers (advisor): A=climate; B=climate+{NDVI,NDRE,EVI} crop-specific window means; C=B+soil;
D=C+{NDVI,NDRE,EVI} phenological metrics (median/std/CV/P10/P90/range). RS comes from the
crop-SPECIFIC masks (P2). Per-crop selection (collinearity + LOYO-train RF importance + count cap)
→ best RS list per crop + which variables are ineffective (reported, per advisor).
Outputs: advisor_tier_{A,B,C,D}_<crop>.csv, advisor_selected_features.json,
advisor_selection_report.csv, advisor_percrop_rs_list.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

LAYERC = E.PROJECT_ROOT / "data" / "processed" / "calibration_features_layerC.csv"
CROP_FULL = {"bugday": "bugday", "aycicegi": "aycicegi_yaglik"}
CLIMATE = ["gdd_cum_season", "gdd_flowering", "vernalization_days", "tp_season_sum",
           "tp_winter_sum", "tp_flowering", "tp_grain_fill", "aridity_index", "heat_stress_days",
           "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean", "ssr_flowering_sum", "ssr_season_sum"]
IDX = ["NDVI", "NDRE", "EVI"]
PHENO = ["median", "stdDev", "p10", "p90", "cv", "range"]
KEYS = ["ilce_id", "ilce", "il", "year"]


def _impute(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median()).fillna(0.0)


def fold_importance(X, y, groups):
    imp = np.zeros(X.shape[1]); nf = 0
    for tr, _ in LeaveOneGroupOut().split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1).fit(X.iloc[tr].values, y[tr])
        imp += m.feature_importances_; nf += 1
    return imp / max(nf, 1)


def select(X, y, groups, n, groups_of):
    feats = list(X.columns)
    imp = pd.Series(fold_importance(X, y, groups), index=feats)
    corr = X.corr().abs()
    keep, dropped = list(feats), {}
    prio = {"climate": 0, "idx_mean": 1, "soil": 2, "idx_pheno": 3}
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = feats[i], feats[j]
            if a not in keep or b not in keep:
                continue
            if corr.loc[a, b] > 0.9:
                ga, gb = prio[groups_of[a]], prio[groups_of[b]]
                loser = a if ga > gb else b if gb > ga else (a if imp[a] < imp[b] else b)
                keep.remove(loser); dropped[loser] = f"collinear({corr.loc[a,b]:.2f})"
    k = max(5, n // 15)
    ranked = imp[keep].sort_values(ascending=False)
    return list(ranked.index[:k]), imp, dropped


def main():
    panel = pd.read_csv(LAYERC)
    soil = pd.read_csv(E.OUT / "soil_features.csv")
    soil_feat = [c for c in soil.columns if c.endswith("_0_30_mean") or c == "awc_0_30"]
    selected, report, rslist = {}, [], []
    for crop in ("bugday", "aycicegi"):
        idx = pd.read_csv(E.OUT / f"crop_specific_indices_{crop}.csv")
        wins = list(E.CROP_WINDOWS[crop].keys())
        idx_mean = [f"{i}_{w}_mean" for i in IDX for w in wins]
        idx_pheno = [f"{i}_{w}_{m}" for i in IDX for w in wins for m in PHENO]
        ph = panel[panel.crop == CROP_FULL[crop]][KEYS + CLIMATE + ["verim_kg_da"]]
        base = ph.merge(idx, on=KEYS, how="inner").merge(soil.drop(columns=["ilce", "il"]), on="ilce_id", how="left")
        tiers = {"A": CLIMATE, "B": CLIMATE + idx_mean, "C": CLIMATE + idx_mean + soil_feat,
                 "D": CLIMATE + idx_mean + soil_feat + idx_pheno}
        grp_of = ({f: "climate" for f in CLIMATE} | {f: "idx_mean" for f in idx_mean} |
                  {f: "soil" for f in soil_feat} | {f: "idx_pheno" for f in idx_pheno})
        selected[crop] = {}
        for t, feats in tiers.items():
            feats = [f for f in feats if f in base.columns]
            base[KEYS + ["verim_kg_da"] + feats].to_csv(E.OUT / f"advisor_tier_{t}_{crop}.csv", index=False)
            X = _impute(base[feats].astype(float)); y = base.verim_kg_da.values; g = base.year.astype(int).values
            sel, imp, dropped = select(X, y, g, len(base), grp_of)
            selected[crop][t] = sel
            for f in feats:
                report.append(dict(crop=crop, tier=t, feature=f, group=grp_of[f],
                                   importance=round(float(imp[f]), 5),
                                   status="selected" if f in sel else ("dropped_collinear" if f in dropped else "dropped_lowrank")))
        # per-crop RS finding: which RS indices survive in tier D (effective) vs not
        d_sel = selected[crop]["D"]
        rs_in_D = [f for f in d_sel if grp_of.get(f) in ("idx_mean", "idx_pheno")]
        eff_indices = sorted(set(f.split("_")[0] for f in rs_in_D))
        all_idx = set(IDX)
        rslist.append(dict(crop=crop, n=len(base), effective_RS_indices=", ".join(eff_indices),
                           ineffective_RS_indices=", ".join(sorted(all_idx - set(eff_indices))) or "(none)",
                           selected_RS_features="; ".join(rs_in_D)))
        print(f"[p3] {crop}: n={len(base)} | tier sizes A/B/C/D = "
              f"{[len(selected[crop][t]) for t in 'ABCD']} | effective RS: {eff_indices}", flush=True)
    (E.OUT / "advisor_selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    pd.DataFrame(report).to_csv(E.OUT / "advisor_selection_report.csv", index=False)
    pd.DataFrame(rslist).to_csv(E.OUT / "advisor_percrop_rs_list.csv", index=False)
    print("[p3] saved advisor tiers + selection + per-crop RS list", flush=True)


if __name__ == "__main__":
    main()
