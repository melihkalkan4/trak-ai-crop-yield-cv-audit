"""T7 — Per-crop feature selection (critical for n=213/209).

Per crop × tier:
 (a) drop one of any collinear pair |r|>0.9 (keep the more interpretable group:
     climate > index_mean > soil > topo > index_dist; tie-break by higher importance);
 (b) rank remaining by RandomForest impurity importance averaged over LOYO TRAINING folds only;
 (c) cap final count to k = max(5, n // 15) (≈ sane ratio for the sample).
Surviving sets reported as a FINDING (dropped features documented, not silently removed).
Output: outputs/selected_features.json, outputs/t7_selection_report.csv, outputs/t7_ratio.csv
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

KEYS = ["ilce_id", "ilce", "il", "year"]
GROUP_PRIORITY = {"climate": 0, "index_mean": 1, "soil": 2, "topo": 3, "index_dist": 4}


def _impute(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median()).fillna(0.0)


def fold_importance(X, y, groups):
    """RF impurity importance averaged over LOYO training folds (never sees test)."""
    imp = np.zeros(X.shape[1])
    logo = LeaveOneGroupOut()
    nf = 0
    for tr, _te in logo.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
        m.fit(X.iloc[tr].values, y[tr])
        imp += m.feature_importances_
        nf += 1
    return imp / max(nf, 1)


def group_of(feat, groups_def):
    for g in ("climate", "index_mean", "soil", "topo", "index_dist"):
        if feat in groups_def.get(g, []):
            return g
    return "index_dist"


def main():
    fg = json.loads((E.OUT / "feature_groups.json").read_text(encoding="utf-8"))
    selected, report, ratios = {}, [], []
    for crop in ("bugday", "aycicegi"):
        selected[crop] = {}
        gdef = fg[crop]
        for tier in ("A", "B", "C", "D"):
            f = E.OUT / f"tier_{tier}_{crop}.csv"
            if not f.exists():
                print(f"STOP: {f.name} missing — run T6 first."); return 2
            df = pd.read_csv(f)
            feats = [c for c in df.columns if c not in KEYS + ["verim_kg_da"]]
            X = _impute(df[feats].astype(float))
            y = df["verim_kg_da"].astype(float).values
            groups = df["year"].astype(int).values
            n = len(df)
            imp = pd.Series(fold_importance(X, y, groups), index=feats)

            # (a) collinearity drop
            corr = X.corr().abs()
            dropped = {}
            keep = list(feats)
            for i in range(len(feats)):
                for j in range(i + 1, len(feats)):
                    a, b = feats[i], feats[j]
                    if a not in keep or b not in keep:
                        continue
                    if corr.loc[a, b] > 0.9:
                        ga, gb = GROUP_PRIORITY[group_of(a, gdef)], GROUP_PRIORITY[group_of(b, gdef)]
                        # drop less interpretable (higher group prio num); tie -> lower importance
                        loser = (a if ga > gb else b if gb > ga else (a if imp[a] < imp[b] else b))
                        keep.remove(loser)
                        dropped[loser] = f"collinear(|r|={corr.loc[a,b]:.2f}) with {b if loser==a else a}"
            # (b)+(c) rank survivors, cap
            k = max(5, n // 15)
            ranked = imp[keep].sort_values(ascending=False)
            final = list(ranked.index[:k])
            selected[crop][tier] = final
            for fcol in feats:
                report.append(dict(crop=crop, tier=tier, feature=fcol,
                                   group=group_of(fcol, gdef), importance=round(float(imp[fcol]), 5),
                                   status=("selected" if fcol in final else
                                           ("dropped_collinear" if fcol in dropped else "dropped_lowrank")),
                                   note=dropped.get(fcol, "")))
            ratios.append(dict(crop=crop, tier=tier, n=n, n_candidate=len(feats),
                               n_selected=len(final), cap_k=k,
                               obs_per_feat=round(n / max(len(final), 1), 1),
                               risky=bool(n / max(len(final), 1) < 8)))
            print(f"[t7] {crop} tier {tier}: n={n} cand={len(feats)} -> selected={len(final)} "
                  f"(cap {k}, {n/max(len(final),1):.1f} obs/feat)", flush=True)

    (E.OUT / "selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    pd.DataFrame(report).to_csv(E.OUT / "t7_selection_report.csv", index=False)
    pd.DataFrame(ratios).to_csv(E.OUT / "t7_ratio.csv", index=False)
    print("[t7] saved selected_features.json + selection report + ratio table", flush=True)
    print("\n=== surviving features per crop/tier ===", flush=True)
    for crop in selected:
        for tier in ("A", "B", "C", "D"):
            print(f"  {crop} {tier}: {selected[crop][tier]}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
