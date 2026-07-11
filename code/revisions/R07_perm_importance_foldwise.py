"""R07 — Fold-wise permutation importance (#13).

For each outer LOYO fold: fit on the training years, compute permutation importance ON THE
HELD-OUT test fold (metric = increase in RMSE when a feature is shuffled), then aggregate across
folds → mean + 95% CI. Model = RandomForest (robust, scale-free) for tiers A and C, both crops.
Importance is predictive contribution, NOT causation.

Output: revisions/permutation_importance_foldwise.csv
Columns: crop, tier, model, feature, imp_mean, imp_ci_low, imp_ci_high, metric, n_folds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402
rc = R.rc

MODEL = "random_forest"
N_REPEATS = 10


def run(tier, crop):
    sub = rc.crop_subset(rc.load_layer(tier), crop)
    feats = rc.FEATURES[tier]
    X = rc._impute(sub[feats].astype(float)).reset_index(drop=True)
    y = sub["verim_kg_da"].astype(float).values
    groups = sub["year"].astype(int).values
    logo = LeaveOneGroupOut()
    per_fold = {f: [] for f in feats}
    nf = 0
    for tr, te in logo.split(X, y, groups=groups):
        if len(te) < 3:
            continue
        m = rc._make_model(MODEL).fit(X.iloc[tr].values, y[tr])
        r = permutation_importance(m, X.iloc[te].values, y[te], n_repeats=N_REPEATS,
                                   random_state=42, scoring="neg_root_mean_squared_error")
        # importance = increase in RMSE = -(neg_rmse change) = importances_mean (sklearn already
        # reports drop in score; with neg_rmse score, a drop means RMSE increased → positive imp)
        for j, f in enumerate(feats):
            per_fold[f].append(float(r.importances_mean[j]))
        nf += 1
    rows = []
    for f in feats:
        vals = np.array(per_fold[f])
        if len(vals) == 0:
            continue
        # bootstrap CI over folds
        rng = np.random.default_rng(42)
        bs = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(2000)])
        rows.append(dict(crop=crop, tier=tier, model=MODEL, feature=f,
                         imp_mean=round(float(vals.mean()), 4),
                         imp_ci_low=round(float(np.percentile(bs, 2.5)), 4),
                         imp_ci_high=round(float(np.percentile(bs, 97.5)), 4),
                         metric="increase in RMSE (kg/da)", n_folds=nf))
    return rows


def main():
    out = []
    for tier in ("A", "C"):
        for crop in rc.CROPS:
            print(f"[imp] tier {tier} {crop} ...", flush=True)
            out += run(tier, crop)
    df = pd.DataFrame(out).sort_values(["crop", "tier", "imp_mean"], ascending=[True, True, False])
    df.to_csv(R.REV / "permutation_importance_foldwise.csv", index=False)
    print(f"[save] permutation_importance_foldwise.csv rows={len(df)}", flush=True)
    for (crop, tier), g in df.groupby(["crop", "tier"]):
        top = g.head(5)
        print(f"  {crop} tier{tier} top: " +
              ", ".join(f"{r.feature}({r.imp_mean:+.2f})" for r in top.itertuples()), flush=True)


if __name__ == "__main__":
    main()
