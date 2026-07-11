"""R10 — Hyperparameter / tuning protocol (#4).

Decision: hyperparameters are FIXED and pre-specified (literature/reasonable defaults),
identical across crops, tiers, CV regimes and folds; test-fold results NEVER informed them
→ there is no information leakage from tuning. (This is the 'simple, pre-registered' option.)
Verified verbatim in src/cp25/05,06,07. Stacking meta-learner uses internal cv=3 to produce
out-of-fold base predictions (no leakage into the meta-model).

Emits the protocol as a table (no value is invented; all read from source).
Output: revisions/hyperparameter_protocol.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402

ROWS = [
    dict(model="PLS", search_range="none (fixed)", inner_cv="none", opt_metric="n/a",
         selected="n_components=3, scale=True", scaler="StandardScaler(train-fold)",
         oof_stacking="n/a"),
    dict(model="ElasticNet", search_range="none (fixed)", inner_cv="none", opt_metric="n/a",
         selected="alpha=1.0, l1_ratio=0.5, max_iter=10000", scaler="StandardScaler(train-fold)",
         oof_stacking="n/a"),
    dict(model="RandomForest", search_range="none (fixed)", inner_cv="none", opt_metric="n/a",
         selected="n_estimators=300, max_depth=5", scaler="none", oof_stacking="n/a"),
    dict(model="XGBoost", search_range="none (fixed)", inner_cv="none", opt_metric="n/a",
         selected="n_estimators=200, max_depth=4, learning_rate=0.05", scaler="none",
         oof_stacking="n/a"),
    dict(model="GPR", search_range="none (fixed)", inner_cv="none", opt_metric="n/a",
         selected="Matern(nu=2.5)+WhiteKernel, normalize_y=True, alpha=1e-4",
         scaler="StandardScaler(train-fold)", oof_stacking="n/a"),
    dict(model="Stacking (tier C, LOYO only)", search_range="none (fixed)", inner_cv="cv=3 (OOF base preds)",
         opt_metric="n/a", selected="base=[RF,XGB,GPR], meta=Ridge(alpha=1.0)",
         scaler="StandardScaler(train-fold)", oof_stacking="YES (cv=3 cross_val_predict)"),
]


def main():
    df = pd.DataFrame(ROWS)
    df.insert(0, "seed", 42)
    df.insert(0, "protocol", "fixed pre-specified (no test-set tuning); leakage-free")
    df.to_csv(R.REV / "hyperparameter_protocol.csv", index=False)
    print(f"[save] hyperparameter_protocol.csv rows={len(df)}", flush=True)
    print(df[["model", "selected", "oof_stacking"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
