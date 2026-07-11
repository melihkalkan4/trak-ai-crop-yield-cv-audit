"""R02 — Rolling-origin / expanding-window forward validation (#3, most important).

Genuine forward-in-time evaluation alongside (not replacing) LOYO. For each test year Y,
ONLY data with year < Y is used for imputation, standardization, model fit, and the
climatology baseline (per-ilce mean over training years). No post-Y information leaks.

Tier A (climate, 2004–2025): min 7 training years → test 2011..2025 (15 test years).
Tier B/C (NDVI, 2017+): run with a 'few test years' warning (short panel).

Output: revisions/rolling_origin_results.csv + revisions/rolling_origin_summary.csv
Columns: crop, tier, model, test_year, n, model_rmse, persistence_rmse, climatology_rmse,
         model_r2, skill_score_vs_clim
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402
rc = R.rc

MIN_TRAIN = {"A": 7, "B": 4, "C": 4}


def _impute_train(X_tr, X_te):
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)
    med = X_tr.median()
    med = med.fillna(0.0)
    return X_tr.fillna(med), X_te.fillna(med)


def run_tier_crop(tier, crop):
    sub = rc.crop_subset(rc.load_layer(tier), crop)
    feats = rc.FEATURES[tier]
    years = sorted(sub["year"].astype(int).unique())
    ymin = years[0]
    test_years = [y for y in years if y - ymin >= MIN_TRAIN[tier] and (sub["year"] == y).sum() >= 5]
    rows = []
    for Y in test_years:
        tr = sub[sub["year"].astype(int) < Y]
        te = sub[sub["year"].astype(int) == Y]
        if len(tr) < 30 or te.empty:
            continue
        Xtr, Xte = _impute_train(tr[feats].astype(float), te[feats].astype(float))
        ytr = tr["verim_kg_da"].astype(float).values
        yte = te["verim_kg_da"].astype(float).values
        # climatology: per-ilce mean over training years (fallback global train mean)
        ilce_mean = tr.groupby("ilce_id")["verim_kg_da"].mean()
        gmean = float(ytr.mean())
        clim = te["ilce_id"].map(ilce_mean).fillna(gmean).astype(float).values
        # persistence: same ilce previous year (Y-1) from full sub
        prev = sub[sub["year"].astype(int) == Y - 1].set_index("ilce_id")["verim_kg_da"]
        pers = te["ilce_id"].map(prev).astype(float).values
        pers_mask = ~np.isnan(pers)
        clim_rmse = float(np.sqrt(mean_squared_error(yte, clim)))
        pers_rmse = (float(np.sqrt(mean_squared_error(yte[pers_mask], pers[pers_mask])))
                     if pers_mask.sum() >= 2 else np.nan)
        for model in rc.MODELS_BY_LAYER[tier]:
            if model == "stacking":
                continue  # expensive; ablation/LOYO cover it
            if model in rc._NEEDS_SCALER:
                sc = StandardScaler().fit(Xtr)
                a, b = sc.transform(Xtr), sc.transform(Xte)
            else:
                a, b = Xtr.values, Xte.values
            m = rc._make_model(model).fit(a, ytr)
            pred = m.predict(b).ravel()
            mrmse = float(np.sqrt(mean_squared_error(yte, pred)))
            rows.append(dict(
                crop=crop, tier=tier, model=model, test_year=int(Y), n=len(te),
                model_rmse=round(mrmse, 3), persistence_rmse=round(pers_rmse, 3) if pers_rmse == pers_rmse else np.nan,
                climatology_rmse=round(clim_rmse, 3),
                model_r2=round(float(r2_score(yte, pred)), 4),
                skill_score_vs_clim=round(1.0 - mrmse / clim_rmse, 4) if clim_rmse > 0 else np.nan,
            ))
    return rows, len(test_years)


def main():
    all_rows = []
    notes = []
    for tier in ("A", "B", "C"):
        for crop in rc.CROPS:
            rows, nty = run_tier_crop(tier, crop)
            all_rows += rows
            note = f"{crop}/{tier}: {nty} test years"
            if tier in ("B", "C") and nty < 6:
                note += "  [FEW TEST YEARS — interpret with caution]"
            notes.append(note)
            print(f"[roll] {note}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(R.REV / "rolling_origin_results.csv", index=False)
    print(f"[save] rolling_origin_results.csv rows={len(df)}", flush=True)

    # summary: mean over test years per crop×tier×model + pooled skill
    summ = (df.groupby(["crop", "tier", "model"])
            .agg(n_test_years=("test_year", "nunique"),
                 mean_model_rmse=("model_rmse", "mean"),
                 mean_clim_rmse=("climatology_rmse", "mean"),
                 mean_pers_rmse=("persistence_rmse", "mean"),
                 mean_skill_vs_clim=("skill_score_vs_clim", "mean"),
                 median_skill_vs_clim=("skill_score_vs_clim", "median"),
                 frac_years_skill_pos=("skill_score_vs_clim", lambda s: float((s > 0).mean())))
            .round(4).reset_index())
    summ.to_csv(R.REV / "rolling_origin_summary.csv", index=False)
    print("[save] rolling_origin_summary.csv", flush=True)
    print("\n=== rolling-origin mean skill vs climatology (forward-in-time) ===", flush=True)
    for _, r in summ.sort_values(["crop", "tier", "mean_skill_vs_clim"]).iterrows():
        print(f"  {r.crop:9s} {r.tier} {r.model:14s} yrs={int(r.n_test_years):2d} "
              f"mean_SS={r.mean_skill_vs_clim:+.3f} median_SS={r.median_skill_vs_clim:+.3f} "
              f"frac_yrs+={r.frac_years_skill_pos:.2f}", flush=True)
    (R.REV / "rolling_origin_notes.txt").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
