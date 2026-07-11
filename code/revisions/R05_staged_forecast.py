"""R05 — Forecast-issuance staged tiers (#7).

The season-level features are phenology-tagged, so issuance-time tiers can be built by feature
subsetting (constant matched sample = layer C rows, n=213/209; only the feature set grows as the
season progresses). Each tier uses ONLY variables observable by that issuance date.

Feature → earliest issuance mapping (from feature semantics; documented, not invented):
  pre_season  : soil (static)
  vegetative  : + vernalization_days, tp_winter_sum, ndvi_spring_slope
  flowering   : + gdd_flowering, tp_flowering, t2m_flowering_mean, t2m_flowering_max,
                  ssr_flowering_sum, ndvi_flowering
  grain_fill  : + tp_grain_fill, ndvi_grain_fill, heat_stress_days
  end_of_season: + gdd_cum_season, tp_season_sum, ssr_season_sum, aridity_index, tdiff_mean,
                  ndvi_max, ndvi_mean_season, ndvi_integral, greenness_days  (= full Layer C)

CV = LOYO (forecast-relevant temporal). Baseline = per-ilce LOO climatology (matched).
Output: revisions/staged_forecast_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402
rc = R.rc

SOIL = rc.FEATURES_SOIL
STAGE_ADD = {
    "pre_season":    list(SOIL),
    "vegetative":    ["vernalization_days", "tp_winter_sum", "ndvi_spring_slope"],
    "flowering":     ["gdd_flowering", "tp_flowering", "t2m_flowering_mean",
                      "t2m_flowering_max", "ssr_flowering_sum", "ndvi_flowering"],
    "grain_fill":    ["tp_grain_fill", "ndvi_grain_fill", "heat_stress_days"],
    "end_of_season": ["gdd_cum_season", "tp_season_sum", "ssr_season_sum", "aridity_index",
                      "tdiff_mean", "ndvi_max", "ndvi_mean_season", "ndvi_integral",
                      "greenness_days"],
}
STAGE_ORDER = ["pre_season", "vegetative", "flowering", "grain_fill", "end_of_season"]


def cumulative_feats():
    feats, acc = {}, []
    for s in STAGE_ORDER:
        acc = acc + STAGE_ADD[s]
        feats[s] = list(acc)
    return feats


def main():
    feats_by_stage = cumulative_feats()
    rows = []
    for crop in rc.CROPS:
        sub = rc.crop_subset(rc.load_layer("C"), crop)
        y = sub["verim_kg_da"].astype(float).values
        year_g = sub["year"].astype(int).values
        b0 = rc.b0_per_sample("C", crop)["b0_pred"].values
        clim_rmse = float(np.sqrt(mean_squared_error(y, b0)))
        for stage in STAGE_ORDER:
            feats = feats_by_stage[stage]
            X = rc._impute(sub[feats].astype(float))
            for model in ["pls", "elastic_net", "random_forest", "xgboost", "gpr"]:
                pred = rc._cv_predict(model, X, y, year_g)
                mrmse = float(np.sqrt(mean_squared_error(y, pred)))
                rows.append(dict(
                    crop=crop, issuance_stage=stage, n_features=len(feats), model=model,
                    cv="LOYO", n=len(y), model_rmse=round(mrmse, 3),
                    climatology_rmse=round(clim_rmse, 3),
                    model_r2=round(float(r2_score(y, pred)), 4),
                    skill_score_vs_clim=round(1.0 - mrmse / clim_rmse, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(R.REV / "staged_forecast_results.csv", index=False)
    print(f"[save] staged_forecast_results.csv rows={len(df)}", flush=True)

    # best model per stage
    best = (df.loc[df.groupby(["crop", "issuance_stage"])["model_r2"].idxmax()]
            [["crop", "issuance_stage", "n_features", "model", "model_r2", "skill_score_vs_clim"]])
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    best = best.sort_values(["crop", "issuance_stage"], key=lambda s: s.map(order) if s.name == "issuance_stage" else s)
    print("\n=== best model per issuance stage (LOYO) ===", flush=True)
    print(best.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
