"""R09 — Parcel per-stage reconciliation, single authoritative table (#9).

Resolves the Table-5 discrepancy: the manuscript's per-stage values (flowering -0.51,
grain-fill +0.59, maturity -13.7) are the REAL-coords 2025 results (interpolated NDVI_int),
whereas reports/cp25/.. per_stage_metrics (flowering +0.74, grain-fill -0.96, maturity -5.65)
were the PLACEHOLDER-coords run. The authoritative source = the real-coords matched files,
recomputed here WITH persistence per stage and MAE/RMSE/median|err| (since R² is unstable at
low-variance senescence stages).

Source: revisions/.. analysis/prospective_real/matched_{2025,2026}_raw_s2.csv (have predicted_ndvi,
last_observed_ndvi=persistence, actual_ndvi, target_date). Stage from target_date DOY.

Output: revisions/parcel_per_stage.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402

PR = R.ANALYSIS / "prospective_real"
STAGES = [("pre_season", 1, 104), ("emergence", 105, 130), ("vegetative", 131, 170),
          ("flowering", 171, 200), ("grain_fill", 201, 240), ("maturity", 241, 280),
          ("post_harvest", 281, 366)]


def stage_for(doy):
    for nm, a, b in STAGES:
        if a <= doy <= b:
            return nm
    return "unknown"


def _r2(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((np.asarray(yt, float) - np.asarray(yp, float)) ** 2)))


def _mae(yt, yp):
    return float(np.mean(np.abs(np.asarray(yt, float) - np.asarray(yp, float))))


def process(path, season):
    if not path.exists():
        return []
    m = pd.read_csv(path)
    m["target_date"] = pd.to_datetime(m["target_date"])
    m["stage"] = m["target_date"].dt.dayofyear.map(stage_for)
    out = []
    for stage, g in m.groupby("stage"):
        yt = g["actual_ndvi"].values
        ym = g["predicted_ndvi"].values
        yp = g["last_observed_ndvi"].values  # persistence
        out.append(dict(
            season=season, actuals_source="raw_s2", stage=stage, n_dates=len(g),
            model_mae=round(_mae(yt, ym), 4), persistence_mae=round(_mae(yt, yp), 4),
            model_rmse=round(_rmse(yt, ym), 4), persistence_rmse=round(_rmse(yt, yp), 4),
            model_median_abs_err=round(float(np.median(np.abs(yt - ym))), 4),
            persistence_median_abs_err=round(float(np.median(np.abs(yt - yp))), 4),
            model_r2=round(_r2(yt, ym), 4), persistence_r2=round(_r2(yt, yp), 4),
        ))
    return out


def main():
    rows = []
    rows += process(PR / "matched_2025_raw_s2.csv", 2025)
    rows += process(PR / "matched_2026_raw_s2.csv", 2026)
    order = {nm: i for i, (nm, _, _) in enumerate(STAGES)}
    df = pd.DataFrame(rows).sort_values(["season", "stage"], key=lambda s: s.map(order) if s.name == "stage" else s)
    df.to_csv(R.REV / "parcel_per_stage.csv", index=False)
    print(f"[save] parcel_per_stage.csv rows={len(df)}", flush=True)
    print(df[df.season == 2025][["stage", "n_dates", "model_r2", "persistence_r2",
                                 "model_mae", "persistence_mae"]].to_string(index=False), flush=True)

    note = (
        "PER-STAGE RECONCILIATION (#9)\n=============================\n"
        "AUTHORITATIVE source = real surveyed parcel, raw Sentinel-2 actuals (this table).\n"
        "Manuscript Table-5 values (flowering -0.51 / grain-fill +0.59 / maturity -13.7) came from\n"
        "the REAL-coords 2025 run (interpolated NDVI_int variant); the older reports/cp25 per_stage\n"
        "(flowering +0.74 / grain-fill -0.96 / maturity -5.65) came from the PLACEHOLDER-coords run.\n"
        "Both are now superseded by this single real-parcel table, which ALSO reports persistence and\n"
        "MAE/RMSE/median|err| per stage. R2 is unstable at low-NDVI-variance stages (emergence,\n"
        "maturity) so MAE/RMSE/median|err| are the well-posed per-stage metrics; the robust statement\n"
        "is the overall model-vs-persistence comparison (R03/F4: persistence wins, Wilcoxon p=1.000).\n"
    )
    (R.REV / "parcel_per_stage_reconciliation.txt").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
