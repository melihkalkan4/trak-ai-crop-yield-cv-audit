"""T5 — Anomaly features: per-district z-scores of index/window metrics AND yield.

z = (value - district_long_term_mean) / district_std, computed WITHIN each district over the
2017–2024 record (the index era). Enables an optional parallel 'predict yield ANOMALY' analysis.
Yield z from the existing panel (verim_kg_da), read-only.
Output: outputs/anomaly_<crop>.csv  (ilce_id,ilce,il,year, <metric>_z..., yield_z)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

LAYERC = E.PROJECT_ROOT / "data" / "processed" / "calibration_features_layerC.csv"
CROP_FULL = {"bugday": "bugday", "aycicegi": "aycicegi_yaglik"}


def zscore_within(df, keycol, valcols):
    out = df[[keycol]].copy()
    g = df.groupby(keycol)
    for c in valcols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        out[c + "_z"] = (df[c] - mu) / sd.replace(0, np.nan)
    return out


def main():
    panel = pd.read_csv(LAYERC)
    for crop in ("bugday", "aycicegi"):
        idx_path = E.OUT / f"indices_{crop}.csv"
        if not idx_path.exists():
            print(f"STOP: {idx_path.name} missing — run T2 first."); return 2
        idx = pd.read_csv(idx_path)
        metric_cols = [c for c in idx.columns if c not in ("ilce_id", "ilce", "il", "year")]
        z = zscore_within(idx, "ilce_id", metric_cols)
        anom = pd.concat([idx[["ilce_id", "ilce", "il", "year"]], z[[c + "_z" for c in metric_cols]]], axis=1)

        # yield z-score within district (2017–2024), from existing panel (read-only)
        yp = panel[panel["crop"] == CROP_FULL[crop]][["ilce_id", "year", "verim_kg_da"]].copy()
        gy = yp.groupby("ilce_id")["verim_kg_da"]
        yp["yield_z"] = (yp["verim_kg_da"] - gy.transform("mean")) / gy.transform("std").replace(0, np.nan)
        anom = anom.merge(yp[["ilce_id", "year", "yield_z"]], on=["ilce_id", "year"], how="left")

        out = E.OUT / f"anomaly_{crop}.csv"
        anom.to_csv(out, index=False)
        nfin = int(np.isfinite(anom.filter(like="_z").to_numpy(dtype=float)).sum())
        print(f"[t5] {crop}: saved {out.name} {anom.shape} | finite z-values={nfin} | "
              f"yield_z non-null={int(anom['yield_z'].notna().sum())}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
