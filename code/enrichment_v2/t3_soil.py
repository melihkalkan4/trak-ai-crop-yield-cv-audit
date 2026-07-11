"""T3 — Soil enrichment (SoilGrids), STATIC, over T1 district cropland mask.

Properties: clay, sand, silt, phh2o, soc (existing) + cec, bdod, nitrogen, cfvo (new).
Root zone 0–30 cm = depth-weighted mean of 0-5,5-15,15-30 cm (weights 5,10,15).
ISRIC conversion d_factors applied (divide mapped value): clay/sand/silt/soc/phh2o/cfvo=10,
cec=10, bdod=100, nitrogen=100.
AWC (plant-available water capacity, cm3/cm3) via the Saxton & Rawls (2006) pedotransfer
[Soil Sci. Soc. Am. J. 70:1569-1578]: PAW = theta_33 - theta_1500 from sand, clay, organic matter
(OM% = SOC% x 1.724). District mean + P10/P90 over cropland.
Output: outputs/soil_features.csv (one row per district; static).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

D_FACTOR = {"clay": 10, "sand": 10, "silt": 10, "phh2o": 10, "soc": 10,
            "cec": 10, "bdod": 100, "nitrogen": 100, "cfvo": 10}
DEPTHS = [("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15)]


def soil_0_30_image(ee, prop):
    img = ee.Image(f"projects/soilgrids-isric/{prop}_mean")
    num = ee.Image(0)
    for depth, w in DEPTHS:
        num = num.add(img.select(f"{prop}_{depth}_mean").multiply(w))
    return num.divide(30.0).divide(D_FACTOR[prop]).rename(prop)


def saxton_rawls_awc(sand_pct, clay_pct, soc_gkg):
    """PAW (cm3/cm3) from Saxton & Rawls (2006). Inputs: sand/clay in %, SOC in g/kg."""
    S = np.clip(sand_pct, 0, 100) / 100.0
    C = np.clip(clay_pct, 0, 100) / 100.0
    OM = (np.clip(soc_gkg, 0, None) / 10.0) * 1.724  # SOC% -> OM%
    t1500 = (-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM)
             - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031)
    theta1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = (-0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * (S * OM)
           - 0.027 * (C * OM) + 0.452 * (S * C) + 0.299)
    theta33 = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    return theta33 - theta1500


def main():
    ee = E.gee_init()
    fc = E.districts_fc(ee)
    mask = E.cropland_mask(ee)
    bands = [soil_0_30_image(ee, p).updateMask(mask) for p in D_FACTOR]
    img = ee.Image.cat(bands)
    red = (ee.Reducer.mean()
           .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True))
    rr = img.reduceRegions(collection=fc, reducer=red, scale=E.REDUCE_SCALE_M, tileScale=4).getInfo()
    rows = []
    for f in rr["features"]:
        p = f["properties"]
        d = {"ilce_id": int(p["ilce_id"]), "ilce": p["ilce"], "il": p["il"]}
        for prop in D_FACTOR:
            d[f"{prop}_0_30_mean"] = p.get(f"{prop}_mean")
            d[f"{prop}_0_30_p10"] = p.get(f"{prop}_p10")
            d[f"{prop}_0_30_p90"] = p.get(f"{prop}_p90")
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("ilce_id").reset_index(drop=True)
    df["awc_0_30"] = saxton_rawls_awc(df["sand_0_30_mean"], df["clay_0_30_mean"], df["soc_0_30_mean"])
    out = E.OUT / "soil_features.csv"
    df.to_csv(out, index=False)
    print(f"[t3] saved {out.name}: {df.shape[0]} districts × {df.shape[1]} cols", flush=True)
    # validation: plausible ranges
    print("[t3] ranges:", flush=True)
    for c in ["clay_0_30_mean", "sand_0_30_mean", "phh2o_0_30_mean", "soc_0_30_mean",
              "cec_0_30_mean", "bdod_0_30_mean", "awc_0_30"]:
        print(f"   {c}: {df[c].min():.2f} .. {df[c].max():.2f} (median {df[c].median():.2f})", flush=True)
    awc_ok = bool(((df["awc_0_30"] > 0) & (df["awc_0_30"] < 0.35)).all())
    print(f"[t3] AWC in plausible (0,0.35) cm3/cm3: {awc_ok}  [Saxton & Rawls 2006]", flush=True)


if __name__ == "__main__":
    main()
