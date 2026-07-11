"""T4 — Topography from DEM (STATIC), zonal mean over T1 district cropland polygons.

DEM = SRTM (USGS/SRTMGL1_003, 30 m). elevation, slope (ee.Terrain), aspect → northness=cos(aspect),
eastness=sin(aspect); TWI = ln(SCA / tan(slope)) with specific catchment area SCA = upstream area
(MERIT Hydro 'upa', km2 → m2) / cell width (~90 m) [Beven & Kirkby 1979]. Cropland-masked mean.
Output: outputs/topo_features.csv (one row per district).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E


def main():
    ee = E.gee_init()
    fc = E.districts_fc(ee)
    mask = E.cropland_mask(ee)

    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    terr = ee.Terrain.products(dem)
    slope = terr.select("slope")            # degrees
    aspect = terr.select("aspect")          # degrees
    asp_rad = aspect.multiply(np.pi / 180.0)
    northness = asp_rad.cos().rename("northness")
    eastness = asp_rad.sin().rename("eastness")

    img1 = ee.Image.cat([dem.rename("elevation_m"), slope.rename("slope_deg"),
                         northness, eastness]).updateMask(mask)
    r1 = img1.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                            scale=30, tileScale=4).getInfo()

    # TWI from MERIT Hydro upstream area
    upa = ee.Image("MERIT/Hydro/v1_0_1").select("upa")        # km2
    sca = upa.multiply(1e6).divide(90.0)                       # m2 / ~90 m cell width
    slope_rad = slope.multiply(np.pi / 180.0).max(ee.Image(0.001))
    twi = sca.divide(slope_rad.tan()).log().rename("twi").updateMask(mask)
    r2 = twi.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                           scale=90, tileScale=4).getInfo()
    twi_map = {int(f["properties"]["ilce_id"]): f["properties"].get("mean") for f in r2["features"]}

    rows = []
    for f in r1["features"]:
        p = f["properties"]
        i = int(p["ilce_id"])
        rows.append(dict(ilce_id=i, ilce=p["ilce"], il=p["il"],
                         elevation_m=p.get("elevation_m"), slope_deg=p.get("slope_deg"),
                         northness=p.get("northness"), eastness=p.get("eastness"),
                         twi=twi_map.get(i)))
    df = pd.DataFrame(rows).sort_values("ilce_id").reset_index(drop=True)
    out = E.OUT / "topo_features.csv"
    df.to_csv(out, index=False)
    print(f"[t4] saved {out.name}: {df.shape[0]} districts × {df.shape[1]} cols", flush=True)
    for c in ["elevation_m", "slope_deg", "northness", "eastness", "twi"]:
        print(f"   {c}: {df[c].min():.3f} .. {df[c].max():.3f} (median {df[c].median():.3f})", flush=True)


if __name__ == "__main__":
    main()
