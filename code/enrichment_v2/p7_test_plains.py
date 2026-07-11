"""P7 — Test the interior of two large agricultural plains (Vize: Ahmetbey & Müsellim direction).

For each plain polygon, 2017–2024: crop-specific phenology classification (wheat vs sunflower) over
the cropland, classified crop area, and crop-specific NDVI/EVI/NDRE (peak-window mean + heterogeneity)
inside the polygon. GEE-only (no CDS). Real extraction; no synthetic values.
Output: outputs/plains_test_summary.csv, outputs/plains_geometry.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E
import p2_cropmask_indices as P2

PLAINS = {
    "ova1_ahmetbey": [(41.535997, 27.748237), (41.495878, 27.844822), (41.511029, 27.867134), (41.562146, 27.757665)],
    "ova2_musellim": [(41.559397, 27.601614), (41.540943, 27.690005), (41.463964, 27.644073), (41.467024, 27.586506)],
}
PEAK = {"bugday": ((4, 1), (5, 31)), "aycicegi": ((7, 1), (8, 31))}


def ee_polygon(ee, pts_latlon):
    # convex-hull order in lon-lat, closed ring
    ll = [(lo, la) for la, lo in pts_latlon]
    ring = list(Polygon(ll).convex_hull.exterior.coords)
    return ee.Geometry.Polygon([[list(c) for c in ring]])


def main():
    ee = E.gee_init()
    px = ee.Image.pixelArea()
    geom_info = {}
    rows = []
    for name, pts in PLAINS.items():
        region = ee_polygon(ee, pts)
        total_ha = region.area().divide(1e4).getInfo()
        geom_info[name] = {"corners_latlon": pts, "area_ha": round(total_ha, 1)}
        for year in E.NDVI_YEARS:
            wmask, smask = P2.crop_masks(ee, region, year)
            # cropland + crop areas within the plain
            aimg = ee.Image.cat([E.cropland_mask(ee).unmask(0).multiply(px).rename("crop_ha"),
                                 wmask.unmask(0).multiply(px).rename("w_ha"),
                                 smask.unmask(0).multiply(px).rename("s_ha")])
            a = aimg.reduceRegion(ee.Reducer.sum(), region, scale=E.REDUCE_SCALE_M, maxPixels=1e10, tileScale=4).getInfo()
            # crop-specific peak NDVI/EVI/NDRE means
            rec = dict(plain=name, year=year, plain_ha=round(total_ha, 1),
                       cropland_ha=round((a.get("crop_ha", 0) or 0) / 1e4, 1),
                       wheat_ha=round((a.get("w_ha", 0) or 0) / 1e4, 1),
                       sunflower_ha=round((a.get("s_ha", 0) or 0) / 1e4, 1))
            for crop, mask in (("bugday", wmask), ("aycicegi", smask)):
                comp, _ = E.s2_index_image(ee, region, *E.window_dates(year, PEAK[crop]))
                vals = comp.select(["NDVI", "EVI", "NDRE"]).updateMask(mask).reduceRegion(
                    ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                    region, scale=E.REDUCE_SCALE_M, maxPixels=1e10, tileScale=4).getInfo()
                pre = "w" if crop == "bugday" else "s"
                for idx in ("NDVI", "EVI", "NDRE"):
                    rec[f"{pre}_{idx}_peak_mean"] = round(vals[f"{idx}_mean"], 4) if vals.get(f"{idx}_mean") is not None else None
                    rec[f"{pre}_{idx}_peak_std"] = round(vals[f"{idx}_stdDev"], 4) if vals.get(f"{idx}_stdDev") is not None else None
            rows.append(rec)
            print(f"  [p7] {name} {year}: cropland {rec['cropland_ha']}ha wheat {rec['wheat_ha']} sun {rec['sunflower_ha']} | "
                  f"wNDVI {rec.get('w_NDVI_peak_mean')} sNDVI {rec.get('s_NDVI_peak_mean')}", flush=True)
    pd.DataFrame(rows).to_csv(E.OUT / "plains_test_summary.csv", index=False)
    (E.OUT / "plains_geometry.json").write_text(json.dumps(geom_info, indent=2), encoding="utf-8")
    print("[p7] saved plains_test_summary.csv + plains_geometry.json", flush=True)
    # quick aggregate per plain
    df = pd.DataFrame(rows)
    for name in PLAINS:
        d = df[df.plain == name]
        print(f"\n{name}: mean cropland {d.cropland_ha.mean():.0f}ha | mean wheat {d.wheat_ha.mean():.0f}ha "
              f"sun {d.sunflower_ha.mean():.0f}ha | wheat peak NDVI {d.w_NDVI_peak_mean.mean():.3f} "
              f"NDRE {d.w_NDRE_peak_mean.mean():.3f} | sun peak NDVI {d.s_NDVI_peak_mean.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
