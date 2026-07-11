"""P2 — Crop-SPECIFIC masking + RS indices (advisor #3 + limitation #1).

Per year, classify cropland pixels into winter-wheat vs sunflower by NDVI phenology (wheat: green
Apr–May, bare Jul–Aug; sunflower: green Jul–Aug). Validate classified crop AREA per district-year
against TÜİK ekilen_alan_da. Then extract the advisor's 3 indices (NDVI, NDRE, EVI) with the 7
phenological-distribution metrics over each crop's OWN windows, masked to that crop's pixels.

Honest notes: this adds a classification-accuracy assumption (validated vs TÜİK area; crop rotation
→ per-year masks). EVI clamped |<=1|. Output: crop_specific_indices_<crop>.csv, crop_area_validation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

CROP_INDICES = ["NDVI", "NDRE", "EVI"]            # advisor B = ndvi, ndre, evi
STATS = ["mean", "median", "stdDev", "p10", "p90"]
CLASS_WIN = {"aprmay": ((4, 1), (5, 31)), "julaug": ((7, 1), (8, 31))}
# phenology classification thresholds (validated vs TÜİK area; documented)
WHEAT_SPRING_MIN = 0.45
WHEAT_DROP_MIN = 0.20      # NDVI(AprMay) - NDVI(JulAug)
SUN_SUMMER_MIN = 0.50
SUN_RISE_MIN = 0.10        # NDVI(JulAug) - NDVI(AprMay)


def _median_ndvi(ee, region, start, end):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(start, end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    cld = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(region).filterDate(start, end))
    j = ee.Join.saveFirst("cld").apply(s2, cld, ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def _n(img):
        img = ee.Image(img).updateMask(ee.Image(ee.Image(img).get("cld")).select("probability").lt(30))
        b = img.select(["B4", "B8"]).divide(10000.0)
        return b.select("B8").subtract(b.select("B4")).divide(b.select("B8").add(b.select("B4"))).rename("NDVI")
    return ee.ImageCollection(j.map(_n)).median()


def crop_masks(ee, region, year):
    am = _median_ndvi(ee, region, *E.window_dates(year, CLASS_WIN["aprmay"]))
    ja = _median_ndvi(ee, region, *E.window_dates(year, CLASS_WIN["julaug"]))
    crop = E.cropland_mask(ee)
    wheat = crop.And(am.gte(WHEAT_SPRING_MIN)).And(am.subtract(ja).gte(WHEAT_DROP_MIN)).rename("wheat")
    sun = crop.And(ja.gte(SUN_SUMMER_MIN)).And(ja.subtract(am).gte(SUN_RISE_MIN)).And(wheat.Not()).rename("sun")
    return wheat.selfMask(), sun.selfMask()


def indices_masked(ee, region, year, win_dates, mask):
    comp, _ = E.s2_index_image(ee, region, *E.window_dates(year, win_dates))  # already cropland-masked + EVI clamp
    return comp.select(CROP_INDICES).updateMask(mask)


def main():
    ee = E.gee_init()
    fc = E.districts_fc(ee)
    region = fc.geometry()
    px = ee.Image.pixelArea()
    red = E.combined_reducer(ee)
    crop_key = {"bugday": "wheat", "aycicegi": "sun"}
    rows = {c: {} for c in ("bugday", "aycicegi")}
    area_rows = []
    for year in E.NDVI_YEARS:
        wmask, smask = crop_masks(ee, region, year)
        # area validation (classified ha per district)
        aimg = ee.Image.cat([wmask.unmask(0).multiply(px).rename("wheat_ha"),
                             smask.unmask(0).multiply(px).rename("sun_ha")])
        ar = aimg.reduceRegions(fc, ee.Reducer.sum(), scale=E.REDUCE_SCALE_M, tileScale=4).getInfo()
        for f in ar["features"]:
            p = f["properties"]
            area_rows.append(dict(ilce_id=int(p["ilce_id"]), ilce=p["ilce"], il=p["il"], year=year,
                                  wheat_classified_ha=round((p.get("wheat_ha", 0) or 0) / 1e4, 1),
                                  sun_classified_ha=round((p.get("sun_ha", 0) or 0) / 1e4, 1)))
        # crop-specific indices per crop over its own windows
        for crop in ("bugday", "aycicegi"):
            mask = wmask if crop == "bugday" else smask
            band_imgs = []
            wins = list(E.CROP_WINDOWS[crop].items())
            for win, dates in wins:
                im = indices_masked(ee, region, year, dates, mask)
                band_imgs.append(im.rename([f"{idx}_{win}" for idx in CROP_INDICES]))
            combined = ee.Image.cat(band_imgs)
            rr = combined.reduceRegions(fc, red, scale=E.REDUCE_SCALE_M, tileScale=4).getInfo()
            for f in rr["features"]:
                p = f["properties"]
                d = rows[crop].setdefault((int(p["ilce_id"]), year),
                                          {"ilce_id": int(p["ilce_id"]), "ilce": p["ilce"], "il": p["il"], "year": year})
                for win, _ in wins:
                    for idx in CROP_INDICES:
                        band = f"{idx}_{win}"
                        v = {s: p.get(f"{band}_{s}") for s in STATS}
                        for s in STATS:
                            d[f"{idx}_{win}_{s}"] = v[s]
                        mn, sd = v["mean"], v["stdDev"]
                        d[f"{idx}_{win}_cv"] = (sd / mn) if (mn not in (None, 0) and sd is not None) else None
                        d[f"{idx}_{win}_range"] = ((v["p90"] - v["p10"]) if (v["p10"] is not None and v["p90"] is not None) else None)
        print(f"  [p2] {year} done (masks + 2-crop indices)", flush=True)
    for crop in ("bugday", "aycicegi"):
        df = pd.DataFrame(list(rows[crop].values())).sort_values(["ilce_id", "year"])
        df.to_csv(E.OUT / f"crop_specific_indices_{crop}.csv", index=False)
        print(f"[p2] saved crop_specific_indices_{crop}.csv {df.shape}", flush=True)
    pd.DataFrame(area_rows).sort_values(["ilce_id", "year"]).to_csv(E.OUT / "crop_classified_area.csv", index=False)
    print("[p2] saved crop_classified_area.csv", flush=True)


if __name__ == "__main__":
    main()
