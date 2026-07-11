"""A1 raw-sample re-extraction (READ-ONLY audit): per-PIXEL index anomaly rate with NO clamp.

Since pre-clamp raw values are not persisted, re-extract a small representative sample (a few
district-years) over the district cropland ROI, computing each index per pixel with the clamp
DISABLED, and count out-of-bounds / non-finite / extreme cropland pixels. Quantifies the TRUE raw
blow-up rate per index (esp. the unguarded CIre/NDRE/NDWI/GNDVI/OSAVI). Writes only to audit/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

# representative districts: forested/low-cropland (Demirköy), high-cropland plain (Hayrabolu), mixed (Çorlu)
SAMPLE = {"Demirköy": 1471 if False else None}  # resolved by name below
NAMES = ["Demirköy", "Kofçaz", "Pehlivanköy", "Hayrabolu"]  # small→large; incremental save below
YEAR = 2020
WIN = ((4, 1), (5, 31))   # peak window


def main():
    ee = E.gee_init()
    import geopandas as gpd
    gdf = gpd.read_file(E.DISTRICTS_GEOJSON)
    cl = E.cropland_mask(ee)
    start, end = E.window_dates(YEAR, WIN)
    # build per-pixel index image (median composite) WITHOUT EVI clamp, cropland-masked
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(gdf_region(ee, gdf))
          .filterDate(start, end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    cld = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterDate(start, end))
    j = ee.Join.saveFirst("cld").apply(s2, cld, ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def idxbands(img):
        img = ee.Image(img).updateMask(ee.Image(ee.Image(img).get("cld")).select("probability").lt(30))
        b = img.select(["B2", "B3", "B4", "B5", "B7", "B8", "B11"]).divide(10000.0)
        B2, B3, B4, B5, B7, B8, B11 = [b.select(x) for x in ["B2", "B3", "B4", "B5", "B7", "B8", "B11"]]
        return ee.Image.cat([
            B8.subtract(B4).divide(B8.add(B4)).rename("NDVI"),
            B8.subtract(B4).multiply(2.5).divide(B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1)).rename("EVI"),
            B8.subtract(B4).multiply(2.5).divide(B8.add(B4.multiply(2.4)).add(1)).rename("EVI2"),
            B8.subtract(B5).divide(B8.add(B5)).rename("NDRE"),
            B7.divide(B5).subtract(1).rename("CIre"),
            B8.subtract(B11).divide(B8.add(B11)).rename("NDWI"),
            B8.subtract(B3).divide(B8.add(B3)).rename("GNDVI"),
            B8.subtract(B4).divide(B8.add(B4).add(0.16)).rename("OSAVI"),
        ])  # NO clamp on any index
    comp = ee.ImageCollection(j.map(idxbands)).median().updateMask(cl)

    # build ONE multi-band image: per index -> total, oob, extreme; + the index bands for minMax
    stat_bands, idx_bands = [], []
    for ind in E.INDICES:
        im = comp.select(ind)
        lo, hi = (-1.0, 1.0) if ind != "CIre" else (-1.0, 10.0)
        stat_bands += [im.mask().rename(f"{ind}__tot"),
                       im.lt(lo).Or(im.gt(hi)).rename(f"{ind}__oob"),
                       im.abs().gt(1000).rename(f"{ind}__ext")]
        idx_bands.append(im)
    statimg = ee.Image.cat(stat_bands)
    idximg = ee.Image.cat(idx_bands)
    rows = []
    for nm in NAMES:
        row = gdf[gdf.ilce == nm]
        if row.empty:
            print(f"  {nm}: NOT FOUND in geojson"); continue
        region = ee.Geometry(row.iloc[0].geometry.__geo_interface__)
        s = statimg.reduceRegion(ee.Reducer.sum(), region, scale=100, maxPixels=1e10, tileScale=4).getInfo()
        mr = idximg.reduceRegion(ee.Reducer.minMax(), region, scale=100, maxPixels=1e10, tileScale=4).getInfo()
        for ind in E.INDICES:
            tot = s.get(f"{ind}__tot", 0) or 0
            oob = s.get(f"{ind}__oob", 0) or 0
            rows.append(dict(district=nm, year=YEAR, index=ind, clamp_in_pipeline=("Y" if ind in ("EVI", "EVI2") else "N"),
                             cropland_px=int(tot), oob_px=int(oob), oob_frac=round(oob / tot, 5) if tot else None,
                             extreme_px=int(s.get(f"{ind}__ext", 0) or 0),
                             px_min=round(mr.get(ind + "_min"), 3) if mr.get(ind + "_min") is not None else None,
                             px_max=round(mr.get(ind + "_max"), 3) if mr.get(ind + "_max") is not None else None))
            print(f"  {nm} {ind:6s} px={int(tot)} OOB={int(oob)} ({(oob/tot*100 if tot else 0):.3f}%) max={mr.get(ind+'_max')}", flush=True)
        pd.DataFrame(rows).to_csv(Path(__file__).resolve().parents[1] / "audit" / "raw_sample_anomaly_rates.csv", index=False)
        print(f"  [saved after {nm}] rows={len(rows)}", flush=True)
    print("[raw-sample] saved raw_sample_anomaly_rates.csv")


def gdf_region(ee, gdf):
    return ee.FeatureCollection([ee.Feature(ee.Geometry(r.geometry.__geo_interface__)) for r in gdf.itertuples()]).geometry()


if __name__ == "__main__":
    main()
