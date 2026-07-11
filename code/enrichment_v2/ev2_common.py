"""ev2_common — shared foundation for enrichment_v2 (T1–T9).

Isolation: writes ONLY under enrichment_v2/. Reads existing repo read-only. GEE via the existing
service-account key (keys/*.json). All values from REAL extraction; on auth/source failure → raise
(caller STOPs). EVI/EVI2 clamped to |≤1| before compositing (mandatory).
"""
from __future__ import annotations

import os

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EV2 = PROJECT_ROOT / "enrichment_v2"
CODE = EV2 / "code"
INPUTS = CODE / "inputs"
OUT = EV2 / "outputs"
GEO = OUT / "geometries"
TAB = OUT / "tables"
FIGS = OUT / "figures"
for d in (INPUTS, OUT, GEO, TAB, FIGS):
    d.mkdir(parents=True, exist_ok=True)

TUIK = PROJECT_ROOT / "data" / "external" / "tuik"
ILCE_COORDS = TUIK / "ilce_coords.csv"
ADMIN_ZIP = Path(os.environ.get("TRAKAI_ADMIN_ZIP", str(PROJECT_ROOT / "data" / "Turkey_Administrative_Levels.zip")))  # set TRAKAI_ADMIN_ZIP or place the ESA/GADM admin zip here
DISTRICTS_GEOJSON = GEO / "trakya_districts_adm2.geojson"

SEED = 42

# Crop-specific phenological windows (spec defaults; advisor-adjustable).
# (start_month, start_day) .. (end_month, end_day), inclusive.
CROP_WINDOWS = {
    "bugday": {  # winter wheat
        "greenup":    ((2, 1), (3, 31)),
        "peak":       ((4, 1), (5, 31)),
        "grainfill":  ((6, 1), (6, 30)),
    },
    "aycicegi": {  # sunflower
        "greenup":    ((6, 1), (6, 30)),
        "peak":       ((7, 1), (8, 31)),
        "harvest":    ((9, 1), (9, 30)),
    },
}
NDVI_YEARS = list(range(2017, 2025))  # 2017–2024 (Sentinel-2 era)
INDICES = ["NDVI", "EVI", "EVI2", "NDRE", "CIre", "NDWI", "GNDVI", "OSAVI"]
REDUCE_SCALE_M = 30  # district aggregation scale (matches existing 03b district-NDVI precedent;
#                      tractable for percentile reducers over large cropland districts)


# --------------------------------------------------------------------------- GEE
_ee = None


def gee_init():
    global _ee
    if _ee is not None:
        return _ee
    import ee
    keys = glob.glob(str(PROJECT_ROOT / "keys" / "*.json"))
    if not keys:
        raise RuntimeError("GEE service-account key not found in keys/ — STOP (no synthetic data).")
    info = json.load(open(keys[0]))
    ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], keys[0]))
    _ee = ee
    return ee


# ----------------------------------------------------------------- WorldCover mask
def cropland_mask(ee):
    """ESA WorldCover v200 (2021): cropland (40) with EXPLICIT built-up (50) exclusion."""
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    cropland = wc.eq(40)
    builtup = wc.eq(50)
    return cropland.And(builtup.Not()).rename("cropland")  # == cropland (40 excludes 50 by class)


# ----------------------------------------------------------------- Sentinel-2 indices
def s2_index_image(ee, region, start, end, cloud_prob_max=30):
    """Median composite of the 8 indices over [start,end], s2cloudless<thr, cropland-masked.
    EVI/EVI2 clamped to |value|<=1 (out-of-range -> masked) BEFORE compositing.
    Returns (image_with_8_index_bands_masked_to_cropland, n_scenes)."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    cld = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
           .filterBounds(region).filterDate(start, end))
    joined = ee.Join.saveFirst("cld").apply(
        primary=s2, secondary=cld,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def _mask_and_index(img):
        img = ee.Image(img)
        prob = ee.Image(img.get("cld")).select("probability")
        img = img.updateMask(prob.lt(cloud_prob_max))
        b = img.select(["B2", "B3", "B4", "B5", "B7", "B8", "B11"]).divide(10000.0)
        B2, B3, B4, B5, B7, B8, B11 = [b.select(x) for x in ["B2", "B3", "B4", "B5", "B7", "B8", "B11"]]
        ndvi = B8.subtract(B4).divide(B8.add(B4)).rename("NDVI")
        evi = B8.subtract(B4).multiply(2.5).divide(
            B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1)).rename("EVI")
        evi2 = B8.subtract(B4).multiply(2.5).divide(B8.add(B4.multiply(2.4)).add(1)).rename("EVI2")
        ndre = B8.subtract(B5).divide(B8.add(B5)).rename("NDRE")
        cire = B7.divide(B5).subtract(1).rename("CIre")
        ndwi = B8.subtract(B11).divide(B8.add(B11)).rename("NDWI")
        gndvi = B8.subtract(B3).divide(B8.add(B3)).rename("GNDVI")
        osavi = B8.subtract(B4).divide(B8.add(B4).add(0.16)).rename("OSAVI")
        # MANDATORY clamp: EVI/EVI2 out of [-1,1] -> mask (prevents the prior EVI~4.47e9 bug)
        evi = evi.updateMask(evi.abs().lte(1.0))
        evi2 = evi2.updateMask(evi2.abs().lte(1.0))
        return ee.Image.cat([ndvi, evi, evi2, ndre, cire, ndwi, gndvi, osavi]) \
            .copyProperties(img, ["system:time_start"])

    idx = ee.ImageCollection(joined.map(_mask_and_index))
    n = idx.size()
    comp = idx.median().updateMask(cropland_mask(ee))
    return comp, n


def window_dates(year, win):
    (sm, sd), (em, ed) = win
    return f"{year}-{sm:02d}-{sd:02d}", f"{year}-{em:02d}-{ed:02d}"


# ----------------------------------------------------------------- districts
def load_districts_gdf():
    import geopandas as gpd
    if not DISTRICTS_GEOJSON.exists():
        raise RuntimeError(f"{DISTRICTS_GEOJSON} missing — run T1 first.")
    return gpd.read_file(DISTRICTS_GEOJSON)


def districts_fc(ee):
    """EE FeatureCollection of the 29 Trakya district polygons (from T1 geojson), with ilce_id."""
    gdf = load_districts_gdf()
    feats = []
    for r in gdf.itertuples():
        geom = r.geometry.__geo_interface__
        feats.append(ee.Feature(ee.Geometry(geom), {"ilce_id": int(r.ilce_id),
                                                     "ilce": r.ilce, "il": r.il}))
    return ee.FeatureCollection(feats)


def combined_reducer(ee):
    return (ee.Reducer.mean()
            .combine(ee.Reducer.median(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.percentile([10, 90]), sharedInputs=True))
