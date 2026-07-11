"""T2 — Multi-index extraction + distribution metrics (per crop-specific window).

8 indices × {mean,median,stdDev,P10,P90} (+CV, range) over the T1 district cropland mask, per
phenological window, 2017–2024. s2cloudless<30%. EVI/EVI2 clamped |<=1| BEFORE compositing.
Output: outputs/indices_<crop>.csv  (wide: ilce_id,ilce,il,year, {index}_{window}_{metric}...)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

STATS = ["mean", "median", "stdDev", "p10", "p90"]


def extract_crop(ee, crop):
    """One reduceRegions call per YEAR over a combined image of all windows (8 idx x 3 win = 24 bands)."""
    fc = E.districts_fc(ee)
    region = fc.geometry()
    red = E.combined_reducer(ee)
    wins = list(E.CROP_WINDOWS[crop].items())
    rows = {}
    clamp_diag = None
    for year in E.NDVI_YEARS:
        band_imgs = []
        for win, dates in wins:
            start, end = E.window_dates(year, dates)
            comp, _ = E.s2_index_image(ee, region, start, end)
            comp = comp.rename([f"{idx}_{win}" for idx in E.INDICES])  # {index}_{window}
            band_imgs.append(comp)
        combined = ee.Image.cat(band_imgs)   # 24 bands
        rr = combined.reduceRegions(collection=fc, reducer=red,
                                    scale=E.REDUCE_SCALE_M, tileScale=4)
        for f in rr.getInfo()["features"]:
            p = f["properties"]
            d = rows.setdefault((int(p["ilce_id"]), year),
                                {"ilce_id": int(p["ilce_id"]), "ilce": p["ilce"],
                                 "il": p["il"], "year": year})
            for win, _ in wins:
                for idx in E.INDICES:
                    band = f"{idx}_{win}"
                    vals = {s: p.get(f"{band}_{s}") for s in STATS}
                    for s in STATS:
                        d[f"{idx}_{win}_{s}"] = vals[s]
                    mn, sd = vals["mean"], vals["stdDev"]
                    d[f"{idx}_{win}_cv"] = (sd / mn) if (mn not in (None, 0) and sd is not None) else None
                    p10, p90 = vals["p10"], vals["p90"]
                    d[f"{idx}_{win}_range"] = (p90 - p10) if (p10 is not None and p90 is not None) else None
        print(f"  [{crop}] {year} done ({len(wins)} windows, 1 reduce)", flush=True)
        if clamp_diag is None:
            w0 = wins[1][1]  # peak window dates
            s, e = E.window_dates(year, w0)
            clamp_diag = _clamp_fraction(ee, region, s, e)
    return pd.DataFrame(list(rows.values())), clamp_diag


def _clamp_fraction(ee, region, start, end):
    """Representative count of EVI pixels removed by the |EVI|<=1 clamp (over cropland)."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(start, end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
    cld = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(region).filterDate(start, end))
    j = ee.Join.saveFirst("cld").apply(s2, cld, ee.Filter.equals(leftField="system:index", rightField="system:index"))

    def _evi(img):
        img = ee.Image(img)
        img = img.updateMask(ee.Image(img.get("cld")).select("probability").lt(30))
        b = img.select(["B2", "B4", "B8"]).divide(10000.0)
        evi = b.select("B8").subtract(b.select("B4")).multiply(2.5).divide(
            b.select("B8").add(b.select("B4").multiply(6)).subtract(b.select("B2").multiply(7.5)).add(1))
        valid = evi.mask()
        outof = valid.And(evi.abs().gt(1.0))
        return ee.Image.cat([valid.rename("tot"), outof.rename("out")])
    ic = ee.ImageCollection(j.map(_evi)).sum().updateMask(E.cropland_mask(ee))
    st = ic.reduceRegion(ee.Reducer.sum(), region, scale=E.REDUCE_SCALE_M, maxPixels=1e10, tileScale=4).getInfo()
    tot, out = st.get("tot", 0) or 0, st.get("out", 0) or 0
    return {"window": f"{start}..{end}", "evi_px_total": tot, "evi_px_clamped": out,
            "clamped_frac": round(out / tot, 6) if tot else None}


def main():
    ee = E.gee_init()
    diags = []
    for crop in ("bugday", "aycicegi"):
        print(f"[t2] extracting {crop} ...", flush=True)
        df, diag = extract_crop(ee, crop)
        df = df.sort_values(["ilce_id", "year"]).reset_index(drop=True)
        out = E.OUT / f"indices_{crop}.csv"
        df.to_csv(out, index=False)
        print(f"[t2] saved {out.name}: {df.shape[0]} rows × {df.shape[1]} cols", flush=True)
        # post-hoc clamp validation: all EVI/EVI2 stats within [-1,1]
        evicols = [c for c in df.columns if c.startswith(("EVI_", "EVI2_")) and c.rsplit("_", 1)[-1] in
                   ("mean", "median", "p10", "p90")]
        vals = df[evicols].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        in_range = bool((np.abs(vals) <= 1.0 + 1e-9).all())
        print(f"[t2] {crop} EVI/EVI2 stats in [-1,1]: {in_range} "
              f"(min={vals.min():.3f} max={vals.max():.3f}) | clamp diag: {diag}", flush=True)
        diag.update({"crop": crop, "evi_stats_in_range": in_range})
        diags.append(diag)
    pd.DataFrame(diags).to_csv(E.OUT / "t2_evi_clamp_diagnostic.csv", index=False)
    print("[t2] done.", flush=True)


if __name__ == "__main__":
    main()
