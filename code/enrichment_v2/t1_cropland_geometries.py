"""T1 — Cropland mask + admin clip + built-up exclusion.

Extract adm2/adm1 from the user-provided zip (into WORKDIR), filter to the 29 Trakya districts via
(il, ilce) <-> (adm1_tr, adm2_tr), save polygons, and compute per-district cropland area (ESA
WorldCover 40, EXPLICIT built-up 50 exclusion). STOP if any of the 29 districts fails to join.
Outputs: outputs/geometries/trakya_districts_adm2.geojson, cropland_area_per_district.csv
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

COMPONENTS = [".shp", ".shx", ".dbf", ".prj", ".cpg"]


def _norm(s):
    if s is None:
        return ""
    s = str(s).strip()
    # Turkish-aware casefold for robust name matching
    table = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})
    return s.translate(table).lower().replace(" ", "")


def main() -> int:
    if not E.ADMIN_ZIP.exists():
        print(f"STOP: admin zip not found at {E.ADMIN_ZIP}"); return 2
    z = zipfile.ZipFile(E.ADMIN_ZIP)
    for layer in ("tur_polbna_adm2", "tur_polbnda_adm1"):
        for c in COMPONENTS:
            name = layer + c
            if name in z.namelist():
                (E.INPUTS / name).write_bytes(z.read(name))
    print(f"[t1] extracted adm2/adm1 components -> {E.INPUTS}", flush=True)

    adm2 = gpd.read_file(E.INPUTS / "tur_polbna_adm2.shp")
    if adm2.crs is None or adm2.crs.to_epsg() != 4326:
        adm2 = adm2.to_crs(4326)
    coords = pd.read_csv(E.ILCE_COORDS)
    coords = coords[coords["is_trakya"] == True].copy()
    print(f"[t1] target Trakya districts: {len(coords)} | adm2 features: {len(adm2)}", flush=True)

    adm2["_il"] = adm2["adm1_tr"].map(_norm)
    adm2["_ilce"] = adm2["adm2_tr"].map(_norm)
    coords["_il"] = coords["il"].map(_norm)
    coords["_ilce"] = coords["ilce"].map(_norm)

    # Documented administrative correspondences for the 3 "Merkez" central districts (NOT fabrication):
    #  - non-metropolitan provinces label the central district by the province name in the shapefile;
    #  - Tekirdağ Merkez (id 1673, 2004–2012) was reorganized into Süleymanpaşa in 2014 — same area,
    #    temporally disjoint from Süleymanpaşa (id 2096, 2013–2025); Merkez has ZERO NDVI-era (2017–24)
    #    rows, so it does not enter the NDVI tiers. The two ilce_ids therefore share one modern polygon.
    MANUAL = {(_norm("Edirne"), _norm("Merkez")): _norm("EDIRNE"),
              (_norm("Kırklareli"), _norm("Merkez")): _norm("KIRKLARELI"),
              (_norm("Tekirdağ"), _norm("Merkez")): _norm("SÜLEYMANPAŞA")}
    coords["_geomnote"] = ""
    for i, r in coords.iterrows():
        k = (r["_il"], r["_ilce"])
        if k in MANUAL:
            coords.at[i, "_ilce"] = MANUAL[k]
            coords.at[i, "_geomnote"] = f"central-district correspondence ({r['il']} Merkez)"
            if r["ilce_id"] == 1673:
                coords.at[i, "_geomnote"] = "shared polygon w/ Süleymanpaşa (2014 reorg; no NDVI-era rows)"

    merged = coords.merge(adm2, on=["_il", "_ilce"], how="left", suffixes=("", "_adm"))
    miss = merged[merged["geometry"].isna()]
    if len(miss) > 0:
        print("STOP: districts failed to join (need manual name mapping):", flush=True)
        print(miss[["ilce_id", "il", "ilce"]].to_string(index=False), flush=True)
        # show candidate adm2 names in those provinces to aid mapping
        for il in miss["_il"].unique():
            cand = adm2[adm2["_il"] == il]["adm2_tr"].tolist()
            print(f"   adm2 in {il}: {cand}", flush=True)
        return 3

    gdf = gpd.GeoDataFrame(
        merged[["ilce_id", "ilce", "il", "adm2_tr", "adm1_tr", "pcode", "_geomnote", "geometry"]]
        .rename(columns={"_geomnote": "geom_note"}),
        geometry="geometry", crs="EPSG:4326")
    gdf.to_file(E.DISTRICTS_GEOJSON, driver="GeoJSON")
    print(f"[t1] saved {len(gdf)} district polygons -> {E.DISTRICTS_GEOJSON.name}", flush=True)

    # ---- GEE: per-district cropland area + explicit built-up exclusion check ----
    ee = E.gee_init()
    fc = E.districts_fc(ee)
    wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    px = ee.Image.pixelArea()
    img = ee.Image.cat([
        wc.eq(40).multiply(px).rename("cropland_m2"),
        wc.eq(50).multiply(px).rename("builtup_m2"),
        px.rename("total_m2"),
    ])
    red = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=10)
    rows = []
    for f in red.getInfo()["features"]:
        p = f["properties"]
        crop = p.get("cropland_m2", 0) or 0
        built = p.get("builtup_m2", 0) or 0
        tot = p.get("total_m2", 0) or 0
        rows.append(dict(ilce_id=int(p["ilce_id"]), ilce=p["ilce"], il=p["il"],
                         cropland_ha=round(crop / 1e4, 1), builtup_ha=round(built / 1e4, 1),
                         district_ha=round(tot / 1e4, 1),
                         cropland_frac=round(crop / tot, 4) if tot else None))
    area = pd.DataFrame(rows).sort_values("ilce_id")
    area.to_csv(E.GEO / "cropland_area_per_district.csv", index=False)
    print(f"[t1] cropland area saved; districts={len(area)}", flush=True)
    print("\n=== cropland fraction summary ===", flush=True)
    print(f"  cropland fraction: min={area.cropland_frac.min():.2f} "
          f"median={area.cropland_frac.median():.2f} max={area.cropland_frac.max():.2f}", flush=True)
    print(f"  total cropland: {area.cropland_ha.sum():.0f} ha | total built-up (EXCLUDED): "
          f"{area.builtup_ha.sum():.0f} ha", flush=True)
    print(area[["ilce", "il", "cropland_ha", "builtup_ha", "cropland_frac"]].head(8).to_string(index=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
