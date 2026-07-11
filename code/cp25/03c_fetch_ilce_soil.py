"""ÇP-2.5 / Adım 1.3 — SoilGrids statik toprak özellikleri (GEE).

ISRIC SoilGrids 250m, GEE üzerinden tek nokta ortalaması.  Her ilçe için
3 derinlik agreatları (0-5cm, 5-15cm, 15-30cm) ve aşağıdaki özellikler:

* `clay`  (g/kg → %)
* `sand`  (g/kg → %)
* `silt`  (türeitilmiş = 100 - clay% - sand%)
* `phh2o` (pH × 10 → pH)
* `soc`   (dg/kg → %)
* `awc`   (Available Water Capacity, türetilmiş = clay × 0.4 + silt × 0.3 + sand × 0.05)

Akademik notlar
---------------
- ISRIC SoilGrids 0.1 ölçekleme faktörü (mod_soil_isric.py'den port).
- Sınırlama: 250m çözünürlük, parsel-içi varyasyon yakalanamaz; ilçe ortalaması
  Layer C feature olarak girer (sistematik ofset). Trakya'da iyi karışım var
  (kireçtaşı/silt, kil/tın geçişleri).

Çıktı
-----
``data/processed/soil_ilce.csv``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("cp25.task03c")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUIK_DIR     = PROJECT_ROOT / "data" / "external" / "tuik"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
LOG_DIR      = PROJECT_ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GEE_KEY_PATH = PROJECT_ROOT / "keys" / "trak-ai-kds-d3e5e5b6e168.json"
SOIL_PROPS = ["clay", "sand", "phh2o", "soc"]    # silt türetilir
SOIL_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]
BUFFER_M = 2000                                  # mod_soil_isric.py default
AUDIT_LOG = LOG_DIR / "gee_soil_audit.jsonl"

_EE = None


def _import_ee():
    global _EE
    if _EE is not None:
        return _EE
    import ee                                                        # type: ignore
    creds_info = json.loads(GEE_KEY_PATH.read_text())
    credentials = ee.ServiceAccountCredentials(
        creds_info["client_email"], str(GEE_KEY_PATH))
    ee.Initialize(credentials)
    _EE = ee
    return ee


def _audit(event: str, payload: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(),
           "event": event, "source": "gee_soilgrids", **payload}
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def fetch_soil_for_point(lat: float, lon: float) -> dict:
    """Return dict with all prop_depth values, scaled to agronomic units."""
    ee = _import_ee()
    region = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)
    results: dict = {}
    for prop in SOIL_PROPS:
        for depth in SOIL_DEPTHS:
            asset_id = f"projects/soilgrids-isric/{prop}_mean"
            band = f"{prop}_{depth}_mean"
            try:
                img = ee.Image(asset_id).select(band)
                proj = img.projection()
                native_scale = proj.nominalScale()
                val = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region, crs=proj,
                    scale=native_scale, maxPixels=1e9, bestEffort=True
                ).getInfo().get(band)
                # ISRIC scaling: clay/sand g/kg → %, phh2o ×10 → pH (all ÷10).
                # SOC stored as dg/kg ×10 → divide by 100 for %.
                if val is None:
                    results[f"{prop}_{depth}"] = None
                elif prop == "soc":
                    results[f"{prop}_{depth}"] = round(val * 0.01, 3)
                else:
                    results[f"{prop}_{depth}"] = round(val * 0.1, 3)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("%s/%s failed at (%.3f,%.3f): %s",
                               prop, depth, lat, lon, exc)
                results[f"{prop}_{depth}"] = None
    # Derive silt + AWC at each depth
    for depth in SOIL_DEPTHS:
        clay = results.get(f"clay_{depth}")
        sand = results.get(f"sand_{depth}")
        if clay is not None and sand is not None:
            silt = round(100.0 - clay - sand, 3)
            results[f"silt_{depth}"] = silt
            # Available Water Capacity proxy (Saxton & Rawls 2006 simplification)
            results[f"awc_{depth}"] = round(clay * 0.004 + silt * 0.003 + sand * 0.0005, 4)
        else:
            results[f"silt_{depth}"] = None
            results[f"awc_{depth}"] = None
    return results


def main() -> None:
    coords = pd.read_csv(TUIK_DIR / "ilce_coords.csv")
    coords = coords[coords["is_trakya"]].reset_index(drop=True)
    logger.info("loaded %d Trakya ilçe", len(coords))

    rows = []
    t0 = time.time()
    for i, r in coords.iterrows():
        ilce_id, ilce, il = int(r["ilce_id"]), r["ilce"], r["il"]
        lat, lon = float(r["lat"]), float(r["lon"])
        logger.info("[%2d/%d] %s (id=%d) lat=%.3f lon=%.3f",
                    i+1, len(coords), ilce, ilce_id, lat, lon)
        t1 = time.time()
        try:
            soil = fetch_soil_for_point(lat, lon)
            row = {"ilce_id": ilce_id, "ilce": ilce, "il": il,
                   "lat": lat, "lon": lon, **soil}
            rows.append(row)
            _audit("ok", {"ilce_id": ilce_id, "elapsed_s": round(time.time() - t1, 2),
                          "n_props": sum(1 for v in soil.values() if v is not None)})
        except Exception as exc:                                    # noqa: BLE001
            logger.error("%s failed: %s", ilce, exc)
            _audit("failed", {"ilce_id": ilce_id, "error": str(exc)[:200]})

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "soil_ilce.csv"
    df.to_csv(out_path, index=False)
    logger.info("yazıldı: %s  (%d ilçe × %d kolon)  wall-clock %.1fs",
                out_path, len(df), len(df.columns), time.time() - t0)

    # Quick QA
    print("\n=== QA ÖZET ===")
    print(f"İlçe: {len(df)}")
    print(f"Clay 0-5cm mean: {df['clay_0-5cm'].mean():.1f}% (range {df['clay_0-5cm'].min():.1f}-{df['clay_0-5cm'].max():.1f})")
    print(f"Sand 0-5cm mean: {df['sand_0-5cm'].mean():.1f}% (range {df['sand_0-5cm'].min():.1f}-{df['sand_0-5cm'].max():.1f})")
    print(f"pH 0-5cm mean: {df['phh2o_0-5cm'].mean():.2f} (range {df['phh2o_0-5cm'].min():.2f}-{df['phh2o_0-5cm'].max():.2f})")
    print(f"SOC 0-5cm mean: {df['soc_0-5cm'].mean():.2f}%")


if __name__ == "__main__":
    main()
