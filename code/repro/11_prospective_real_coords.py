"""11 — Prospektif (FLOV) validasyonu GERÇEK parsel koordinatlarıyla.

Kullanıcının sağladığı 4 gerçek köşe noktasıyla EVR_01 ("Kendi tarlam") sahasını
açar. Frozen LSTM (NDVI t+7) + persistence karşılaştırmasını ve per-stage
metrikleri GERÇEKTEN yeniden koşturur (S2 GEE + ERA5 CDS canlı; frozen model
SADECE inference, integrity-hash doğrulamalı).

İZOLASYON (logs/decisions.md D-18):
* config.AUDIT_FILE / LOG_FILE kendi klasörüme yönlendirilir (tracked logs'a
  yazılmaz). configure_logging() ÇAĞRILMAZ → logs/flov.log el değmez.
* build_unified_features(save=False) → data/prospective ezilmez. Tüm çıktılar
  paper1_generalization/analysis/prospective_real/ içine.
* API cache (data/cache/api/, UNTRACKED) additive olarak yazılır — kabul edilebilir.
* geometry.site_polygon_coords monkeypatch ile GERÇEK polygon döndürülür (orijinal
  dosya DEĞİŞTİRİLMEZ; runtime override).
* Frozen model yeniden eğitilmez; predict_ndvi_series(verify_integrity=True).

Kullanım:
  python 11_prospective_real_coords.py --mode smoke   # kısa pencere (hızlı test)
  python 11_prospective_real_coords.py --mode full    # 2025 tam + 2026 kısmi
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[0]
sys.path.insert(0, str(ROOT / "src"))

OUT = PAPER / "analysis" / "prospective_real"
OUT.mkdir(parents=True, exist_ok=True)

# --- isolation: redirect tracked-file writes BEFORE importing submodules ---
from prospective_validation import config  # noqa: E402
config.AUDIT_FILE = OUT / "api_audit_real.jsonl"
config.LOG_FILE = OUT / "flov_real.log"

from prospective_validation import geometry, actuals as actuals_mod  # noqa: E402
from prospective_validation.feature_builder import build_unified_features  # noqa: E402
from prospective_validation.frozen_model_predictor import make_predictor  # noqa: E402
from prospective_validation.live_validator import LiveValidator  # noqa: E402

# --- real surveyed parcel corners (lat, lon), user-provided 2026-06-20 ---
CORNERS_LATLON = [
    (41.531790, 27.861590),
    (41.531521, 27.862092),
    (41.530694, 27.861425),
    (41.530760, 27.860755),
]
INWARD_BUFFER_M = 10.0  # one S2 pixel; small field → modest buffer (see report)


def _equirect(clat):
    R = 6371000.0
    mlat = R * np.pi / 180.0
    mlon = R * np.pi / 180.0 * np.cos(np.radians(clat))
    return mlat, mlon


def build_real_site_and_polygon():
    lats = [p[0] for p in CORNERS_LATLON]
    lons = [p[1] for p in CORNERS_LATLON]
    clat, clon = float(np.mean(lats)), float(np.mean(lons))
    mlat, mlon = _equirect(clat)
    from shapely.geometry import Polygon
    # polygon in local meters (x=east, y=north), lon-lat input order
    xy = [((lo - clon) * mlon, (la - clat) * mlat) for la, lo in CORNERS_LATLON]
    poly_m = Polygon(xy)
    if not poly_m.is_valid:
        poly_m = poly_m.convex_hull
    area_m2 = poly_m.area
    area_da = area_m2 / 1000.0

    buf = INWARD_BUFFER_M
    inner = poly_m.buffer(-buf)
    while (inner.is_empty or inner.area < 0.30 * area_m2) and buf > 0:
        buf -= 5.0
        inner = poly_m.buffer(-buf) if buf > 0 else poly_m
    used_poly = inner if (buf > 0 and not inner.is_empty) else poly_m
    used_buf = buf if (buf > 0 and not inner.is_empty) else 0.0

    # back to lon-lat closed ring
    ring = [(clon + x / mlon, clat + y / mlat) for x, y in used_poly.exterior.coords]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    ring = tuple((round(lo, 7), round(la, 7)) for lo, la in ring)

    site = config.Site("EVR01R", "Gercek parsel (Vize) — real surveyed field",
                       clat, clon, "self", round(area_da, 3))
    half_side_m = float(np.sqrt(used_poly.area) / 2.0)
    poly = geometry.SitePolygon(
        site_id=site.id, coords_lonlat=ring, half_side_m=half_side_m,
        inward_buffer_m=used_buf, subpixel_risk=(area_da / 10.0 < 5.0),
    )
    info = dict(centroid_lat=clat, centroid_lon=clon, area_m2=round(area_m2, 1),
                area_da=round(area_da, 3), area_ha=round(area_da / 10.0, 4),
                inward_buffer_m_used=used_buf, used_area_m2=round(used_poly.area, 1),
                n_ring_pts=len(ring))
    return site, poly, info


def run_window(site, predictor, year_label, start, end):
    df_feat = build_unified_features(site, start, end, save=False)
    preds = predictor.predict_ndvi_series(df_feat)
    if preds.empty:
        return {"year": year_label, "error": "no predictions"}, None, None

    # actuals: raw S2 (gold standard, FLOV §7) + unified NDVI_int (matches original)
    act_raw = actuals_mod.from_sentinel2_fetch(site, start, end)
    act_uni = df_feat[["date", "NDVI_int"]].rename(columns={"NDVI_int": "actual_ndvi"})
    act_uni["source"] = "unified_features_NDVI_int"

    out = {}
    matched_to_save = None
    per_stage_to_save = None
    for tag, act in (("raw_s2", act_raw), ("unified", act_uni)):
        rep = LiveValidator(site=site).report(preds, act, tolerance_days=2)
        out[tag] = {
            "n_predictions": rep.n_predictions, "n_matched": rep.n_matched,
            "coverage_pct": round(rep.coverage_pct, 2),
            "overall_model": rep.overall, "overall_naive_persistence": rep.overall_naive,
            "wilcoxon_model_vs_naive": rep.wilcoxon,
        }
        if tag == "raw_s2":
            matched_to_save = rep.matched
            per_stage_to_save = rep.per_stage
        # save per-stage per tag
        if not rep.per_stage.empty:
            rep.per_stage.assign(year=year_label, actual_source=tag).to_csv(
                OUT / f"per_stage_{year_label}_{tag}.csv", index=False)

    # persist predictions + matched
    preds.to_csv(OUT / f"predictions_{year_label}.csv", index=False)
    if matched_to_save is not None:
        matched_to_save.to_csv(OUT / f"matched_{year_label}_raw_s2.csv", index=False)
    return {"year": year_label, **out}, df_feat, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = ap.parse_args()

    site, poly, info = build_real_site_and_polygon()
    print("[real-coords] site:", site.id, info, flush=True)

    # monkeypatch geometry to use the TRUE polygon for our site (runtime only)
    _orig = geometry.site_polygon_coords
    def _patched(s, **kw):
        return poly if s.id == site.id else _orig(s, **kw)
    geometry.site_polygon_coords = _patched

    predictor = make_predictor(load_climatology=True)

    if args.mode == "smoke":
        windows = [("2025smoke", date(2025, 4, 1), date(2025, 6, 15))]
    else:
        today = datetime.now(timezone.utc).date()
        windows = [
            ("2025", date(2025, 1, 1), date(2025, 12, 31)),
            ("2026", date(2026, 1, 1), today),
        ]

    def _write_summary(summaries):
        payload = {
            "site": {"id": site.id, "name": site.name, **info,
                     "corners_latlon": CORNERS_LATLON,
                     "polygon_lonlat_ring": list(poly.coords_lonlat)},
            "tolerance_days": 2,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "windows": summaries,
        }
        (OUT / "real_coords_validation_summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    summaries = []
    for yl, s, e in windows:
        print(f"\n[real-coords] window {yl}: {s} -> {e}", flush=True)
        summ, _df, _pr = run_window(site, predictor, yl, s, e)
        summaries.append(summ)
        _write_summary(summaries)  # incremental save (resumable)
        print(f"[real-coords] {yl} done: {json.dumps(summ.get('raw_s2', summ), default=str)[:400]}", flush=True)

    payload = _write_summary(summaries)
    print("\n[save] real_coords_validation_summary.json", flush=True)
    print(json.dumps(payload, indent=2, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
