"""ÇP-2.5 / Görev 4 — Sezonluk feature engineering.

Her (ilce_id, year, crop) için 3 katmanlı feature seti üretir:

* **Layer A (climate-only)**: NASA POWER günlük → sezonluk agregatlar (GDD,
  yağış, ısı stresi, vernalizasyon, ET-proxy).  Climate ETL zorunlu.
* **Layer B (Layer A + NDVI)**: GEE 16-günlük composite → sezon shape
  descriptor'leri (peak, flowering mean, integral, greenness days).
  NDVI ETL bittikçe satır-satır eklenir; eksikse o (ilce, year, crop)
  satırı Layer B'den dışlanır (otomatik fallback A).
* **Layer C (Layer B + soil)**: SoilGrids statik (clay/sand/silt/pH/SOC/AWC),
  derinlik agreatlı (sezon başında değişmez).

Fenolojik pencereler (BBCH + Kern 2018 + Trakya literatürü)
-----------------------------------------------------------
Buğday (kışlık):  1 Eki (t-1) → 15 Tem (t)
  - Vernalizasyon: Kas-Oca
  - Çiçeklenme:    May
  - Tane dolum:    Haz

Ayçiçeği (yağlık): 1 Nis → 30 Eyl
  - Çiçeklenme:   Tem
  - Tane dolum:   Ağu

Çıktılar
--------
* ``data/processed/calibration_features_layerA.csv``   (climate-only, n=~1165)
* ``data/processed/calibration_features_layerB.csv``   (+ NDVI, NDVI dataset hazırsa)
* ``data/processed/calibration_features_layerC.csv``   (+ soil; B'nin üstü)
* ``reports/cp25/04_features_qa.md``                  — coverage + QA özeti
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cp25.task04")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUIK_DIR     = PROJECT_ROOT / "data" / "external" / "tuik"
CLIMATE_DIR  = PROJECT_ROOT / "data" / "processed" / "openmeteo_ilce"
NDVI_DIR     = PROJECT_ROOT / "data" / "processed" / "ndvi_ilce"
SOIL_PATH    = PROJECT_ROOT / "data" / "processed" / "soil_ilce.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Sezon pencereleri (mm, dd)
SEASON = {
    "bugday": {
        "start": (10, 1),  "end":   (7, 15),
        "year_offset": -1,
        "vernalization": ((11, 1), (1, 31)),
        "flowering":     ((5, 1),  (5, 31)),
        "grain_fill":    ((6, 1),  (6, 30)),
        "gdd_base": 0.0,
    },
    "aycicegi_yaglik": {
        "start": (4, 1),  "end":   (9, 30),
        "year_offset": 0,
        "vernalization": None,
        "flowering":     ((7, 1),  (7, 31)),
        "grain_fill":    ((8, 1),  (8, 31)),
        "gdd_base": 8.0,
    },
}

# Heat-stress thresholds (Tmax) per crop, Trakya literatürü:
HEAT_STRESS_TMAX = {"bugday": 30.0, "aycicegi_yaglik": 32.0}

# NDVI greenness threshold (Kogan 1990 VCI lower-bound)
NDVI_GREENNESS = 0.6


# ---------------------------------------------------------------------------
def _safe_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                date_col: str = "date") -> pd.DataFrame:
    return df.loc[(df[date_col] >= start) & (df[date_col] <= end)].copy()


def _season_window(year: int, crop: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = SEASON[crop]
    sm, sd = cfg["start"]; em, ed = cfg["end"]
    yoff = cfg["year_offset"]
    return (pd.Timestamp(year + yoff, sm, sd), pd.Timestamp(year, em, ed))


# ---------------------------------------------------------------------------
def climate_features(climate_df: pd.DataFrame, year: int, crop: str
                     ) -> Optional[dict]:
    """Tamamen NASA POWER günlük serisinden sezonluk klima feature'ları."""
    cfg = SEASON[crop]
    s, e = _season_window(year, crop)
    sub = _safe_slice(climate_df, s, e)
    if len(sub) < 60:               # min ~2 ay
        return None

    # GDD (Tbase-dependent)
    gdd_base = cfg["gdd_base"]
    daily_gdd = ((sub["T2M_MAX"] + sub["T2M_MIN"]) / 2.0 - gdd_base).clip(lower=0.0)
    gdd_cum_season = float(daily_gdd.sum())

    # Flowering window features
    fl_s = pd.Timestamp(year, *cfg["flowering"][0])
    fl_e = pd.Timestamp(year, *cfg["flowering"][1])
    fl = _safe_slice(sub, fl_s, fl_e)
    gdd_flowering = float(((fl["T2M_MAX"] + fl["T2M_MIN"]) / 2.0 - gdd_base
                           ).clip(lower=0.0).sum()) if not fl.empty else 0.0

    # Vernalization (only wheat)
    vern_days = 0
    if cfg["vernalization"]:
        # The vernalization window crosses calendar boundary for wheat (Nov-Jan)
        vm0, vd0 = cfg["vernalization"][0]
        vm1, vd1 = cfg["vernalization"][1]
        v_s = pd.Timestamp(year + cfg["year_offset"], vm0, vd0)
        v_e = pd.Timestamp(year, vm1, vd1)
        v = _safe_slice(sub, v_s, v_e)
        vern_days = int(((v["T2M"] >= 0.0) & (v["T2M"] <= 10.0)).sum())

    # Heat-stress (Tmax > threshold during reproductive phase)
    rep_s = pd.Timestamp(year, *cfg["flowering"][0])
    rep_e = pd.Timestamp(year, *cfg["grain_fill"][1])
    rep = _safe_slice(sub, rep_s, rep_e)
    hs_thresh = HEAT_STRESS_TMAX[crop]
    heat_stress_days = int((rep["T2M_MAX"] > hs_thresh).sum())

    # Precipitation aggregates
    tp_season_sum = float(sub["PRECTOTCORR"].sum())
    if crop == "bugday":
        # Kış rezervi: Eki-Şub
        win_s = pd.Timestamp(year + cfg["year_offset"], 10, 1)
        win_e = pd.Timestamp(year, 2, 28)
        winter = _safe_slice(sub, win_s, win_e)
        tp_winter_sum = float(winter["PRECTOTCORR"].sum())
    else:
        tp_winter_sum = float("nan")
    tp_flowering = float(fl["PRECTOTCORR"].sum()) if not fl.empty else 0.0
    gf = _safe_slice(sub, pd.Timestamp(year, *cfg["grain_fill"][0]),
                          pd.Timestamp(year, *cfg["grain_fill"][1]))
    tp_grain_fill = float(gf["PRECTOTCORR"].sum()) if not gf.empty else 0.0

    # Aridity proxy (Tp / ET-eq using radiation+temp)
    # FAO-56 reduced-set ET0 proxy: Hargreaves-like
    # ET0_pseudo = 0.0023 × (Tmean + 17.8) × (Tmax-Tmin)^0.5 × Ra
    # Burada Ra yerine ALLSKY_SFC_SW_DWN (MJ/m²/day eşdeğeri).
    rad = sub["ALLSKY_SFC_SW_DWN"].clip(lower=0.0)
    et0 = 0.0023 * (sub["T2M"] + 17.8) * np.sqrt(
        (sub["T2M_MAX"] - sub["T2M_MIN"]).clip(lower=0.0)) * rad
    et0_sum = float(et0.sum())
    aridity_index = (tp_season_sum / et0_sum) if et0_sum > 0 else float("nan")

    # Temperature stats
    t2m_flowering_mean = float(fl["T2M"].mean()) if not fl.empty else float("nan")
    t2m_flowering_max  = float(fl["T2M_MAX"].mean()) if not fl.empty else float("nan")
    tdiff_mean = float((sub["T2M_MAX"] - sub["T2M_MIN"]).mean())

    # Radiation
    ssr_flowering_sum = float(fl["ALLSKY_SFC_SW_DWN"].sum()) if not fl.empty else 0.0
    ssr_season_sum    = float(rad.sum())

    return {
        "gdd_cum_season":   gdd_cum_season,
        "gdd_flowering":    gdd_flowering,
        "vernalization_days": vern_days,
        "tp_season_sum":    tp_season_sum,
        "tp_winter_sum":    tp_winter_sum,
        "tp_flowering":     tp_flowering,
        "tp_grain_fill":    tp_grain_fill,
        "aridity_index":    aridity_index,
        "heat_stress_days": heat_stress_days,
        "t2m_flowering_mean": t2m_flowering_mean,
        "t2m_flowering_max":  t2m_flowering_max,
        "tdiff_mean":       tdiff_mean,
        "ssr_flowering_sum":ssr_flowering_sum,
        "ssr_season_sum":   ssr_season_sum,
    }


def ndvi_features(ndvi_df: pd.DataFrame, year: int, crop: str
                  ) -> Optional[dict]:
    cfg = SEASON[crop]
    s, e = _season_window(year, crop)
    sub = _safe_slice(ndvi_df, s, e)
    if len(sub) < 8:                # min ~4 ay × 2 composite/ay
        return None
    ndvi = sub["ndvi_mean"].astype(float)

    fl_s = pd.Timestamp(year, *cfg["flowering"][0])
    fl_e = pd.Timestamp(year, *cfg["flowering"][1])
    gf_s = pd.Timestamp(year, *cfg["grain_fill"][0])
    gf_e = pd.Timestamp(year, *cfg["grain_fill"][1])
    fl = _safe_slice(sub, fl_s, fl_e)
    gf = _safe_slice(sub, gf_s, gf_e)

    # İlkbahar yeşillenme eğimi: Mart 1 → ndvi peak tarihi arası lineer eğim
    spr_s = pd.Timestamp(year, 3, 1)
    spr_e = pd.Timestamp(year, 6, 30)
    spr = _safe_slice(sub, spr_s, spr_e)
    if len(spr) >= 2:
        x = (spr["date"] - spr["date"].iloc[0]).dt.days.values.astype(float)
        y = spr["ndvi_mean"].astype(float).values
        slope = float(np.polyfit(x, y, 1)[0]) if x.var() > 0 else float("nan")
    else:
        slope = float("nan")

    return {
        "ndvi_max":           float(np.nanmax(ndvi)),
        "ndvi_mean_season":   float(np.nanmean(ndvi)),
        "ndvi_integral":      float(np.nansum(ndvi)),
        "ndvi_flowering":     float(fl["ndvi_mean"].mean()) if not fl.empty else float("nan"),
        "ndvi_grain_fill":    float(gf["ndvi_mean"].mean()) if not gf.empty else float("nan"),
        "ndvi_spring_slope":  slope,
        "greenness_days":     int((ndvi > NDVI_GREENNESS).sum() * 16),   # 16-gün composite
    }


def soil_features(soil_row: pd.Series) -> dict:
    """Static soil features; ilçe-bazlı."""
    out = {}
    for prop in ("clay", "sand", "silt", "phh2o", "soc", "awc"):
        for depth in ("0-5cm", "5-15cm", "15-30cm"):
            key = f"{prop}_{depth}"
            out[key] = float(soil_row[key]) if pd.notna(soil_row.get(key)) else float("nan")
    return out


# ---------------------------------------------------------------------------
def _load_climate(ilce_id: int) -> Optional[pd.DataFrame]:
    p = CLIMATE_DIR / f"nasapower_ilce_{ilce_id}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def _load_ndvi(ilce_id: int) -> Optional[pd.DataFrame]:
    p = NDVI_DIR / f"ndvi_{ilce_id}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return df


# ---------------------------------------------------------------------------
def build_layers() -> dict[str, pd.DataFrame]:
    yields = pd.read_csv(TUIK_DIR / "tuik_ilce_yields_clean.csv")
    yields["year"] = yields["year"].astype(int)
    coords = pd.read_csv(TUIK_DIR / "ilce_coords.csv")
    yields = yields.merge(coords[["ilce_id", "lat", "lon"]], on="ilce_id", how="left")

    soil_df = pd.read_csv(SOIL_PATH) if SOIL_PATH.exists() else None
    has_soil = soil_df is not None

    rows_a, rows_b, rows_c = [], [], []
    n_ndvi_missing_total = 0
    n_climate_missing_total = 0
    for (ilce_id, crop), grp in yields.groupby(["ilce_id", "crop"]):
        clim = _load_climate(int(ilce_id))
        if clim is None or clim.empty:
            n_climate_missing_total += len(grp); continue
        ndvi_df = _load_ndvi(int(ilce_id))
        soil_row = (soil_df.loc[soil_df["ilce_id"] == ilce_id].iloc[0]
                    if has_soil and (soil_df["ilce_id"] == ilce_id).any() else None)
        for _, r in grp.iterrows():
            base = {"ilce_id": int(ilce_id), "ilce": r["ilce"], "il": r["il"],
                    "year": int(r["year"]), "crop": crop,
                    "verim_kg_da": float(r["verim_kg_da"])}
            cfeat = climate_features(clim, int(r["year"]), crop)
            if cfeat is None:
                continue
            rec_a = {**base, **cfeat}
            rows_a.append(rec_a)

            # Layer B requires NDVI for that (ilce, year)
            ndvi_feats = (ndvi_features(ndvi_df, int(r["year"]), crop)
                          if ndvi_df is not None else None)
            if ndvi_feats is None:
                n_ndvi_missing_total += 1
                continue
            rec_b = {**rec_a, **ndvi_feats}
            rows_b.append(rec_b)

            if soil_row is not None:
                rec_c = {**rec_b, **soil_features(soil_row)}
                rows_c.append(rec_c)

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_c = pd.DataFrame(rows_c)
    logger.info("Layer A: %d rows", len(df_a))
    logger.info("Layer B: %d rows  (NDVI missing skipped: %d)",
                len(df_b), n_ndvi_missing_total)
    logger.info("Layer C: %d rows", len(df_c))
    return {"A": df_a, "B": df_b, "C": df_c,
            "_skipped_no_ndvi": n_ndvi_missing_total,
            "_skipped_no_climate": n_climate_missing_total}


# ---------------------------------------------------------------------------
def main() -> None:
    layers = build_layers()
    for name, df in layers.items():
        if name.startswith("_") or df.empty:
            continue
        out = OUT_DIR / f"calibration_features_layer{name}.csv"
        df.to_csv(out, index=False)
        logger.info("yazıldı: %s (%d × %d)", out.name, df.shape[0], df.shape[1])

    # QA Markdown
    md = ["# ÇP-2.5 — Görev 4: Sezonluk Feature QA", "",
          f"## Layer A (climate-only): n={len(layers['A'])}",
          f"## Layer B (+ NDVI):       n={len(layers['B'])}  (NDVI eksik: {layers['_skipped_no_ndvi']})",
          f"## Layer C (+ soil):       n={len(layers['C'])}",
          ""]
    if not layers["A"].empty:
        md.append("### Layer A — climate features (her ürün için)")
        for crop in sorted(layers["A"]["crop"].unique()):
            sub = layers["A"][layers["A"]["crop"] == crop]
            md.append(f"\n**{crop}** (n={len(sub)})")
            num = sub.select_dtypes(include=[np.number]).drop(
                columns=["ilce_id", "year"], errors="ignore")
            md.append("\n| Feature | Mean | Std | Min | Max | NaN |")
            md.append("|---|---|---|---|---|---|")
            for col in num.columns:
                s = num[col]
                md.append(f"| {col} | {s.mean():.2f} | {s.std():.2f} | "
                          f"{s.min():.2f} | {s.max():.2f} | {s.isna().sum()} |")
    (REPORT_DIR / "04_features_qa.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== ÖZET ===")
    print(f"Layer A (climate-only): {len(layers['A'])} satır")
    print(f"Layer B (+ NDVI)      : {len(layers['B'])} satır (NDVI ETL ilerledikçe büyür)")
    print(f"Layer C (+ soil)      : {len(layers['C'])} satır")


if __name__ == "__main__":
    main()
