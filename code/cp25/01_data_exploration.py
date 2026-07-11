"""ÇP-2.5 / Görev 1 — Veri Schema Doğrulaması ve EDA.

İki dataseti yükler, schema'larını ve JOIN olabilirliğini test eder,
4 EDA görsel üretir.  Sonuç ``reports/cp25/01_data_exploration.md``.

Akademik öncelikler
-------------------
* Spatial granülarite tespiti (il / ilçe / koordinat) — H1 hipotezinin
  veri tabanını kuruyor.
* JOIN sonrası gerçek örnek sayısı (master matrix × TÜİK).
* Verim ile korele top-5 özelliği tespit (climate-only baseline yön).
* Veri kalitesi sorunları (eksik veri, duplicate, outlier) raporlanır.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("cp25.task01")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MFM_PATH     = PROJECT_ROOT / "data" / "processed" / "master_feature_matrix_2017_2024.csv"
TUIK_DIR     = PROJECT_ROOT / "data" / "external" / "tuik"
YIELDS_PATH  = TUIK_DIR / "tuik_ilce_yields_clean.csv"
COORDS_PATH  = TUIK_DIR / "ilce_coords.csv"
OUT_DIR      = PROJECT_ROOT / "reports" / "cp25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
def _schema(df: pd.DataFrame, name: str) -> dict:
    """Return a JSON-serialisable schema summary."""
    miss = df.isna().mean() * 100
    cols = []
    for c in df.columns:
        cols.append({
            "name": str(c),
            "dtype": str(df[c].dtype),
            "missing_pct": round(float(miss[c]), 2),
            "n_unique": int(df[c].nunique(dropna=True)),
            "sample": str(df[c].dropna().iloc[0]) if df[c].notna().any() else None,
        })
    return {"name": name, "shape": list(df.shape), "columns": cols}


def _spatial_granularity(df: pd.DataFrame) -> dict:
    """Detect whether MFM has il / ilce_id / parsel / coordinate keys."""
    cols = set(df.columns)
    detected = {
        "il":         "il" in cols,
        "ilce":       any(c in cols for c in ("ilce", "ilce_id", "ilce_adi")),
        "parsel":     any(c in cols for c in ("parsel_id", "site_id")),
        "lat_lon":    {"lat", "lon"}.issubset(cols),
        "single_pt":  False,
    }
    # Sniff: if soil columns are constant → single point
    soil_cand = [c for c in df.columns if c.startswith(("clay", "sand", "phh2o", "soil"))]
    if soil_cand:
        unique_counts = {c: int(df[c].nunique(dropna=True)) for c in soil_cand}
        if all(v <= 1 for v in unique_counts.values()):
            detected["single_pt"] = True
        detected["_soil_uniques"] = unique_counts
    return detected


def _join_feasibility(mfm: pd.DataFrame, yields: pd.DataFrame) -> dict:
    """Try to merge MFM with TÜİK yields on every plausible key."""
    res = {}
    mfm = mfm.copy()
    yields = yields.copy()

    # Ensure year col on MFM
    if "year" not in mfm.columns and "date" in mfm.columns:
        mfm["date"] = pd.to_datetime(mfm["date"])
        mfm["year"] = mfm["date"].dt.year

    yields_year = yields.copy()
    yields_year["year"] = yields_year["year"].astype(int)

    # Attempt 1: (ilce_id, year)
    if "ilce_id" in mfm.columns and "ilce_id" in yields_year.columns:
        m = mfm.merge(yields_year, on=["ilce_id", "year"], how="inner")
        res["join_ilce_year"] = {"n_rows": len(m),
                                 "n_unique_keys": m.groupby(["ilce_id","year"]).ngroups}
    else:
        res["join_ilce_year"] = "skipped (key missing)"

    # Attempt 2: (il, year)
    if "il" in mfm.columns and "il" in yields_year.columns:
        m = mfm.merge(yields_year, on=["il", "year"], how="inner")
        res["join_il_year"] = {"n_rows": len(m),
                               "n_unique_keys": m.groupby(["il","year"]).ngroups}
    else:
        res["join_il_year"] = "skipped (key missing)"

    # Attempt 3: temporal-only (year), centroid proxy
    if "year" in mfm.columns and "year" in yields_year.columns:
        # Cartesian by year then deduplicate per (year, ilce_id) on yields side
        mfm_year = mfm.groupby("year").mean(numeric_only=True).reset_index()
        m = yields_year.merge(mfm_year, on="year", how="inner")
        res["join_year_only_centroid_proxy"] = {
            "n_rows": len(m),
            "note": "Vize centroid features replicated to every (ilçe, year) label."}
    return res


def _correlation_top5(joined: pd.DataFrame) -> dict:
    """Compute top-5 features by abs corr with verim_kg_da (per crop)."""
    out = {}
    if "verim_kg_da" not in joined.columns or "crop" not in joined.columns:
        return out
    numeric = joined.select_dtypes(include=[np.number])
    for crop in joined["crop"].unique():
        sub = joined[joined["crop"] == crop]
        if len(sub) < 30:
            out[str(crop)] = {"note": f"n={len(sub)} insufficient"}
            continue
        s = sub[numeric.columns].corr()["verim_kg_da"].drop("verim_kg_da")
        top5 = s.abs().sort_values(ascending=False).head(5)
        out[str(crop)] = {
            "n": int(len(sub)),
            "top5": {k: round(float(s[k]), 3) for k in top5.index},
        }
    return out


# ---------------------------------------------------------------------------
def fig_yield_distribution(yields: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=yields, x="il", y="verim_kg_da",
                   hue="crop", split=False, inner="quartile")
    plt.title("Verim Dağılımı — İl × Ürün (2004-2025)")
    plt.xlabel("İl"); plt.ylabel("Verim (kg/dekar)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()


def fig_yield_vs_year(yields: pd.DataFrame, anom: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, crop, ttl in [
        (axes[0], "bugday",          "Buğday"),
        (axes[1], "aycicegi_yaglik", "Ayçiçeği"),
    ]:
        sub = yields[yields["crop"] == crop]
        sns.lineplot(data=sub, x="year", y="verim_kg_da",
                     hue="il", estimator="mean", errorbar="sd", ax=ax)
        # Anomaly markers
        an = anom[anom["urun_tr"].str.contains("Buğday" if crop=="bugday" else "Ayçiçeği",
                                                regex=False)]
        if not an.empty:
            for _, r in an.iterrows():
                ax.axvline(int(r["year"]), color="red", alpha=0.15, linestyle="--")
        ax.set_title(ttl); ax.set_xlabel("Yıl"); ax.set_ylabel("Verim (kg/da)")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def fig_correlation_matrix(joined: pd.DataFrame, out: Path) -> None:
    if joined.empty: return
    numeric = joined.select_dtypes(include=[np.number])
    # Skip useless columns
    keep = [c for c in numeric.columns if c not in ("year",)]
    corr = numeric[keep].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("Korelasyon Matrisi — JOIN edilmiş veri")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def fig_spatial_yield_map(yields: pd.DataFrame, coords: pd.DataFrame, out: Path) -> None:
    means = (yields.groupby(["ilce_id", "crop"])["verim_kg_da"].mean()
                   .reset_index().merge(coords, on="ilce_id", how="left"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, crop, ttl in [(axes[0], "bugday", "Buğday — 22yıl ortalama"),
                          (axes[1], "aycicegi_yaglik", "Ayçiçeği — 22yıl ortalama")]:
        sub = means[means["crop"] == crop].dropna(subset=["lat", "lon"])
        if sub.empty:
            ax.set_title(f"{ttl} — koordinat eksik"); continue
        sc = ax.scatter(sub["lon"], sub["lat"], c=sub["verim_kg_da"],
                        s=70, cmap="YlGn", edgecolor="black", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="kg/da")
        ax.set_title(ttl); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("loading datasets")
    mfm = pd.read_csv(MFM_PATH, parse_dates=["date"])
    yields = pd.read_csv(YIELDS_PATH)
    coords = pd.read_csv(COORDS_PATH)
    anom   = pd.read_csv(TUIK_DIR / "ilce_anomaly_years.csv")

    # ---- Schemas ----
    schemas = {
        "master_feature_matrix": _schema(mfm, "MFM"),
        "tuik_ilce_yields":      _schema(yields, "TUIK_ilce"),
        "ilce_coords":           _schema(coords, "ilce_coords"),
    }

    # ---- Spatial granularity ----
    gran = _spatial_granularity(mfm)
    logger.info("MFM spatial granularity: %s", gran)

    # ---- Coverage stats ----
    yields["year"] = yields["year"].astype(int)
    coverage = {
        "yields": {
            "n_rows":      int(len(yields)),
            "n_ilce":      int(yields["ilce_id"].nunique()),
            "n_il":        int(yields["il"].nunique()),
            "year_range":  [int(yields["year"].min()), int(yields["year"].max())],
            "crops":       sorted(yields["crop"].unique().tolist()),
            "rows_per_crop": yields.groupby("crop").size().to_dict(),
        },
        "coords": {
            "n_ilce":      int(coords["ilce_id"].nunique()),
            "lat_range":   [float(coords["lat"].min()), float(coords["lat"].max())],
            "lon_range":   [float(coords["lon"].min()), float(coords["lon"].max())],
        },
        "mfm": {
            "year_range":  [int(mfm["date"].dt.year.min()), int(mfm["date"].dt.year.max())],
            "n_days":      int(len(mfm)),
            "n_unique_dates": int(mfm["date"].nunique()),
        },
    }

    # ---- JOIN feasibility ----
    joins = _join_feasibility(mfm, yields)
    logger.info("JOIN feasibility: %s", joins)

    # ---- Build the actual joined table (proxy fallback for centroid) ----
    mfm2 = mfm.copy()
    mfm2["year"] = mfm2["date"].dt.year
    mfm_year = mfm2.groupby("year").mean(numeric_only=True).reset_index()
    joined = yields.merge(mfm_year, on="year", how="inner")
    logger.info("joined (year-only centroid proxy) shape: %s", joined.shape)

    # ---- Top-5 correlations ----
    top5 = _correlation_top5(joined)
    logger.info("top5 corr with verim: %s", top5)

    # ---- Anomaly years summary ----
    anomaly_summary = {
        "n_anomaly_rows":  int(len(anom)),
        "z_lt_minus_1p5":  int((anom["z"] < -1.5).sum()),
        "z_gt_plus_1p5":   int((anom["z"] >  1.5).sum()),
        "per_crop_count":  anom["urun_tr"].value_counts().to_dict(),
    }

    # ---- Data quality checks ----
    dq = {
        "yields_duplicates":       int(yields.duplicated(subset=["ilce_id","year","crop"]).sum()),
        "coords_missing_ilce":     sorted(list(set(yields["ilce_id"]) - set(coords["ilce_id"]))),
        "yield_lt_50_kg_da":       int((yields["verim_kg_da"] < 50).sum()),
        "yield_gt_700_kg_da":      int((yields["verim_kg_da"] > 700).sum()),
        "mfm_ndvi_int_missing_pct": round(float(mfm["NDVI_int"].isna().mean() * 100), 2),
        "mfm_NDVI_raw_missing_pct": round(float(mfm["NDVI"].isna().mean() * 100), 2),
    }

    # ---- Figures ----
    fig_yield_distribution(yields, OUT_DIR / "fig_yield_distribution.png")
    fig_yield_vs_year(yields, anom, OUT_DIR / "fig_yield_vs_year.png")
    fig_correlation_matrix(joined, OUT_DIR / "fig_correlation_matrix.png")
    fig_spatial_yield_map(yields, coords, OUT_DIR / "fig_spatial_yield_map.png")

    # ---- JSON summary ----
    summary = {
        "schemas": schemas,
        "spatial_granularity": gran,
        "coverage": coverage,
        "join_feasibility": joins,
        "top5_correlations": top5,
        "anomaly_summary": anomaly_summary,
        "data_quality": dq,
    }
    (OUT_DIR / "01_data_exploration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    # ---- Markdown report ----
    md = ["# ÇP-2.5 — Görev 1: Veri Schema Doğrulaması ve EDA", "",
          "## 1. Schema Özeti", ""]
    md.append(f"### `master_feature_matrix_2017_2024.csv`")
    md.append(f"- Shape: **{mfm.shape[0]} × {mfm.shape[1]}**")
    md.append(f"- Yıl aralığı: {coverage['mfm']['year_range']}")
    md.append(f"- Unique dates: {coverage['mfm']['n_unique_dates']}")
    md.append(f"- Spatial granularity tespiti: **{gran}**")
    md.append("")
    md.append(f"### `tuik_ilce_yields_clean.csv` (YENİ)")
    md.append(f"- Shape: **{yields.shape[0]} × {yields.shape[1]}**")
    md.append(f"- {coverage['yields']['n_ilce']} ilçe × {coverage['yields']['n_il']} il × "
              f"{len(coverage['yields']['crops'])} ürün × "
              f"{coverage['yields']['year_range'][1]-coverage['yields']['year_range'][0]+1} yıl")
    md.append(f"- Kolonlar: {[c['name'] for c in schemas['tuik_ilce_yields']['columns']]}")
    md.append(f"- Per crop: {coverage['yields']['rows_per_crop']}")
    md.append("")
    md.append(f"### `ilce_coords.csv`")
    md.append(f"- {coverage['coords']['n_ilce']} ilçe centroid (lat/lon)")
    md.append(f"- Lat: {coverage['coords']['lat_range']}, Lon: {coverage['coords']['lon_range']}")
    md.append("")
    md.append("## 2. JOIN Olabilirliği")
    md.append("")
    for k, v in joins.items():
        md.append(f"- **{k}**: `{v}`")
    md.append("")
    md.append("## 3. Top-5 Korelasyon (verim_kg_da'ya göre)")
    md.append("")
    for crop, info in top5.items():
        if "top5" in info:
            md.append(f"### {crop} (n={info['n']})")
            for k, v in info["top5"].items():
                md.append(f"- {k}: {v:+.3f}")
        else:
            md.append(f"### {crop}: {info.get('note','-')}")
        md.append("")
    md.append("## 4. Anomali Yıl Özeti")
    md.append("")
    md.append(f"- Toplam anomali satırı (|z|>1.5): **{anomaly_summary['n_anomaly_rows']}**")
    md.append(f"- z<-1.5 (kuraklık şokları): {anomaly_summary['z_lt_minus_1p5']}")
    md.append(f"- z>+1.5 (yüksek-verim): {anomaly_summary['z_gt_plus_1p5']}")
    md.append("")
    md.append("## 5. Veri Kalitesi")
    md.append("")
    for k, v in dq.items():
        md.append(f"- **{k}**: `{v}`")
    md.append("")
    md.append("## 6. EDA Görselleri Üretildi")
    md.append("")
    for f in ("fig_yield_distribution", "fig_yield_vs_year",
              "fig_correlation_matrix", "fig_spatial_yield_map"):
        md.append(f"- `reports/cp25/{f}.png`")
    (OUT_DIR / "01_data_exploration.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("report → %s", OUT_DIR / "01_data_exploration.md")


if __name__ == "__main__":
    main()
