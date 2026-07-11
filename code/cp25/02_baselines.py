"""ÇP-2.5 / Görev 2 — Baseline modeller (B0/B1/B2/B3).

Hiçbir gelişmiş model bu 4 baseline'ı geçemiyorsa deploy edilemez.

Baseline'lar
------------
* **B0 — Naive Climatology**: ``(ilce, crop)`` 22-yıl ortalaması.  Kara çizgi.
* **B1 — Year-Only Trend**: ``yield ~ year`` per ``(ilce, crop)`` lineer regresyon.
* **B2 — Persistence**: ``yield(t) = yield(t-1)`` (t-1 yıl yoksa B0'a düş).
* **B3 — Climate-Mean Naive**: yağış-only regresyon (climate ETL beklemediği
  için TÜİK ekilen/hasat alanı + uretim_ton vekil olarak; gerçek climate
  Görev 3 sonrası ``05_layer_a_climate_only.py`` içinde test edilecek).

Validasyon
----------
LOYO (Leave-One-Year-Out) — 22 yıl × 29 ilçe = 1276 fold-tahmin per crop.
Bu, akademik altın standart; random k-fold **kullanılmaz**.

Çıktılar
--------
* ``models/cp25/baselines.pkl``                    — fit edilen helper'lar
* ``reports/cp25/02_baselines.csv``                — full LOYO sonuçları
* ``reports/cp25/02_baselines.md``                 — özet markdown
* ``reports/cp25/fig_baseline_comparison.png``     — bar chart
* ``reports/cp25/02_baseline_loyo_predictions.csv`` — her satır LOYO çıktısı
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

logger = logging.getLogger("cp25.task02")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUIK_DIR     = PROJECT_ROOT / "data" / "external" / "tuik"
YIELDS_PATH  = TUIK_DIR / "tuik_ilce_yields_clean.csv"
OUT_REPORTS  = PROJECT_ROOT / "reports" / "cp25"
OUT_MODELS   = PROJECT_ROOT / "models"  / "cp25"
OUT_REPORTS.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
             rmse_b0: float | None = None) -> dict:
    """Compute the standard CP-2.5 metric suite."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out = {
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse_kg_da": rmse,
        "mae_kg_da":  float(mean_absolute_error(y_true, y_pred)),
        "mape_pct":   float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "bias_kg_da": float(np.mean(y_pred - y_true)),
        "n":          int(len(y_true)),
    }
    out["skill_score_vs_B0"] = (
        float(1.0 - rmse / rmse_b0) if rmse_b0 and rmse_b0 > 0 else None)
    return out


# ---------------------------------------------------------------------------
# LOYO predictors
# ---------------------------------------------------------------------------
def loyo_climatology(df: pd.DataFrame) -> pd.Series:
    """B0 — (ilce, crop) 22-yıl ortalaması; test yılı dışlanır."""
    preds = pd.Series(index=df.index, dtype=float)
    for (ilce, crop), grp in df.groupby(["ilce_id", "crop"]):
        for idx, row in grp.iterrows():
            mask = (grp.index != idx)
            if mask.sum() == 0:
                preds.loc[idx] = np.nan
            else:
                preds.loc[idx] = grp.loc[mask, "verim_kg_da"].mean()
    return preds


def loyo_year_trend(df: pd.DataFrame) -> pd.Series:
    """B1 — yield ~ year lineer regresyon per (ilce, crop), LOYO."""
    preds = pd.Series(index=df.index, dtype=float)
    for (ilce, crop), grp in df.groupby(["ilce_id", "crop"]):
        if len(grp) < 4:
            preds.loc[grp.index] = grp["verim_kg_da"].mean()
            continue
        for idx, row in grp.iterrows():
            tr = grp.drop(idx)
            if len(tr) < 3:
                preds.loc[idx] = tr["verim_kg_da"].mean()
                continue
            X = tr[["year"]].values
            y = tr["verim_kg_da"].values
            try:
                m = LinearRegression().fit(X, y)
                preds.loc[idx] = float(m.predict([[row["year"]]])[0])
            except Exception:                                       # noqa: BLE001
                preds.loc[idx] = tr["verim_kg_da"].mean()
    return preds


def loyo_persistence(df: pd.DataFrame) -> pd.Series:
    """B2 — yield(t) = yield(t-1). Eğer t-1 yoksa B0 fallback."""
    sorted_df = df.sort_values(["ilce_id", "crop", "year"]).copy()
    sorted_df["prev_yield"] = (sorted_df.groupby(["ilce_id","crop"])["verim_kg_da"]
                                .shift(1))
    # Fallback: climatology of training years (B0)
    fallback = loyo_climatology(df)
    preds = sorted_df["prev_yield"].copy()
    preds.update(fallback[preds.isna()])
    return preds.reindex(df.index)


def loyo_climate_proxy(df: pd.DataFrame) -> pd.Series:
    """B3 — Climate-mean naive.  Climate ETL henüz yok, dolayısıyla
    ``ekilen_alan_da`` (yetiştirici tercihi) + ``year`` üzerinden vekil
    bir regresyon kurarız.  Bu, **gerçek climate B3 baseline'ı değildir**;
    Görev 3 ETL bitince ``05_layer_a_climate_only.py`` içinde gerçeği
    test edilecek.  Şimdilik tek amaç çerçeveyi kurmak."""
    preds = pd.Series(index=df.index, dtype=float)
    for (ilce, crop), grp in df.groupby(["ilce_id", "crop"]):
        if len(grp) < 5:
            preds.loc[grp.index] = grp["verim_kg_da"].mean()
            continue
        feats = ["ekilen_alan_da", "year"]
        for idx, row in grp.iterrows():
            tr = grp.drop(idx)
            if len(tr) < 3:
                preds.loc[idx] = tr["verim_kg_da"].mean()
                continue
            X = tr[feats].values
            y = tr["verim_kg_da"].values
            try:
                m = LinearRegression().fit(X, y)
                preds.loc[idx] = float(m.predict(row[feats].values.reshape(1, -1))[0])
            except Exception:                                       # noqa: BLE001
                preds.loc[idx] = tr["verim_kg_da"].mean()
    return preds


# ---------------------------------------------------------------------------
def evaluate_per_crop(df: pd.DataFrame) -> pd.DataFrame:
    """Run all baselines on the data, return tidy metrics table."""
    out_rows: list[dict] = []
    pred_records: list[pd.DataFrame] = []
    for crop in sorted(df["crop"].unique()):
        sub = df[df["crop"] == crop].copy().reset_index(drop=True)
        logger.info("=== %s (n=%d) ===", crop, len(sub))

        preds = {
            "B0_Climatology":      loyo_climatology(sub),
            "B1_YearTrend":        loyo_year_trend(sub),
            "B2_Persistence":      loyo_persistence(sub),
            "B3_ClimateProxy":     loyo_climate_proxy(sub),
        }

        # Drop rows where any predictor is NaN (small minority for B2)
        valid_idx = sub.index
        for s in preds.values():
            valid_idx = valid_idx.intersection(s.dropna().index)
        sub_v = sub.loc[valid_idx].reset_index(drop=True)
        preds_v = {k: v.loc[valid_idx].reset_index(drop=True) for k, v in preds.items()}
        y_true = sub_v["verim_kg_da"].values

        rmse_b0 = float(np.sqrt(mean_squared_error(y_true, preds_v["B0_Climatology"].values)))

        for name, pred in preds_v.items():
            m = _metrics(y_true, pred.values, rmse_b0=rmse_b0)
            m["model"] = name; m["crop"] = crop
            out_rows.append(m)
            logger.info("  %s  R²=%+.3f  RMSE=%.1f  MAE=%.1f  SS_vs_B0=%s",
                        name, m["r2"], m["rmse_kg_da"], m["mae_kg_da"],
                        f"{m['skill_score_vs_B0']:+.3f}"
                            if m["skill_score_vs_B0"] is not None else "N/A")

        # Record per-row predictions for downstream analysis
        rec = sub_v[["ilce_id", "il", "year", "crop", "verim_kg_da"]].copy()
        for name, pred in preds_v.items():
            rec[name] = pred.values
        pred_records.append(rec)

    metrics_df = pd.DataFrame(out_rows)[
        ["crop", "model", "n", "r2", "rmse_kg_da", "mae_kg_da",
         "mape_pct", "bias_kg_da", "skill_score_vs_B0"]
    ].round(3)
    return metrics_df, pd.concat(pred_records, ignore_index=True)


def plot_comparison(metrics: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in [
        (axes[0], "r2",         "R² (yüksek iyi)"),
        (axes[1], "rmse_kg_da", "RMSE kg/da (düşük iyi)"),
        (axes[2], "skill_score_vs_B0", "Skill Score vs B0 (>0 iyi)"),
    ]:
        for crop in sorted(metrics["crop"].unique()):
            sub = metrics[metrics["crop"] == crop]
            ax.bar([f"{m}\n{crop[:7]}" for m in sub["model"]],
                   sub[metric], alpha=0.7, label=crop)
        ax.set_title(title)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()


# ---------------------------------------------------------------------------
def main() -> None:
    df = pd.read_csv(YIELDS_PATH)
    df["year"] = df["year"].astype(int)
    logger.info("loaded %d rows from %s", len(df), YIELDS_PATH.name)

    metrics, preds = evaluate_per_crop(df)

    # ---- Save artifacts ----
    metrics.to_csv(OUT_REPORTS / "02_baselines.csv", index=False)
    preds.to_csv(OUT_REPORTS / "02_baseline_loyo_predictions.csv", index=False)

    bundle = {
        "metrics": metrics.to_dict(orient="records"),
        "predictor_functions": ["loyo_climatology", "loyo_year_trend",
                                "loyo_persistence", "loyo_climate_proxy"],
        "validation": "LOYO (Leave-One-Year-Out, in-place)",
        "n_rows_per_crop": df.groupby("crop").size().to_dict(),
        "n_ilce": int(df["ilce_id"].nunique()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "seed": SEED,
    }
    with (OUT_MODELS / "baselines.pkl").open("wb") as fh:
        pickle.dump(bundle, fh)

    # ---- Markdown ----
    md = ["# ÇP-2.5 — Görev 2: Baseline Modeller (LOYO)", "",
          "## Veri", "",
          f"- TÜİK ilçe-bazlı dataset: **{len(df)} satır**, "
          f"{df['ilce_id'].nunique()} ilçe, "
          f"{df['year'].min()}-{df['year'].max()}",
          ""]
    md.append("## Sonuçlar — LOYO (Leave-One-Year-Out)")
    md.append("")
    # Per crop table
    for crop in sorted(metrics["crop"].unique()):
        sub = metrics[metrics["crop"] == crop]
        md.append(f"### {crop} (n={int(sub['n'].iloc[0])})")
        md.append("")
        md.append("| Model | R² | RMSE | MAE | MAPE | Bias | Skill Score vs B0 |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            ss = f"{r['skill_score_vs_B0']:+.3f}" if pd.notna(r["skill_score_vs_B0"]) else "—"
            md.append(f"| {r['model']} | {r['r2']:+.3f} | {r['rmse_kg_da']:.1f} | "
                      f"{r['mae_kg_da']:.1f} | {r['mape_pct']:.1f}% | "
                      f"{r['bias_kg_da']:+.1f} | {ss} |")
        md.append("")
    md.append("## Kabul Kriteri")
    md.append("")
    md.append("Hiçbir ileri model bu baseline'ları yenmiyorsa **deploy edilemez**.")
    md.append("Özellikle B3 (climate-proxy) R²>0 olmalı (yoksa veri/kod hatası).")
    md.append("")
    md.append("⚠️ **Not**: B3 burada ETL beklemediği için TÜİK ekilen_alan_da + year"
              " üzerinden vekildir. Gerçek climate B3 baseline'ı Görev 3 ETL sonrası"
              " Katman A modellerinde test edilecek.")
    md.append("")
    md.append("## Görseller")
    md.append("- `reports/cp25/fig_baseline_comparison.png`")

    (OUT_REPORTS / "02_baselines.md").write_text("\n".join(md), encoding="utf-8")
    plot_comparison(metrics, OUT_REPORTS / "fig_baseline_comparison.png")

    logger.info("report → %s", OUT_REPORTS / "02_baselines.md")
    print("\n=== ÖZET ===")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
