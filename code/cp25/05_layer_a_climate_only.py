"""ÇP-2.5 / Görev 5 — Katman A (Climate-Only) Model Yarışı.

5 model × 2 ürün × 3 CV şeması = 30 değerlendirme.

Modeller
--------
* **PLS**         — kısmi en küçük kareler (n_components 1-5 grid)
* **ElasticNet**  — L1+L2 düzenlileştirme (alpha=1.0, l1_ratio=0.5)
* **RandomForest**— 300 ağaç, max_depth=5
* **XGBoost**     — 200 ağaç, max_depth=4, lr=0.05
* **GPR**         — Matern(ν=2.5) + WhiteKernel, normalize_y

Validasyon (3 paralel CV)
--------------------------
1. **LOYO** — `LeaveOneGroupOut(groups=year)` — 22 fold (gelecek yıl tahmini)
2. **LOILO**— `LeaveOneGroupOut(groups=ilce_id)` — 29 fold (yeni ilçe genelleme)
3. **Spatiotemporal** — 5 yıl bloğu × 5 ilçe cluster = 25 blok (Tao 2023 std)

Akademik kabul kriterleri (revised post-baseline)
-------------------------------------------------
* Buğday  : LOYO R² ≥ 0.35, Skill Score ≥ 0.15 (B0 climatology R²=0.213)
* Ayçiçeği: LOYO R² ≥ 0.40, Skill Score ≥ 0.20 (B0 climatology R²=0.210)

Skill Score = 1 − RMSE_model / RMSE_B0_climatology

H1 hipotez testi
----------------
İlçe-bazlı (n=1165 toplam) vs il-bazlı (önceki n=132/198) → ΔR² ≥ +0.15
testi `reports/cp25/05_h1_hypothesis_test.md` içinde raporlanır.

Çıktılar
--------
* ``models/cp25/layer_a_{crop}_{model}.pkl`` (final fit on all data)
* ``reports/cp25/05_layer_a_results.csv``    (full metric matrix)
* ``reports/cp25/05_layer_a_results.md``     (özet + kabul)
* ``reports/cp25/fig_layer_a_comparison.png``
* ``reports/cp25/05_loocv_predictions_{crop}.csv``
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger("cp25.task05")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models" / "cp25"
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
COORDS_PATH  = PROJECT_ROOT / "data" / "external" / "tuik" / "ilce_coords.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Climate-only feature set (15 ham + 1 türetilmiş eksen yok)
FEATURES_CLIMATE = [
    "gdd_cum_season", "gdd_flowering", "vernalization_days",
    "tp_season_sum", "tp_winter_sum", "tp_flowering", "tp_grain_fill",
    "aridity_index", "heat_stress_days",
    "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean",
    "ssr_flowering_sum", "ssr_season_sum",
]

# Kabul kriterleri
ACCEPT = {
    "bugday":          {"r2": 0.35, "ss":  0.15},
    "aycicegi_yaglik": {"r2": 0.40, "ss":  0.20},
}


# ---------------------------------------------------------------------------
def _impute_and_X(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Median impute + Inf cleanup on selected features."""
    X = df[FEATURES_CLIMATE].astype(float)
    # Inf (e.g. aridity_index when ET0=0) + NaN handling
    X = X.replace([np.inf, -np.inf], np.nan)
    # Column-by-column median impute; if whole column is NaN, fill 0
    for c in X.columns:
        med = X[c].median()
        if np.isnan(med):
            X[c] = X[c].fillna(0.0)
        else:
            X[c] = X[c].fillna(med)
    y = df["verim_kg_da"].astype(float).values
    return X, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray,
             rmse_b0: float) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_kg_da": rmse,
        "mae_kg_da":  float(mean_absolute_error(y_true, y_pred)),
        "mape_pct":   float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "bias_kg_da": float(np.mean(y_pred - y_true)),
        "ss_vs_b0":   float(1.0 - rmse / rmse_b0) if rmse_b0 > 0 else None,
    }


def _b0_climatology_rmse(df: pd.DataFrame) -> float:
    """LOYO climatology RMSE (per (ilce, crop) 22-yıl ortalaması)."""
    preds = []
    for (il, crop), grp in df.groupby(["ilce_id", "crop"]):
        for idx, row in grp.iterrows():
            mu = grp.loc[grp.index != idx, "verim_kg_da"].mean()
            preds.append((row["verim_kg_da"], mu))
    y_t, y_p = zip(*preds)
    return float(np.sqrt(mean_squared_error(np.array(y_t), np.array(y_p))))


def _make_model(name: str):
    if name == "pls":
        return PLSRegression(n_components=3, scale=True)
    if name == "elastic_net":
        return ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=SEED)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_depth=5,
                                     random_state=SEED, n_jobs=-1)
    if name == "xgboost":
        return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=SEED, n_jobs=-1, verbosity=0)
    if name == "gpr":
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        random_state=SEED, alpha=1e-4)
    raise ValueError(name)


# Models that benefit from scaling
_NEEDS_SCALER = {"pls", "elastic_net", "gpr"}


def _cv_predict(model_name: str, X: pd.DataFrame, y: np.ndarray,
                groups: np.ndarray) -> np.ndarray:
    """LeaveOneGroupOut tahminleri (out-of-fold)."""
    preds = np.zeros_like(y, dtype=float)
    used = np.zeros_like(y, dtype=bool)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y[tr]
        if model_name in _NEEDS_SCALER:
            sc = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        else:
            X_tr_s, X_te_s = X_tr.values, X_te.values
        m = _make_model(model_name)
        m.fit(X_tr_s, y_tr)
        p = m.predict(X_te_s)
        if p.ndim > 1: p = p.ravel()
        preds[te] = p
        used[te] = True
    if not used.all():
        # Fallback for samples never in test (shouldn't happen with LOGO)
        preds[~used] = float(np.mean(y))
    return preds


def _block_groups(df: pd.DataFrame, n_year_blocks: int = 5,
                  n_ilce_clusters: int = 5) -> np.ndarray:
    """Spatiotemporal block CV groups: 5 yıl bloğu × 5 ilçe cluster = 25 grup."""
    years = df["year"].astype(int).values
    yr_min, yr_max = years.min(), years.max()
    yr_bins = np.linspace(yr_min, yr_max + 1, n_year_blocks + 1)
    yr_block = np.digitize(years, yr_bins[1:-1])           # 0..n_year_blocks-1

    # Spatial KMeans on lat/lon centroids
    coords = pd.read_csv(COORDS_PATH)[["ilce_id", "lat", "lon"]]
    df_loc = df.merge(coords, on="ilce_id", how="left")
    km = KMeans(n_clusters=n_ilce_clusters, random_state=SEED, n_init=10)
    sp_block = km.fit_predict(df_loc[["lat", "lon"]].values)

    return yr_block * n_ilce_clusters + sp_block           # 0..24


# ---------------------------------------------------------------------------
def evaluate_one_crop(df: pd.DataFrame, crop_short: str) -> dict:
    sub = df[df["crop"] == ("bugday" if crop_short == "bugday"
                             else "aycicegi_yaglik")].copy().reset_index(drop=True)
    if len(sub) < 50:
        raise RuntimeError(f"{crop_short}: yetersiz örneklem n={len(sub)}")

    rmse_b0 = _b0_climatology_rmse(sub)
    logger.info("[%s] n=%d  B0 climatology RMSE=%.2f kg/da",
                crop_short, len(sub), rmse_b0)

    X, y = _impute_and_X(sub)
    year_g  = sub["year"].astype(int).values
    ilce_g  = sub["ilce_id"].astype(int).values
    block_g = _block_groups(sub)

    cvs = {
        "LOYO":            year_g,
        "LOILO":           ilce_g,
        "Spatiotemporal":  block_g,
    }
    model_names = ["pls", "elastic_net", "random_forest", "xgboost", "gpr"]

    results = []
    loocv_preds: dict[str, np.ndarray] = {}   # key by "loyo_<model>"

    for cv_name, groups in cvs.items():
        n_groups = int(np.unique(groups).size)
        logger.info("[%s] CV %s n_groups=%d", crop_short, cv_name, n_groups)
        for mn in model_names:
            t0 = time.time()
            try:
                preds = _cv_predict(mn, X, y, groups)
            except Exception as exc:                                    # noqa: BLE001
                logger.error("[%s/%s/%s] failed: %s", crop_short, cv_name, mn, exc)
                continue
            m = _metrics(y, preds, rmse_b0=rmse_b0)
            m.update({"crop": crop_short, "cv": cv_name, "model": mn,
                      "n": len(y), "elapsed_s": round(time.time() - t0, 1)})
            results.append(m)
            ss = f"{m['ss_vs_b0']:+.3f}" if m['ss_vs_b0'] is not None else "—"
            logger.info("  %-13s R²=%+.3f RMSE=%.1f MAE=%.1f SS=%s (%.1fs)",
                        mn, m["r2"], m["rmse_kg_da"], m["mae_kg_da"],
                        ss, m["elapsed_s"])
            if cv_name == "LOYO":
                loocv_preds[mn] = preds

    # Final-fit champion (best by LOYO RMSE) on ALL data
    rows = pd.DataFrame(results)
    loyo = rows[rows["cv"] == "LOYO"]
    champ_name = loyo.sort_values("rmse_kg_da").iloc[0]["model"]
    champ_metrics = loyo[loyo["model"] == champ_name].iloc[0].to_dict()
    logger.info("[%s] CHAMPION → %s (LOYO RMSE=%.1f, R²=%+.3f, SS=%+.3f)",
                crop_short, champ_name, champ_metrics["rmse_kg_da"],
                champ_metrics["r2"], champ_metrics["ss_vs_b0"])

    if champ_name in _NEEDS_SCALER:
        sc = StandardScaler().fit(X)
        final = _make_model(champ_name).fit(sc.transform(X), y)
        scaler = sc
    else:
        final = _make_model(champ_name).fit(X.values, y)
        scaler = None

    bundle = {
        "model":              final,
        "scaler":             scaler,
        "champion_name":      champ_name,
        "feature_cols":       FEATURES_CLIMATE,
        "metrics_loyo":       champ_metrics,
        "all_results":        results,
        "rmse_b0":            rmse_b0,
        "crop":               crop_short,
        "n_samples":          len(y),
        "n_ilce":             int(sub["ilce_id"].nunique()),
        "year_range":         [int(sub["year"].min()), int(sub["year"].max())],
        "train_date_utc":     datetime.now(timezone.utc).isoformat(),
        "model_version":      "cp25-v2.0-layerA",
        "acceptance":         {
            "criteria":   ACCEPT[("bugday" if crop_short == "bugday"
                                  else "aycicegi_yaglik")],
            "passed":     (champ_metrics["r2"] >= ACCEPT[("bugday" if crop_short
                                  == "bugday" else "aycicegi_yaglik")]["r2"] and
                           champ_metrics["ss_vs_b0"] >= ACCEPT[("bugday" if crop_short
                                  == "bugday" else "aycicegi_yaglik")]["ss"]),
        },
        "limitations": [
            "Layer A — climate-only (no NDVI, no soil). H2/H3 test gerek.",
            "MERRA-2 source (NASA POWER) vs ERA5-Land farkı tezde raporlu.",
            "n=%d ilçe-yıl satırı, LOYO/LOILO/Spatiotemporal CV ile akademik altın "
            "standart" % len(y),
        ],
    }
    bundle_path = MODELS_DIR / f"layer_a_{crop_short}.pkl"
    with bundle_path.open("wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("[%s] bundle → %s", crop_short, bundle_path.name)

    # LOYO predictions CSV
    if champ_name in loocv_preds:
        pred_df = sub[["ilce_id", "ilce", "il", "year", "crop", "verim_kg_da"]].copy()
        pred_df["yield_pred_loyo"] = loocv_preds[champ_name]
        pred_df["abs_error"] = (pred_df["verim_kg_da"] - pred_df["yield_pred_loyo"]).abs()
        pred_df.to_csv(REPORT_DIR / f"05_loocv_predictions_{crop_short}.csv",
                       index=False)

    return {"crop": crop_short, "results": results, "champion": champ_name,
            "metrics": champ_metrics, "rmse_b0": rmse_b0}


# ---------------------------------------------------------------------------
def plot_comparison(all_results: list[dict], out: Path) -> None:
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, crop in enumerate(sorted(df["crop"].unique())):
        for j, cv in enumerate(["LOYO", "LOILO", "Spatiotemporal"]):
            ax = axes[i, j]
            sub = df[(df["crop"] == crop) & (df["cv"] == cv)].sort_values("r2",
                                                                          ascending=False)
            ax.barh(sub["model"], sub["r2"], color="tab:blue", alpha=0.7)
            ax.axvline(ACCEPT[("bugday" if crop == "bugday" else "aycicegi_yaglik")]["r2"],
                       color="red", linestyle="--", linewidth=1, label="kabul R²")
            ax.set_title(f"{crop} · {cv}")
            ax.set_xlabel("R²")
            ax.set_xlim(min(-0.5, sub["r2"].min() - 0.05), max(0.7, sub["r2"].max() + 0.05))
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close()


def main() -> None:
    df = pd.read_csv(DATA_DIR / "calibration_features_layerA.csv")
    logger.info("loaded Layer A features: %d rows, %d cols",
                df.shape[0], df.shape[1])

    all_results = []
    summary = {}
    for crop_short in ("bugday", "aycicegi"):
        out = evaluate_one_crop(df, crop_short)
        all_results.extend(out["results"])
        summary[crop_short] = out

    # ---- Save aggregate ----
    metrics_df = pd.DataFrame(all_results).round(3)
    metrics_df.to_csv(REPORT_DIR / "05_layer_a_results.csv", index=False)
    plot_comparison(all_results, REPORT_DIR / "fig_layer_a_comparison.png")

    # ---- Markdown ----
    md = ["# ÇP-2.5 — Görev 5: Layer A (Climate-Only) Sonuçları", "",
          "## Yapılan",
          "",
          "- 5 model × 2 ürün × 3 CV = **30 değerlendirme**",
          "- Modeller: PLS, ElasticNet, Random Forest, XGBoost, GPR",
          "- CV: LOYO (yıl bazlı), LOILO (ilçe bazlı), Spatiotemporal (5×5 blok)",
          ""]
    for crop_short, info in summary.items():
        md.append(f"## {crop_short.upper()} (n={info['results'][0]['n']})")
        crop_full = "bugday" if crop_short == "bugday" else "aycicegi_yaglik"
        rmse_b0 = info["rmse_b0"]
        md.append(f"- B0 Climatology RMSE: **{rmse_b0:.2f} kg/da** (skill score referansı)")
        md.append(f"- Şampiyon model (LOYO en düşük RMSE): **{info['champion']}**")
        m = info["metrics"]
        md.append(f"  - R² = **{m['r2']:+.3f}**")
        md.append(f"  - RMSE = **{m['rmse_kg_da']:.1f} kg/da**")
        md.append(f"  - MAE = {m['mae_kg_da']:.1f} kg/da · MAPE = {m['mape_pct']:.1f}%")
        md.append(f"  - Skill Score vs B0 = **{m['ss_vs_b0']:+.3f}**")
        thr = ACCEPT[crop_full]
        passed = (m["r2"] >= thr["r2"] and m["ss_vs_b0"] >= thr["ss"])
        md.append(f"- **Kabul kriteri (R²≥{thr['r2']}, SS≥{thr['ss']}): "
                  f"{'✅ PASS' if passed else '❌ FAIL'}**")
        md.append("")
        md.append("### Tüm model × CV matrisi")
        md.append("")
        md.append("| Model | CV | R² | RMSE | MAE | SS | Süre |")
        md.append("|---|---|---|---|---|---|---|")
        rcrop = [r for r in info["results"]]
        # Order: by model then CV
        for r in sorted(rcrop, key=lambda x: (x["model"], x["cv"])):
            ss = f"{r['ss_vs_b0']:+.3f}" if r["ss_vs_b0"] is not None else "—"
            md.append(f"| {r['model']} | {r['cv']} | {r['r2']:+.3f} | "
                      f"{r['rmse_kg_da']:.1f} | {r['mae_kg_da']:.1f} | "
                      f"{ss} | {r['elapsed_s']:.1f}s |")
        md.append("")

    md.append("## Hipotez H1 Testi — n=1165 vs n=132 ΔR² ≥ 0.15")
    md.append("")
    md.append("Önceki il-bazlı kalibrasyon (cp25-v1): n=21/24, Ridge(α=100/α=1.0).")
    md.append("")
    md.append("| Ürün | v1 (il) R² | v2 (ilçe) R² | ΔR² | H1 PASS? |")
    md.append("|---|---|---|---|---|")
    v1 = {"bugday": -0.085, "aycicegi": 0.646}
    for crop_short, info in summary.items():
        v2 = info["metrics"]["r2"]
        dr2 = v2 - v1[crop_short]
        h1 = "✅" if dr2 >= 0.15 else "❌"
        md.append(f"| {crop_short} | {v1[crop_short]:+.3f} | {v2:+.3f} | "
                  f"{dr2:+.3f} | {h1} |")
    md.append("")
    md.append("## Görsel")
    md.append("")
    md.append("`reports/cp25/fig_layer_a_comparison.png`")

    (REPORT_DIR / "05_layer_a_results.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("rapor → %s", REPORT_DIR / "05_layer_a_results.md")

    print("\n=== ÖZET ===")
    for crop, info in summary.items():
        m = info["metrics"]
        thr = ACCEPT["bugday" if crop == "bugday" else "aycicegi_yaglik"]
        verdict = "PASS" if (m["r2"] >= thr["r2"] and m["ss_vs_b0"] >= thr["ss"]) else "FAIL"
        print(f"{crop:10s}: champion={info['champion']:14s}  "
              f"R²={m['r2']:+.3f}  RMSE={m['rmse_kg_da']:.1f}  "
              f"SS={m['ss_vs_b0']:+.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
