"""ÇP-2.5 / Görev 7 — Katman C (Climate + NDVI + Soil) + Stacking Ensemble.

H3 hipotezi: Layer C, Layer B'yi outperform eder (ΔR² ≥ 0.05).
Ek olarak **stacking ensemble** kurulur (RF + XGBoost + GPR → Ridge meta).

Çıktılar
--------
* ``models/cp25/layer_c_{crop}.pkl`` (best single + stacking ensemble)
* ``reports/cp25/07_layer_c_results.{md,csv}``
* ``reports/cp25/07_h3_hypothesis.md``
"""

from __future__ import annotations

import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (RandomForestRegressor, StackingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger("cp25.task07")
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

FEATURES_A = [
    "gdd_cum_season", "gdd_flowering", "vernalization_days",
    "tp_season_sum", "tp_winter_sum", "tp_flowering", "tp_grain_fill",
    "aridity_index", "heat_stress_days",
    "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean",
    "ssr_flowering_sum", "ssr_season_sum",
]
FEATURES_NDVI = ["ndvi_max", "ndvi_mean_season", "ndvi_integral",
                  "ndvi_flowering", "ndvi_grain_fill", "ndvi_spring_slope",
                  "greenness_days"]
FEATURES_SOIL = ["clay_0-5cm", "sand_0-5cm", "silt_0-5cm",
                  "phh2o_0-5cm", "soc_0-5cm", "awc_0-5cm"]
FEATURES_C = FEATURES_A + FEATURES_NDVI + FEATURES_SOIL

ACCEPT = {
    "bugday":          {"r2": 0.50, "ss":  0.30},
    "aycicegi_yaglik": {"r2": 0.60, "ss":  0.40},
}
_NEEDS_SCALER = {"pls", "elastic_net", "gpr", "stacking"}


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(0.0 if np.isnan(med) else med)
    return X


def _b0_climatology_rmse(df: pd.DataFrame) -> float:
    preds = []
    for (il, crop), grp in df.groupby(["ilce_id", "crop"]):
        for idx, row in grp.iterrows():
            if (grp.index != idx).sum() == 0: continue
            preds.append((row["verim_kg_da"], grp.loc[grp.index != idx,
                            "verim_kg_da"].mean()))
    if not preds: return 1.0
    y_t, y_p = zip(*preds)
    return float(np.sqrt(mean_squared_error(np.array(y_t), np.array(y_p))))


def _make_model(name: str):
    if name == "pls":         return PLSRegression(n_components=3, scale=True)
    if name == "elastic_net": return ElasticNet(alpha=1.0, l1_ratio=0.5,
                                                 max_iter=10000, random_state=SEED)
    if name == "random_forest": return RandomForestRegressor(
        n_estimators=300, max_depth=5, random_state=SEED, n_jobs=-1)
    if name == "xgboost": return XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED,
        n_jobs=-1, verbosity=0)
    if name == "gpr":
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         random_state=SEED, alpha=1e-4)
    if name == "stacking":
        return StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=300, max_depth=5,
                                              random_state=SEED, n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=SEED,
                                      n_jobs=-1, verbosity=0)),
                ("gpr", GaussianProcessRegressor(
                    kernel=Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0),
                    normalize_y=True, random_state=SEED, alpha=1e-4)),
            ],
            final_estimator=Ridge(alpha=1.0, random_state=SEED),
            n_jobs=1, cv=3, passthrough=False)
    raise ValueError(name)


def _cv_predict(model_name, X, y, groups):
    preds = np.zeros_like(y, dtype=float)
    used = np.zeros_like(y, dtype=bool)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]; y_tr = y[tr]
        if model_name in _NEEDS_SCALER:
            sc = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        else:
            X_tr_s, X_te_s = X_tr.values, X_te.values
        m = _make_model(model_name); m.fit(X_tr_s, y_tr)
        p = m.predict(X_te_s)
        if p.ndim > 1: p = p.ravel()
        preds[te] = p; used[te] = True
    if not used.all():
        preds[~used] = float(np.mean(y))
    return preds


def _metrics(y_true, y_pred, rmse_b0):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_kg_da": rmse,
        "mae_kg_da": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "bias_kg_da": float(np.mean(y_pred - y_true)),
        "ss_vs_b0": float(1.0 - rmse / rmse_b0) if rmse_b0 > 0 else None,
    }


def _block_groups(df, n_year_blocks=5, n_clusters=5):
    years = df["year"].astype(int).values
    yr_bins = np.linspace(years.min(), years.max() + 1, n_year_blocks + 1)
    yr_block = np.digitize(years, yr_bins[1:-1])
    coords = pd.read_csv(COORDS_PATH)[["ilce_id", "lat", "lon"]]
    df_loc = df.merge(coords, on="ilce_id", how="left")
    n_eff = min(n_clusters, df_loc["ilce_id"].nunique())
    km = KMeans(n_clusters=n_eff, random_state=SEED, n_init=10)
    sp = km.fit_predict(df_loc[["lat","lon"]].values)
    return yr_block * n_eff + sp


def evaluate_one_crop(df, crop_short):
    crop_full = "bugday" if crop_short == "bugday" else "aycicegi_yaglik"
    sub = df[df["crop"] == crop_full].copy().reset_index(drop=True)
    if len(sub) < 30:
        logger.warning("[%s] n=%d yetersiz", crop_short, len(sub)); return None
    rmse_b0 = _b0_climatology_rmse(sub)
    logger.info("[%s] n=%d  B0 RMSE=%.2f", crop_short, len(sub), rmse_b0)

    X = _impute(sub[FEATURES_C].astype(float))
    y = sub["verim_kg_da"].astype(float).values
    year_g = sub["year"].astype(int).values
    ilce_g = sub["ilce_id"].astype(int).values
    block_g = _block_groups(sub)
    cvs = {"LOYO": year_g, "LOILO": ilce_g, "Spatiotemporal": block_g}
    # Stacking sadece LOYO için (en pahalı CV — yine de zaman içinde)
    model_names = ["pls", "elastic_net", "random_forest", "xgboost", "gpr", "stacking"]

    results = []
    loyo_preds = {}
    for cv_name, gr in cvs.items():
        for mn in model_names:
            if mn == "stacking" and cv_name != "LOYO":
                continue                                              # zaman budget
            t0 = time.time()
            try:
                preds = _cv_predict(mn, X, y, gr)
            except Exception as exc:                                  # noqa: BLE001
                logger.error("[%s/%s/%s] %s", crop_short, cv_name, mn, exc)
                continue
            m = _metrics(y, preds, rmse_b0)
            m.update({"crop": crop_short, "cv": cv_name, "model": mn,
                      "n": len(y), "elapsed_s": round(time.time() - t0, 1)})
            results.append(m)
            ss = f"{m['ss_vs_b0']:+.3f}" if m['ss_vs_b0'] is not None else "—"
            logger.info("  %-13s/%s R²=%+.3f RMSE=%.1f SS=%s (%.1fs)",
                        mn, cv_name, m["r2"], m["rmse_kg_da"], ss, m["elapsed_s"])
            if cv_name == "LOYO":
                loyo_preds[mn] = preds
    rows = pd.DataFrame(results)
    loyo = rows[rows["cv"] == "LOYO"]
    if loyo.empty: return None
    champ_name = loyo.sort_values("rmse_kg_da").iloc[0]["model"]
    champ_metrics = loyo[loyo["model"] == champ_name].iloc[0].to_dict()
    logger.info("[%s] CHAMPION (LOYO) → %s R²=%+.3f", crop_short, champ_name,
                champ_metrics["r2"])

    if champ_name in _NEEDS_SCALER:
        sc = StandardScaler().fit(X)
        final = _make_model(champ_name).fit(sc.transform(X), y); scaler = sc
    else:
        final = _make_model(champ_name).fit(X.values, y); scaler = None

    thr = ACCEPT[crop_full]
    passed = (champ_metrics["r2"] >= thr["r2"] and
              champ_metrics["ss_vs_b0"] >= thr["ss"])
    bundle = {
        "model": final, "scaler": scaler, "champion_name": champ_name,
        "feature_cols": FEATURES_C,
        "metrics_loyo": champ_metrics, "all_results": results,
        "rmse_b0": rmse_b0, "crop": crop_short, "n_samples": len(y),
        "train_date_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "cp25-v2.0-layerC",
        "acceptance": {"criteria": thr, "passed": passed},
    }
    with (MODELS_DIR / f"layer_c_{crop_short}.pkl").open("wb") as fh:
        pickle.dump(bundle, fh)

    if champ_name in loyo_preds:
        pred_df = sub[["ilce_id","ilce","il","year","crop","verim_kg_da"]].copy()
        pred_df["yield_pred_loyo"] = loyo_preds[champ_name]
        pred_df["abs_error"] = (pred_df["verim_kg_da"] - pred_df["yield_pred_loyo"]).abs()
        pred_df.to_csv(REPORT_DIR / f"07_loocv_predictions_{crop_short}.csv", index=False)

    return {"crop": crop_short, "champion": champ_name, "metrics": champ_metrics,
            "results": results, "n": len(y), "passed": passed}


def main() -> None:
    df = pd.read_csv(DATA_DIR / "calibration_features_layerC.csv")
    logger.info("Layer C: %d rows × %d cols", df.shape[0], df.shape[1])
    if len(df) < 30:
        logger.error("Layer C çok küçük (n=%d)", len(df)); return

    all_results = []
    summary = {}
    for c in ("bugday", "aycicegi"):
        info = evaluate_one_crop(df, c)
        if info:
            summary[c] = info; all_results.extend(info["results"])

    pd.DataFrame(all_results).round(3).to_csv(
        REPORT_DIR / "07_layer_c_results.csv", index=False)

    # H3 vs Layer B
    md = ["# ÇP-2.5 — Görev 7: Layer C (Climate + NDVI + Soil) + H3 Testi", ""]
    lb_path = REPORT_DIR / "06_layer_b_results.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        md.append("## H3 — Multimodal Füzyon Marjinal (ΔR² ≥ 0.05 vs Layer B)")
        md.append("")
        md.append("| Ürün | CV | LB Champ R² | LC Champ R² | ΔR² | H3 PASS? |")
        md.append("|---|---|---|---|---|---|")
        for c, info in summary.items():
            for cv in ("LOYO", "LOILO", "Spatiotemporal"):
                lb_sub = lb[(lb["crop"] == c) & (lb["cv"] == cv)]
                lc_sub = pd.DataFrame(info["results"])
                lc_cv = lc_sub[lc_sub["cv"] == cv]
                if lb_sub.empty or lc_cv.empty: continue
                r2_lb = lb_sub.sort_values("rmse_kg_da").iloc[0]["r2"]
                r2_lc = lc_cv.sort_values("rmse_kg_da").iloc[0]["r2"]
                dr2 = r2_lc - r2_lb
                h3 = "✅" if dr2 >= 0.05 else "❌"
                md.append(f"| {c} | {cv} | {r2_lb:+.3f} | {r2_lc:+.3f} | "
                          f"{dr2:+.3f} | {h3} |")
        md.append("")
    for c, info in summary.items():
        thr = ACCEPT["bugday" if c == "bugday" else "aycicegi_yaglik"]
        passed = "✅ PASS" if info["passed"] else "❌ FAIL"
        m = info["metrics"]
        md.append(f"## {c} (n={info['n']})")
        md.append(f"- Champion (LOYO): **{info['champion']}**")
        md.append(f"- R²={m['r2']:+.3f}, RMSE={m['rmse_kg_da']:.1f}, "
                  f"SS={m['ss_vs_b0']:+.3f}")
        md.append(f"- Kabul (R²≥{thr['r2']}, SS≥{thr['ss']}): {passed}")
        md.append("")
        md.append("### Tüm model × CV matrisi")
        md.append("")
        md.append("| Model | CV | R² | RMSE | SS |")
        md.append("|---|---|---|---|---|")
        for r in sorted(info["results"], key=lambda x: (x["model"], x["cv"])):
            ss = f"{r['ss_vs_b0']:+.3f}" if r['ss_vs_b0'] is not None else "—"
            md.append(f"| {r['model']} | {r['cv']} | {r['r2']:+.3f} | "
                      f"{r['rmse_kg_da']:.1f} | {ss} |")
        md.append("")
    (REPORT_DIR / "07_layer_c_results.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("rapor → 07_layer_c_results.md")
    print("\n=== ÖZET ===")
    for c, info in summary.items():
        v = "PASS" if info["passed"] else "FAIL"
        print(f"{c:10s}: {info['champion']:14s} R²={info['metrics']['r2']:+.3f} "
              f"SS={info['metrics']['ss_vs_b0']:+.3f} [{v}]")


if __name__ == "__main__":
    main()
