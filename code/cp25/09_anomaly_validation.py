"""ÇP-2.5 / Görev 9 — Anomali Yıl Out-of-Sample Validasyonu.

`ilce_anomaly_years.csv` (|z|>1.5) kuraklık vakaları train'den çıkarılır,
Layer A/B/C şampiyon modeller kalanla eğitilir, anomali yıllarda test edilir.

H4 hipotezi: SS_anomaly > 0.30 (climatology'i geçer).

Çıktılar
--------
* ``reports/cp25/09_anomaly_validation_{layer}.md``
* ``reports/cp25/09_anomaly_predictions_{layer}.csv``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger("cp25.task09")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
TUIK_DIR     = PROJECT_ROOT / "data" / "external" / "tuik"
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

FEATURES_A = [
    "gdd_cum_season", "gdd_flowering", "vernalization_days",
    "tp_season_sum", "tp_winter_sum", "tp_flowering", "tp_grain_fill",
    "aridity_index", "heat_stress_days",
    "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean",
    "ssr_flowering_sum", "ssr_season_sum",
]
FEATURES_B = FEATURES_A + ["ndvi_max", "ndvi_mean_season", "ndvi_integral",
                            "ndvi_flowering", "ndvi_grain_fill",
                            "ndvi_spring_slope", "greenness_days"]
FEATURES_C = FEATURES_B + ["clay_0-5cm", "sand_0-5cm", "silt_0-5cm",
                            "phh2o_0-5cm", "soc_0-5cm", "awc_0-5cm"]


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(0.0 if np.isnan(med) else med)
    return X


def _train_xgb(X_tr, y_tr):
    return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        random_state=SEED, n_jobs=-1, verbosity=0).fit(X_tr.values, y_tr)


def run_layer(layer: str, features: list[str]) -> None:
    layer_path = DATA_DIR / f"calibration_features_layer{layer}.csv"
    if not layer_path.exists():
        logger.warning("Layer %s features yok", layer); return
    df = pd.read_csv(layer_path)
    anom = pd.read_csv(TUIK_DIR / "ilce_anomaly_years.csv")

    rows = []
    for crop_full in df["crop"].unique():
        crop_short = "bugday" if crop_full == "bugday" else "aycicegi"
        sub = df[df["crop"] == crop_full].copy()
        # Anomaly (ilce_id, year) çiftleri
        crop_urun = "Buğday" if crop_full == "bugday" else "Ayçiçeği (Yağlık)"
        an_sub = anom[anom["urun_tr"] == crop_urun]
        anom_keys = set(zip(an_sub["ilce_id"].astype(int),
                            an_sub["year"].astype(int)))

        sub["_is_anom"] = list(zip(sub["ilce_id"].astype(int),
                                    sub["year"].astype(int)))
        sub["_is_anom"] = sub["_is_anom"].isin(anom_keys)
        train = sub[~sub["_is_anom"]].copy()
        test  = sub[sub["_is_anom"]].copy()
        if test.empty:
            logger.info("[%s] Layer %s: anomali satır yok train'de", crop_short, layer)
            continue

        feats_present = [f for f in features if f in sub.columns]
        X_tr = _impute(train[feats_present].astype(float))
        X_te = _impute(test[feats_present].astype(float))
        y_tr = train["verim_kg_da"].astype(float).values
        y_te = test["verim_kg_da"].astype(float).values

        model = _train_xgb(X_tr, y_tr)
        y_pred = model.predict(X_te.values)

        # Climatology baseline on test (same anom subset)
        clim_preds = []
        for _, r in test.iterrows():
            mu = train[train["ilce_id"] == r["ilce_id"]]["verim_kg_da"].mean()
            if np.isnan(mu): mu = train["verim_kg_da"].mean()
            clim_preds.append(mu)
        clim_preds = np.array(clim_preds)

        rmse_model = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        rmse_clim  = float(np.sqrt(mean_squared_error(y_te, clim_preds)))
        ss_anom    = float(1.0 - rmse_model / rmse_clim) if rmse_clim > 0 else None
        r2_model   = float(r2_score(y_te, y_pred))
        mae_model  = float(mean_absolute_error(y_te, y_pred))

        # Per-row
        for (_, tr_row), pm, pc in zip(test.iterrows(), y_pred, clim_preds):
            rows.append({
                "layer": layer, "crop": crop_short,
                "ilce": tr_row["ilce"], "ilce_id": int(tr_row["ilce_id"]),
                "year": int(tr_row["year"]),
                "yield_real": float(tr_row["verim_kg_da"]),
                "yield_pred": float(pm),
                "yield_clim": float(pc),
                "abs_err_model": float(abs(tr_row["verim_kg_da"] - pm)),
                "abs_err_clim":  float(abs(tr_row["verim_kg_da"] - pc)),
            })
        logger.info("[%s Layer %s] n_test=%d, model RMSE=%.1f, clim RMSE=%.1f, "
                    "SS_anom=%.3f, R²=%.3f, MAE=%.1f",
                    crop_short, layer, len(test), rmse_model, rmse_clim,
                    ss_anom, r2_model, mae_model)

    if not rows:
        return
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(REPORT_DIR / f"09_anomaly_predictions_{layer}.csv", index=False)

    # ---- markdown ----
    md = [f"# ÇP-2.5 — Görev 9: Anomali Yıl Validasyonu (Layer {layer})", "",
          "## Yöntem", "",
          "- |z|>1.5 anomali yılları train setten ÇIKARILDI",
          "- Şampiyon model (XGBoost) kalanla eğitildi, anomali yıllarda test edildi",
          "- Climatology baseline: per-(ilçe, ürün) train ortalaması",
          "- H4 kabul: SS_anomaly > 0.30",
          ""]
    for crop in rows_df["crop"].unique():
        sub = rows_df[rows_df["crop"] == crop]
        rmse_m = float(np.sqrt((sub["yield_real"] - sub["yield_pred"]).pow(2).mean()))
        rmse_c = float(np.sqrt((sub["yield_real"] - sub["yield_clim"]).pow(2).mean()))
        ss = 1.0 - rmse_m / rmse_c if rmse_c > 0 else None
        passed = "✅" if (ss and ss > 0.30) else "❌"
        md.append(f"## {crop.upper()} — n={len(sub)} anomali satırı")
        md.append(f"- Model RMSE: {rmse_m:.1f} | Climatology RMSE: {rmse_c:.1f}")
        md.append(f"- **Skill Score = {ss:+.3f}**  | H4 kabul (>0.30): {passed}")
        md.append("")
        md.append("| İlçe | Yıl | Gerçek | Tahmin | Clim. | Hata (model) | Hata (clim) |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in sub.sort_values("year").iterrows():
            md.append(f"| {r['ilce']} | {r['year']} | {r['yield_real']:.0f} | "
                      f"{r['yield_pred']:.0f} | {r['yield_clim']:.0f} | "
                      f"{r['abs_err_model']:.1f} | {r['abs_err_clim']:.1f} |")
        md.append("")
    (REPORT_DIR / f"09_anomaly_validation_{layer}.md").write_text(
        "\n".join(md), encoding="utf-8")
    logger.info("rapor → 09_anomaly_validation_%s.md", layer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="A", choices=["A", "B", "C"])
    args = p.parse_args()
    feature_map = {"A": FEATURES_A, "B": FEATURES_B, "C": FEATURES_C}
    run_layer(args.layer, feature_map[args.layer])
