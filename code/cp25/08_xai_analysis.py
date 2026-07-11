"""ÇP-2.5 / Görev 8 — XAI / Yorumlanabilirlik (SHAP + PDP + Permutation).

Her katman şampiyon modeli için (tree-based en uygun — XGBoost):
* SHAP global summary plot
* SHAP local waterfall (anomali vakaları)
* Partial Dependence Plots (top-5 özellik)
* Permutation feature importance

Lischeid 2022 itirazına ("ML alone is not enough") cevap.

Layer A ile başlar; Layer B/C bittikçe aynı script ile çalıştırılır.

Çıktılar
--------
* ``reports/cp25/fig_shap_summary_{layer}_{crop}.png``
* ``reports/cp25/fig_shap_anomaly_{layer}_{ilce}_{year}.png``
* ``reports/cp25/fig_pdp_{layer}_{crop}.png``
* ``reports/cp25/08_xai_{layer}.md``
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from xgboost import XGBRegressor

logger = logging.getLogger("cp25.task08")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models" / "cp25"
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

FEATURES_LAYER_A = [
    "gdd_cum_season", "gdd_flowering", "vernalization_days",
    "tp_season_sum", "tp_winter_sum", "tp_flowering", "tp_grain_fill",
    "aridity_index", "heat_stress_days",
    "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean",
    "ssr_flowering_sum", "ssr_season_sum",
]
FEATURES_LAYER_B = FEATURES_LAYER_A + [
    "ndvi_max", "ndvi_mean_season", "ndvi_integral",
    "ndvi_flowering", "ndvi_grain_fill", "ndvi_spring_slope",
    "greenness_days",
]
FEATURES_LAYER_C = FEATURES_LAYER_B + [
    "clay_0-5cm", "sand_0-5cm", "silt_0-5cm", "phh2o_0-5cm",
    "soc_0-5cm", "awc_0-5cm",
]


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(0.0 if np.isnan(med) else med)
    return X


def _train_tree_for_xai(df: pd.DataFrame, features: list[str]) -> tuple:
    """Train XGBoost on full data for SHAP/PDP (no CV here — interpretive)."""
    X = _impute(df[features].astype(float))
    y = df["verim_kg_da"].astype(float).values
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         random_state=SEED, n_jobs=-1, verbosity=0)
    model.fit(X.values, y)
    return model, X, y


# ---------------------------------------------------------------------------
def shap_summary(model, X: pd.DataFrame, out_path: Path) -> dict:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.values)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close()
    abs_mean = np.abs(sv).mean(axis=0)
    return dict(sorted(zip(X.columns, abs_mean.tolist()),
                       key=lambda kv: -kv[1]))


def shap_anomaly_waterfall(model, X: pd.DataFrame, y: np.ndarray,
                            df: pd.DataFrame, anomalies: list[tuple],
                            layer: str, crop: str) -> None:
    """Belirli (ilçe, yıl) anomali satırı için waterfall plot."""
    explainer = shap.TreeExplainer(model)
    for ilce, year in anomalies:
        mask = (df["ilce"].str.lower() == ilce.lower()) & (df["year"] == year)
        if not mask.any():
            logger.warning("anomaly %s %d Layer %s: satır yok", ilce, year, layer)
            continue
        idx = df[mask].index[0]
        # Build SHAP Explanation object
        sv = explainer(X.iloc[[idx]])
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sv[0], show=False, max_display=12)
        plt.title(f"{crop} · {ilce} {year}  (gerçek: {y[idx]:.0f} kg/da)")
        plt.tight_layout()
        out = REPORT_DIR / f"fig_shap_anomaly_{layer}_{ilce.lower()}_{year}_{crop}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
        logger.info("waterfall → %s", out.name)


def pdp_top5(model, X: pd.DataFrame, shap_imp: dict, out_path: Path,
             crop: str, layer: str) -> None:
    top5 = list(shap_imp.keys())[:5]
    fig, ax = plt.subplots(figsize=(13, 4))
    try:
        disp = PartialDependenceDisplay.from_estimator(
            model, X, features=top5, kind="average",
            grid_resolution=20, n_jobs=-1, ax=ax)
        ax.figure.suptitle(f"PDP Top-5 — {crop} (Layer {layer})", y=1.04)
        plt.tight_layout()
        plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()
        logger.info("PDP → %s", out_path.name)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("PDP failed: %s", exc); plt.close()


def perm_importance(model, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    res = permutation_importance(model, X.values, y, n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
    return pd.DataFrame({
        "feature":      X.columns,
        "imp_mean":     res.importances_mean,
        "imp_std":      res.importances_std,
    }).sort_values("imp_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
def run_layer(layer: str, features: list[str], anomaly_cases: dict) -> None:
    layer_path = DATA_DIR / f"calibration_features_layer{layer}.csv"
    if not layer_path.exists():
        logger.warning("missing Layer %s features", layer); return
    df = pd.read_csv(layer_path)
    logger.info("Layer %s: %d rows", layer, len(df))

    md = [f"# ÇP-2.5 — Görev 8: XAI Layer {layer}", "",
          "## Yöntem", "",
          "- Tree-based XGBoost (n=200, max_depth=4) **tüm veride** fit edilir",
          "  (CV burada amaç değil; yorumlanabilirlik).",
          "- SHAP TreeExplainer global summary + waterfall (anomali vakaları).",
          "- PartialDependence top-5 feature.",
          "- Permutation importance (n_repeats=10).",
          ""]
    for crop_full in df["crop"].unique():
        crop_short = "bugday" if crop_full == "bugday" else "aycicegi"
        sub = df[df["crop"] == crop_full].reset_index(drop=True)
        if len(sub) < 30:
            md.append(f"## {crop_full} — n={len(sub)} yetersiz, atlandı.")
            continue

        # Filter features that exist in this subset
        feats_present = [f for f in features if f in sub.columns]
        model, X, y = _train_tree_for_xai(sub, feats_present)

        # SHAP global
        shap_path = REPORT_DIR / f"fig_shap_summary_{layer}_{crop_short}.png"
        shap_imp = shap_summary(model, X, shap_path)
        logger.info("[%s] top-5 SHAP: %s", crop_short,
                    {k: round(v, 2) for k, v in list(shap_imp.items())[:5]})

        # PDP top-5
        pdp_path = REPORT_DIR / f"fig_pdp_{layer}_{crop_short}.png"
        pdp_top5(model, X, shap_imp, pdp_path, crop_short, layer)

        # Permutation
        perm_df = perm_importance(model, X, y)
        perm_df.to_csv(REPORT_DIR / f"08_perm_importance_{layer}_{crop_short}.csv",
                       index=False)

        # Anomali waterfall
        shap_anomaly_waterfall(model, X, y, sub, anomaly_cases.get(crop_short, []),
                                layer, crop_short)

        md.append(f"## {crop_full} (n={len(sub)})")
        md.append("")
        md.append("### SHAP — Top-5 Global Importance")
        md.append("| Feature | mean(|SHAP|) |")
        md.append("|---|---|")
        for k, v in list(shap_imp.items())[:5]:
            md.append(f"| {k} | {v:.2f} |")
        md.append("")
        md.append("### Permutation Importance — Top-5")
        md.append("| Feature | imp_mean | imp_std |")
        md.append("|---|---|---|")
        for _, r in perm_df.head(5).iterrows():
            md.append(f"| {r['feature']} | {r['imp_mean']:.3f} | {r['imp_std']:.3f} |")
        md.append("")
        md.append("### Görseller")
        md.append(f"- `reports/cp25/{shap_path.name}` (SHAP summary)")
        md.append(f"- `reports/cp25/{pdp_path.name}` (PDP top-5)")
        for ilce, year in anomaly_cases.get(crop_short, []):
            f = f"fig_shap_anomaly_{layer}_{ilce.lower()}_{year}_{crop_short}.png"
            md.append(f"- `reports/cp25/{f}` (Local SHAP anomali)")
        md.append("")

    (REPORT_DIR / f"08_xai_{layer}.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("rapor → 08_xai_%s.md", layer)


# ---------------------------------------------------------------------------
ANOMALY_CASES = {
    "aycicegi": [("Çorlu", 2023), ("İpsala", 2025)],
    "bugday":   [("İpsala", 2021)],
}


def main(layer: str) -> None:
    feature_map = {"A": FEATURES_LAYER_A,
                   "B": FEATURES_LAYER_B,
                   "C": FEATURES_LAYER_C}
    run_layer(layer, feature_map[layer], ANOMALY_CASES)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="A", choices=["A", "B", "C"])
    args = p.parse_args()
    main(args.layer)
