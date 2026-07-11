"""ÇP-2.5 / Görev 10 — Belirsizlik Kantifikasyonu.

İki paralel yaklaşım:
1. **GPR analitik posterior**: mean ± 1.96σ (Bayesian doğal belirsizlik).
2. **Bootstrap intervals**: 1000 resample XGBoost, 2.5/97.5 percentile.

PICP (Prediction Interval Coverage Probability):
    PICP = mean(y_test ∈ [PI_l, PI_u]).  Hedef ≈ 0.95.

Sharpness:
    Sharpness = mean(PI_upper - PI_lower).

Çıktılar
--------
* ``reports/cp25/10_uncertainty_{layer}.md``
* ``reports/cp25/fig_uncertainty_calibration_{layer}.png`` (reliability diagram)
* ``reports/cp25/10_uncertainty_predictions_{layer}.csv``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger("cp25.task10")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
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


# ---------------------------------------------------------------------------
def loyo_gpr_intervals(X: pd.DataFrame, y: np.ndarray,
                       year_g: np.ndarray) -> np.ndarray:
    """LOYO mean + std from GPR; returns array shape (n,2) [mean,std]."""
    out = np.zeros((len(y), 2), dtype=float)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=year_g):
        sc = StandardScaler().fit(X.iloc[tr])
        Xt = sc.transform(X.iloc[tr]); Xe = sc.transform(X.iloc[te])
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        random_state=SEED, alpha=1e-4)
        gpr.fit(Xt, y[tr])
        mu, sd = gpr.predict(Xe, return_std=True)
        out[te, 0] = mu; out[te, 1] = sd
    return out


def loyo_bootstrap_xgb(X: pd.DataFrame, y: np.ndarray,
                       year_g: np.ndarray, n_boot: int = 200) -> np.ndarray:
    """LOYO bootstrap XGBoost: (n, 3) [mean, p2.5, p97.5]."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    out = np.zeros((n, 3), dtype=float)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=year_g):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y[tr]
        preds = np.zeros((n_boot, len(te)), dtype=float)
        for b in range(n_boot):
            idx = rng.choice(len(tr), len(tr), replace=True)
            m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              random_state=SEED + b, n_jobs=-1, verbosity=0)
            m.fit(X_tr.iloc[idx].values, y_tr[idx])
            preds[b] = m.predict(X_te.values)
        out[te, 0] = preds.mean(axis=0)
        out[te, 1] = np.percentile(preds, 2.5, axis=0)
        out[te, 2] = np.percentile(preds, 97.5, axis=0)
    return out


# ---------------------------------------------------------------------------
def run_layer(layer: str, features: list[str], n_boot: int = 100,
              skip_gpr: bool = False) -> None:
    p = DATA_DIR / f"calibration_features_layer{layer}.csv"
    if not p.exists():
        logger.warning("Layer %s yok", layer); return
    df = pd.read_csv(p)

    md = [f"# ÇP-2.5 — Görev 10: Belirsizlik Kalibrasyonu (Layer {layer})", "",
          "## Yöntem", "",
          "- **GPR**: Matern(ν=2.5) + WhiteKernel, LOYO her fold için yeniden fit",
          f"- **Bootstrap**: XGBoost {n_boot} resample, LOYO",
          "- **PICP** = mean(y ∈ [PI_2.5, PI_97.5]), hedef ≈ 0.95",
          "- **Sharpness** = mean(PI_upper - PI_lower)",
          ""]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, crop_full in zip(axes, sorted(df["crop"].unique())):
        crop_short = "bugday" if crop_full == "bugday" else "aycicegi"
        sub = df[df["crop"] == crop_full].copy().reset_index(drop=True)
        feats_present = [f for f in features if f in sub.columns]
        X = _impute(sub[feats_present].astype(float))
        y = sub["verim_kg_da"].astype(float).values
        year_g = sub["year"].astype(int).values

        # GPR (optional, expensive)
        if not skip_gpr:
            logger.info("[%s Layer %s] GPR LOYO ...", crop_short, layer)
            gpr_out = loyo_gpr_intervals(X, y, year_g)
            gpr_mean, gpr_std = gpr_out[:, 0], gpr_out[:, 1]
            gpr_lo = gpr_mean - 1.96 * gpr_std
            gpr_up = gpr_mean + 1.96 * gpr_std
            picp_gpr = float(((y >= gpr_lo) & (y <= gpr_up)).mean())
            sharp_gpr = float((gpr_up - gpr_lo).mean())
        else:
            picp_gpr = sharp_gpr = None

        # Bootstrap
        logger.info("[%s Layer %s] XGBoost bootstrap n=%d ...", crop_short, layer, n_boot)
        bt_out = loyo_bootstrap_xgb(X, y, year_g, n_boot=n_boot)
        bt_mean, bt_lo, bt_up = bt_out[:, 0], bt_out[:, 1], bt_out[:, 2]
        picp_bt = float(((y >= bt_lo) & (y <= bt_up)).mean())
        sharp_bt = float((bt_up - bt_lo).mean())

        # Save per-row
        out_csv = REPORT_DIR / f"10_uncertainty_predictions_{layer}_{crop_short}.csv"
        sub_out = sub[["ilce", "year", "crop", "verim_kg_da"]].copy()
        sub_out["bt_mean"] = bt_mean
        sub_out["bt_pi_lower"] = bt_lo
        sub_out["bt_pi_upper"] = bt_up
        if not skip_gpr:
            sub_out["gpr_mean"] = gpr_mean
            sub_out["gpr_std"] = gpr_std
        sub_out.to_csv(out_csv, index=False)

        md.append(f"## {crop_full} (n={len(y)})")
        md.append("")
        md.append("| Yöntem | PICP | Sharpness (kg/da) |")
        md.append("|---|---|---|")
        if not skip_gpr:
            md.append(f"| GPR analitik | **{picp_gpr:.3f}** | {sharp_gpr:.1f} |")
        md.append(f"| XGBoost Bootstrap (n={n_boot}) | **{picp_bt:.3f}** | {sharp_bt:.1f} |")
        md.append("")

        # Reliability: empirical coverage vs nominal
        nominal = np.linspace(0.05, 0.95, 10)
        emp_bt = []
        for nq in nominal:
            half = (1.0 - nq) / 2.0
            lo = np.percentile(np.column_stack([bt_lo, bt_up]),
                                100 * half, axis=1)
            up = np.percentile(np.column_stack([bt_lo, bt_up]),
                                100 * (1 - half), axis=1)
            emp_bt.append(((y >= lo) & (y <= up)).mean())
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="ideal")
        ax.plot(nominal, emp_bt, "o-", color="tab:blue", label="Bootstrap")
        ax.set_title(f"{crop_full}\nBootstrap PICP={picp_bt:.2f}")
        ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"fig_uncertainty_calibration_{layer}.png", dpi=130)
    plt.close()

    (REPORT_DIR / f"10_uncertainty_{layer}.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("rapor → 10_uncertainty_%s.md", layer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="A", choices=["A", "B", "C"])
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--skip-gpr", action="store_true")
    args = p.parse_args()
    feature_map = {"A": FEATURES_A, "B": FEATURES_B, "C": FEATURES_C}
    run_layer(args.layer, feature_map[args.layer],
              n_boot=args.n_boot, skip_gpr=args.skip_gpr)
