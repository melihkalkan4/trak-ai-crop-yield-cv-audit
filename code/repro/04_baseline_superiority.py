"""04 — Baseline (B0 climatology) üstünlüğü: ML modelleri B0'ı geçiyor mu?

Dürüst çekirdek bulgu: buğdayda LOYO'da ML, B0 klimatolojiyi GEÇEMEZ;
ayçiçeğinde multimodal (Layer C) GEÇER.

Yöntem: her layer×crop×model (LOYO) için, B0 klimatoloji per-sample tahminleri
AYNI gözlem alt-kümesinde hesaplanır (leave-one-out ilçe yıl ortalaması). Eşli:
* Skill Score = 1 − RMSE_model / RMSE_B0  (yayınlanan ss_vs_b0 ile aynı tanım)
* SS için eşli-bootstrap %95 CI (gözlem resample, her ikisinin RMSE'si yeniden)
* Wilcoxon signed-rank |err_model| vs |err_B0| (eşli, two-sided)
* beats_B0 = (SS CI alt sınırı > 0)  → istatistiksel olarak B0'dan iyi mi?

Girdi : analysis/per_sample_predictions.csv + repro_common.b0_per_sample
Çıktı : analysis/baseline_superiority.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

N_BOOT = 5000
SEED = 12345


def _rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def main() -> None:
    ps = pd.read_csv(rc.ANALYSIS_DIR / "per_sample_predictions.csv")
    rows = []

    # cache B0 per-sample per layer×crop
    b0cache = {}
    for layer in ("A", "B", "C"):
        for crop in rc.CROPS:
            b0cache[(layer, crop)] = rc.b0_per_sample(layer, crop)

    for layer in ("A", "B", "C"):
        for crop in rc.CROPS:
            b0 = b0cache[(layer, crop)].sort_values(["ilce_id", "year"]).reset_index(drop=True)
            rmse_b0 = _rmse(b0["y_true"].values, b0["b0_pred"].values)
            for model in rc.MODELS_BY_LAYER[layer]:
                mdf = ps[(ps["layer"] == layer) & (ps["crop"] == crop) &
                         (ps["cv"] == "LOYO") & (ps["model"] == model)]
                if mdf.empty:
                    continue
                mdf = mdf.sort_values(["ilce_id", "year"]).reset_index(drop=True)
                assert (mdf["ilce_id"].values == b0["ilce_id"].values).all()
                assert (mdf["year"].values == b0["year"].values).all()

                yt = mdf["y_true"].values.astype(float)
                yp = mdf["y_pred"].values.astype(float)
                yb = b0["b0_pred"].values.astype(float)
                rmse_m = _rmse(yt, yp)
                ss = 1.0 - rmse_m / rmse_b0

                # paired bootstrap of SS [vektörize]
                n = len(yt)
                ssv = rc.boot_skill_score(yt, yp, yb, N_BOOT, SEED)
                ss_lo, ss_hi = rc.ci95(ssv)

                # paired Wilcoxon on |err|
                ae_m = np.abs(yt - yp)
                ae_b = np.abs(yt - yb)
                d = ae_m - ae_b
                nz = d != 0
                if nz.sum() > 0:
                    W, p = wilcoxon(ae_m[nz], ae_b[nz], alternative="two-sided")
                else:
                    W, p = np.nan, np.nan

                rows.append(dict(
                    layer=layer, crop=crop, model=model, cv="LOYO", n=n,
                    rmse_model=round(rmse_m, 3), rmse_b0=round(rmse_b0, 3),
                    skill_score=round(ss, 4),
                    ss_ci_lo=round(float(ss_lo), 4), ss_ci_hi=round(float(ss_hi), 4),
                    wilcoxon_W=float(W) if not np.isnan(W) else np.nan,
                    wilcoxon_p=float(p) if not np.isnan(p) else np.nan,
                    beats_b0_ci=bool(ss_lo > 0),
                    worse_than_b0_ci=bool(ss_hi < 0),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(rc.ANALYSIS_DIR / "baseline_superiority.csv", index=False)
    print(f"[save] baseline_superiority.csv rows={len(df)}", flush=True)

    print("\n=== Does any ML model beat B0 climatology under LOYO? ===", flush=True)
    for crop in rc.CROPS:
        print(f"\n-- {crop} --", flush=True)
        cc = df[df["crop"] == crop].sort_values("skill_score", ascending=False)
        for _, r in cc.iterrows():
            flag = "BEATS B0" if r["beats_b0_ci"] else ("WORSE" if r["worse_than_b0_ci"] else "ns")
            print(f"  L{r['layer']} {r['model']:14s} SS={r['skill_score']:+.3f} "
                  f"CI[{r['ss_ci_lo']:+.3f},{r['ss_ci_hi']:+.3f}] p={r['wilcoxon_p']:.2e} [{flag}]",
                  flush=True)


if __name__ == "__main__":
    main()
