"""02 — Bootstrap %95 güven aralıkları (R²/RMSE/MAE/MAPE/bias). [vektörize]

Girdi : analysis/per_sample_predictions.csv (01'den)
Çıktı : analysis/bootstrap_ci.csv

Yöntem
------
* Birincil = gözlem (iid case) bootstrap, 5000 resample, seed=12345.
* Duyarlılık = küme (cluster) bootstrap — LOYO'da YIL, LOILO'da İLÇE kümeleri.
  Panel korelasyonu iid varsayımını ihlal eder → cluster CI daha geniş/dürüst.
  Spatiotemporal için blok kimliği saklanmadığından küme-bootstrap atlanır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

N_BOOT = 5000
SEED = 12345
METRICS = ["r2", "rmse_kg_da", "mae_kg_da", "mape_pct", "bias_kg_da"]


def _point(yt, yp):
    return rc.metrics(yt, yp)


def _emit(out, layer, crop, cv, model, n, point, acc, method):
    for m in METRICS:
        lo, hi = rc.ci95(acc[m])
        out.append(dict(layer=layer, crop=crop, cv=cv, model=model, metric=m,
                        point=round(point[m], 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        boot_mean=round(float(np.nanmean(acc[m])), 4),
                        boot_sd=round(float(np.nanstd(acc[m])), 4),
                        method=method, n=n, n_boot=N_BOOT, seed=SEED))


def main() -> None:
    ps = pd.read_csv(rc.ANALYSIS_DIR / "per_sample_predictions.csv")
    out = []
    for (layer, crop, cv, model), g in ps.groupby(["layer", "crop", "cv", "model"]):
        yt = g["y_true"].values.astype(float)
        yp = g["y_pred"].values.astype(float)
        point = _point(yt, yp)
        _emit(out, layer, crop, cv, model, len(g), point,
              rc.boot_iid_metrics(yt, yp, N_BOOT, SEED), "iid")
        if cv in ("LOYO", "LOILO"):
            clg = g["year"].values if cv == "LOYO" else g["ilce_id"].values
            _emit(out, layer, crop, cv, model, len(g), point,
                  rc.boot_cluster_metrics(yt, yp, clg, N_BOOT, SEED), "cluster")

    df = pd.DataFrame(out)
    df.to_csv(rc.ANALYSIS_DIR / "bootstrap_ci.csv", index=False)
    print(f"[save] bootstrap_ci.csv rows={len(df)}", flush=True)

    print("\n=== LOYO R² 95% CI (headline) ===", flush=True)
    h = df[(df.cv == "LOYO") & (df.metric == "r2")]
    for _, r in h.sort_values(["crop", "layer", "model", "method"]).iterrows():
        print(f"  {r['crop']:9s} L{r['layer']} {r['model']:14s} [{r['method']:7s}] "
              f"R2={r['point']:+.3f}  CI[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
