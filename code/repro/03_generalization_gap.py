"""03 — Mekânsal vs zamansal genelleme açığı (LOYO vs LOILO).

Girdi : analysis/per_sample_predictions.csv
Çıktı : analysis/generalization_gap.csv + analysis/gap_readme.md

İki tamamlayıcı bakış:
1. Betimsel (best-per-regime): her layer×crop için en iyi model R²'si, her CV
   şemasında. ΔR² = R²(LOILO) − R²(LOYO). Headline genelleme açığı.
2. Eşli (paired) test: SABİT bir model (o layer×crop'un LOILO şampiyonu) için
   AYNI gözlemlerde LOYO vs LOILO mutlak hataları → Wilcoxon signed-rank (eşli),
   effect size (rank-biserial), eşli-bootstrap ΔR² %95 CI. Tüm modeller için
   ΔR² de raporlanır (açığın modelden bağımsız tutarlılığı).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

N_BOOT = 5000
SEED = 12345


def _r2(yt, yp):
    return r2_score(yt, yp)


def _rmse(yt, yp):
    return float(np.sqrt(mean_squared_error(yt, yp)))


def main() -> None:
    ps = pd.read_csv(rc.ANALYSIS_DIR / "per_sample_predictions.csv")
    rows = []

    for layer in ("A", "B", "C"):
        for crop in rc.CROPS:
            sub = ps[(ps["layer"] == layer) & (ps["crop"] == crop)]
            if sub.empty:
                continue
            # best model per CV by R²
            best = {}
            for cv in ("LOYO", "LOILO", "Spatiotemporal"):
                cc = sub[sub["cv"] == cv]
                r2_by_model = {mdl: _r2(g["y_true"], g["y_pred"])
                               for mdl, g in cc.groupby("model")}
                bm = max(r2_by_model, key=r2_by_model.get)
                best[cv] = (bm, float(r2_by_model[bm]))

            # fixed model = LOILO champion (best spatial generalizer)
            fixed_model = best["LOILO"][0]

            # paired LOYO vs LOILO for the fixed model (same observations)
            loyo = sub[(sub["cv"] == "LOYO") & (sub["model"] == fixed_model)].sort_values(
                ["ilce_id", "year"]).reset_index(drop=True)
            loilo = sub[(sub["cv"] == "LOILO") & (sub["model"] == fixed_model)].sort_values(
                ["ilce_id", "year"]).reset_index(drop=True)
            assert (loyo["ilce_id"].values == loilo["ilce_id"].values).all()
            assert (loyo["year"].values == loilo["year"].values).all()

            ae_loyo = loyo["abs_error"].values
            ae_loilo = loilo["abs_error"].values
            d = ae_loyo - ae_loilo  # >0 => LOYO worse
            nz = d != 0
            if nz.sum() > 0:
                W, p = wilcoxon(ae_loyo[nz], ae_loilo[nz], alternative="two-sided")
                ranks = rankdata(np.abs(d[nz]))
                Wp = ranks[d[nz] > 0].sum()
                Wm = ranks[d[nz] < 0].sum()
                rbc = float((Wp - Wm) / (Wp + Wm))
            else:
                W, p, rbc = np.nan, np.nan, np.nan

            # paired bootstrap ΔR² (LOILO − LOYO) for fixed model [vektörize]
            yt = loyo["y_true"].values.astype(float)
            yp_loyo = loyo["y_pred"].values.astype(float)
            yp_loilo = loilo["y_pred"].values.astype(float)
            n = len(yt)
            dvals = rc.boot_paired_dr2(yt, yp_loyo, yp_loilo, N_BOOT, SEED)
            dr2_lo, dr2_hi = rc.ci95(dvals)

            r2_loyo_fixed = _r2(yt, yp_loyo)
            r2_loilo_fixed = _r2(yt, yp_loilo)

            rows.append(dict(
                layer=layer, crop=crop,
                best_loyo_model=best["LOYO"][0], r2_loyo_best=round(best["LOYO"][1], 4),
                best_loilo_model=best["LOILO"][0], r2_loilo_best=round(best["LOILO"][1], 4),
                best_spatio_model=best["Spatiotemporal"][0], r2_spatio_best=round(best["Spatiotemporal"][1], 4),
                dR2_loilo_minus_loyo_best=round(best["LOILO"][1] - best["LOYO"][1], 4),
                fixed_model=fixed_model,
                r2_loyo_fixed=round(r2_loyo_fixed, 4), r2_loilo_fixed=round(r2_loilo_fixed, 4),
                dR2_fixed=round(r2_loilo_fixed - r2_loyo_fixed, 4),
                dR2_fixed_ci_lo=round(float(dr2_lo), 4), dR2_fixed_ci_hi=round(float(dr2_hi), 4),
                n_paired=n, n_nonzero=int(nz.sum()),
                median_abserr_loyo=round(float(np.median(ae_loyo)), 3),
                median_abserr_loilo=round(float(np.median(ae_loilo)), 3),
                wilcoxon_W=float(W) if not np.isnan(W) else np.nan,
                wilcoxon_p=float(p) if not np.isnan(p) else np.nan,
                rank_biserial=round(rbc, 4) if not np.isnan(rbc) else np.nan,
            ))

    gap = pd.DataFrame(rows)
    gap.to_csv(rc.ANALYSIS_DIR / "generalization_gap.csv", index=False)
    print(f"[save] generalization_gap.csv rows={len(gap)}", flush=True)

    # per-model consistency table (ΔR² LOILO-LOYO for every model)
    cons = []
    for (layer, crop, model), g in ps.groupby(["layer", "crop", "model"]):
        if set(g["cv"].unique()) < {"LOYO", "LOILO"}:
            continue
        loyo = g[g["cv"] == "LOYO"]
        loilo = g[g["cv"] == "LOILO"]
        cons.append(dict(layer=layer, crop=crop, model=model,
                         r2_loyo=round(_r2(loyo["y_true"], loyo["y_pred"]), 4),
                         r2_loilo=round(_r2(loilo["y_true"], loilo["y_pred"]), 4),
                         dR2=round(_r2(loilo["y_true"], loilo["y_pred"]) -
                                   _r2(loyo["y_true"], loyo["y_pred"]), 4)))
    pd.DataFrame(cons).to_csv(rc.ANALYSIS_DIR / "generalization_gap_per_model.csv", index=False)

    print("\n=== Generalization gap (fixed = LOILO champion) ===", flush=True)
    for _, r in gap.iterrows():
        print(f"  {r['crop']:9s} L{r['layer']} [{r['fixed_model']:13s}] "
              f"R²_LOYO={r['r2_loyo_fixed']:+.3f}  R²_LOILO={r['r2_loilo_fixed']:+.3f}  "
              f"ΔR²={r['dR2_fixed']:+.3f} CI[{r['dR2_fixed_ci_lo']:+.3f},{r['dR2_fixed_ci_hi']:+.3f}]  "
              f"Wilcoxon p={r['wilcoxon_p']:.2e}  rbc={r['rank_biserial']:+.3f}", flush=True)

    readme = f"""# Generalization Gap — methods & interpretation

## What is compared
- **LOYO** (Leave-One-Year-Out): predict a held-out *year* → **temporal** generalization.
- **LOILO** (Leave-One-İlçe-Out): predict a held-out *district* → **spatial** generalization.
- **Spatiotemporal**: 5 year-blocks × 5 spatial KMeans clusters (25 blocks).

Both LOYO and LOILO operate on the *same* observation set per layer×crop, so per-observation
errors are **paired** (same ilce_id+year), enabling a paired test.

## Two views
1. **Best-per-regime** (`r2_*_best`): the highest R² achievable in each CV scheme (across the
   5–6 models). `dR2_loilo_minus_loyo_best` is the headline gap.
2. **Fixed-model paired** (`*_fixed`): one model (the LOILO champion) evaluated under both
   schemes on identical observations. Paired Wilcoxon signed-rank on |error|; rank-biserial
   effect size; paired bootstrap ({N_BOOT} resamples, seed {SEED}) for ΔR² 95% CI.

`generalization_gap_per_model.csv` shows ΔR² for *every* model → the gap is not model-specific.

## Honest reading
A large positive ΔR² (LOILO ≫ LOYO) means the model interpolates across space far better than it
extrapolates across time: it learns district-level structure but fails to anticipate the
year-to-year climate variability that actually drives yield anomalies. This is the central,
nuanced finding of the paper — spatial skill does NOT imply operational (forward-in-time) skill.
"""
    (rc.ANALYSIS_DIR / "gap_readme.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
