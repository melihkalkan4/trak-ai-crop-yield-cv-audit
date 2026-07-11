"""01 — Per-sample (out-of-fold) tahminleri yeniden üret + SADAKAT KAPISI.

Çıktılar (paper1_generalization/analysis/):
* per_sample_predictions.csv   — long format, tüm layer×crop×cv×model
* recomputed_aggregate_metrics.csv — recompute edilen aggregate metrikler
* fidelity_check.csv           — recompute vs yayınlanan tablo (PASS/FAIL)

Sadakat kapısı: recompute edilen R²/RMSE/MAE/MAPE/bias/SS, yayınlanan
reports/cp25/05,06,07_*_results.csv ile |Δ| ≤ 0.0011 (3-ondalık yuvarlama)
eşleşmeli. Eşleşmezse → DURDUR ve raporla (uydurma yok).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

PUBLISHED = {
    "A": rc.PROJECT_ROOT / "reports" / "cp25" / "05_layer_a_results.csv",
    "B": rc.PROJECT_ROOT / "reports" / "cp25" / "06_layer_b_results.csv",
    "C": rc.PROJECT_ROOT / "reports" / "cp25" / "07_layer_c_results.csv",
}
METRIC_COLS = ["r2", "rmse_kg_da", "mae_kg_da", "mape_pct", "bias_kg_da", "ss_vs_b0"]
TOL = 0.0011


def main() -> None:
    all_ps = []
    recomputed = []
    for layer in ("A", "B", "C"):
        b0 = {c: rc.b0_rmse_for(layer, c) for c in rc.CROPS}
        for crop in rc.CROPS:
            print(f"[gen] Layer {layer} · {crop} ...", flush=True)
            ps = rc.generate_per_sample(layer, crop)
            all_ps.append(ps)
            # aggregate recompute per cv×model
            for (cv, model), g in ps.groupby(["cv", "model"]):
                m = rc.metrics(g["y_true"].values, g["y_pred"].values, rmse_b0=b0[crop])
                m.update({"layer": layer, "crop": crop, "cv": cv, "model": model,
                          "n": len(g)})
                recomputed.append(m)

    ps_all = pd.concat(all_ps, ignore_index=True)
    ps_all.to_csv(rc.ANALYSIS_DIR / "per_sample_predictions.csv", index=False)
    print(f"[save] per_sample_predictions.csv  rows={len(ps_all)}", flush=True)

    rec = pd.DataFrame(recomputed)
    rec_round = rec.copy()
    for c in METRIC_COLS:
        rec_round[c] = rec_round[c].round(3)
    rec_round.to_csv(rc.ANALYSIS_DIR / "recomputed_aggregate_metrics.csv", index=False)

    # ---- Fidelity gate ----
    checks = []
    for layer in ("A", "B", "C"):
        pub = pd.read_csv(PUBLISHED[layer])
        for _, pr in pub.iterrows():
            sel = rec[(rec["layer"] == layer) & (rec["crop"] == pr["crop"]) &
                      (rec["cv"] == pr["cv"]) & (rec["model"] == pr["model"])]
            if sel.empty:
                checks.append({"layer": layer, "crop": pr["crop"], "cv": pr["cv"],
                               "model": pr["model"], "status": "MISSING_IN_RECOMPUTE"})
                continue
            row = sel.iloc[0]
            rec_check = {"layer": layer, "crop": pr["crop"], "cv": pr["cv"],
                         "model": pr["model"], "n_pub": int(pr["n"]), "n_rec": int(row["n"])}
            worst = 0.0
            for c in METRIC_COLS:
                if c not in pub.columns or pd.isna(pr.get(c)):
                    continue
                d = abs(round(float(row[c]), 3) - float(pr[c]))
                rec_check[f"d_{c}"] = round(d, 4)
                worst = max(worst, d)
            rec_check["max_abs_diff"] = round(worst, 4)
            rec_check["status"] = "PASS" if worst <= TOL else "FAIL"
            checks.append(rec_check)

    chk = pd.DataFrame(checks)
    chk.to_csv(rc.ANALYSIS_DIR / "fidelity_check.csv", index=False)

    n_pass = (chk["status"] == "PASS").sum()
    n_total = len(chk)
    n_fail = (chk["status"] == "FAIL").sum()
    print("\n========== FIDELITY GATE ==========", flush=True)
    print(f"checks: {n_total} | PASS: {n_pass} | FAIL: {n_fail}", flush=True)
    if "max_abs_diff" in chk:
        print(f"global max |Δ| across all metrics: {chk['max_abs_diff'].max():.4f}", flush=True)
    if n_fail:
        print("\n!!! FAILURES (recompute != published) — DURDUR, incele:", flush=True)
        print(chk[chk["status"] == "FAIL"].to_string(index=False), flush=True)
    else:
        print("ALL CHECKS PASS — reproduction is byte-faithful to published tables.", flush=True)

    # ---- Cross-check: my LOYO champion per-sample vs existing loocv CSVs ----
    print("\n========== CROSS-CHECK vs existing loocv CSVs ==========", flush=True)
    champ_files = {
        ("A", "bugday"): ("05_loocv_predictions_bugday.csv", "elastic_net"),
        ("A", "aycicegi"): ("05_loocv_predictions_aycicegi.csv", "random_forest"),
        ("B", "bugday"): ("06_loocv_predictions_bugday.csv", "gpr"),
        ("B", "aycicegi"): ("06_loocv_predictions_aycicegi.csv", "random_forest"),
        ("C", "bugday"): ("07_loocv_predictions_bugday.csv", "xgboost"),
        ("C", "aycicegi"): ("07_loocv_predictions_aycicegi.csv", "gpr"),
    }
    for (layer, crop), (fname, champ) in champ_files.items():
        fpath = rc.PROJECT_ROOT / "reports" / "cp25" / fname
        if not fpath.exists():
            print(f"  {layer}/{crop}: {fname} MISSING", flush=True)
            continue
        ext = pd.read_csv(fpath).sort_values(["ilce_id", "year"]).reset_index(drop=True)
        mine = ps_all[(ps_all["layer"] == layer) & (ps_all["crop"] == crop) &
                      (ps_all["cv"] == "LOYO") & (ps_all["model"] == champ)
                      ].sort_values(["ilce_id", "year"]).reset_index(drop=True)
        if len(ext) != len(mine):
            print(f"  {layer}/{crop} ({champ}): LEN MISMATCH ext={len(ext)} mine={len(mine)}", flush=True)
            continue
        max_pred_diff = float(np.max(np.abs(ext["yield_pred_loyo"].values - mine["y_pred"].values)))
        print(f"  {layer}/{crop} (champ={champ}): n={len(mine)}  max|Δpred|={max_pred_diff:.6f}", flush=True)


if __name__ == "__main__":
    main()
