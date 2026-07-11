"""05 — Feature ablasyonu A→B→C (NDVI ve soil marjinal katkısı), DÜRÜST.

Sorun: yayınlanan Layer A (n=589/576) ile Layer B/C (n=213/209) farklı örneklemde
→ A→B kıyası özellik etkisini örneklem etkisiyle KARIŞTIRIR. Bu script ek olarak
climate-only (FEATURES_A) modelleri NDVI-mevcut alt-kümede (layerB satırları)
çalıştırır → eşleştirilmiş (matched-n) A vs B vs C.

Deterministik inference re-run (SEED=42); yeniden eğitim değil.

Çıktı: tables/T2_ablation.csv + analysis/ablation_per_model.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

CVS = ["LOYO", "LOILO"]


def run_tier(data_file: str, features: list[str], crop: str, models: list[str]) -> dict:
    """Bir (veri, özellik seti, ürün) için her model×CV out-of-fold R²/RMSE."""
    df = pd.read_csv(rc.DATA_DIR / data_file)
    sub = rc.crop_subset(df, crop)
    X = rc._impute(sub[features].astype(float))
    y = sub["verim_kg_da"].astype(float).values
    groups = {"LOYO": sub["year"].astype(int).values,
              "LOILO": sub["ilce_id"].astype(int).values}
    out = {}
    for cv in CVS:
        for m in models:
            preds = rc._cv_predict(m, X, y, groups[cv])
            out[(cv, m)] = {
                "r2": float(r2_score(y, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "n": len(y),
            }
    return out


def main() -> None:
    ml = ["pls", "elastic_net", "random_forest", "xgboost", "gpr"]
    rows = []
    per_model = []
    for crop in rc.CROPS:
        tiers = {
            "A_full":    run_tier("calibration_features_layerA.csv", rc.FEATURES_A, crop, ml),
            "A_matched": run_tier("calibration_features_layerB.csv", rc.FEATURES_A, crop, ml),
            "B":         run_tier("calibration_features_layerB.csv", rc.FEATURES["B"], crop, ml),
            "C":         run_tier("calibration_features_layerC.csv", rc.FEATURES["C"], crop, ml),
        }
        for cv in CVS:
            # best model per tier
            best = {}
            for tier, res in tiers.items():
                cand = {m: res[(cv, m)]["r2"] for m in ml}
                bm = max(cand, key=cand.get)
                best[tier] = (bm, cand[bm], res[(cv, bm)]["rmse"], res[(cv, bm)]["n"])
            row = dict(crop=crop, cv=cv)
            for tier in ("A_full", "A_matched", "B", "C"):
                bm, r2, rmse, n = best[tier]
                row[f"{tier}_model"] = bm
                row[f"{tier}_r2"] = round(r2, 4)
                row[f"{tier}_rmse"] = round(rmse, 3)
                row[f"{tier}_n"] = n
            row["dR2_NDVI_matched"] = round(best["B"][1] - best["A_matched"][1], 4)
            row["dR2_soil_matched"] = round(best["C"][1] - best["B"][1], 4)
            row["dR2_NDVI_naive_full"] = round(best["B"][1] - best["A_full"][1], 4)
            rows.append(row)
            for tier in ("A_full", "A_matched", "B", "C"):
                for m in ml:
                    r = tiers[tier][(cv, m)]
                    per_model.append(dict(crop=crop, cv=cv, tier=tier, model=m,
                                          r2=round(r["r2"], 4), rmse=round(r["rmse"], 3),
                                          n=r["n"]))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(rc.TABLES_DIR / "T2_ablation.csv", index=False)
    pd.DataFrame(per_model).to_csv(rc.ANALYSIS_DIR / "ablation_per_model.csv", index=False)
    print(f"[save] T2_ablation.csv rows={len(tbl)}", flush=True)
    print("\n=== Ablation (best model per tier) ===", flush=True)
    for _, r in tbl.iterrows():
        print(f"  {r['crop']:9s} {r['cv']:6s} | A_full(n={r['A_full_n']})={r['A_full_r2']:+.3f} "
              f"A_match(n={r['A_matched_n']})={r['A_matched_r2']:+.3f} "
              f"B={r['B_r2']:+.3f} C={r['C_r2']:+.3f} | "
              f"ΔNDVI(matched)={r['dR2_NDVI_matched']:+.3f} Δsoil={r['dR2_soil_matched']:+.3f}",
              flush=True)


if __name__ == "__main__":
    main()
