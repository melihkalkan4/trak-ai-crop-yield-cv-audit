"""R04 — Per-algorithm matched ablation (#10).

NDVI / soil marginal contribution computed with the SAME algorithm, SAME matched sample
(NDVI-available rows), SAME folds, SAME fixed hyperparameters/preprocessing. Reuses the
fidelity-verified ablation_per_model.csv (tiers A_full, A_matched, B, C × model × cv R²).

ΔR²(NDVI)  = R²(B, algo)        − R²(A_matched, algo)
ΔR²(soil)  = R²(C, algo)        − R²(B, algo)

Output: revisions/ablation_matched_by_algorithm.csv (+ summary across algorithms).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402

ALGOS = ["pls", "elastic_net", "random_forest", "xgboost", "gpr"]


def main():
    ab = pd.read_csv(R.ANALYSIS / "ablation_per_model.csv")  # crop,cv,tier,model,r2,rmse,n
    rows = []
    for crop in ab["crop"].unique():
        for cv in ab["cv"].unique():
            for algo in ALGOS:
                def r2(tier):
                    s = ab[(ab.crop == crop) & (ab.cv == cv) & (ab.tier == tier) & (ab.model == algo)]
                    return float(s["r2"].iloc[0]) if not s.empty else None
                def n(tier):
                    s = ab[(ab.crop == crop) & (ab.cv == cv) & (ab.tier == tier) & (ab.model == algo)]
                    return int(s["n"].iloc[0]) if not s.empty else None
                r_am, r_b, r_c = r2("A_matched"), r2("B"), r2("C")
                if r_am is None or r_b is None:
                    continue
                rows.append(dict(
                    crop=crop, algorithm=algo, cv=cv, n=n("B"),
                    r2_climate_matched=round(r_am, 4), r2_climate_ndvi=round(r_b, 4),
                    r2_climate_ndvi_soil=round(r_c, 4) if r_c is not None else None,
                    delta_r2_ndvi=round(r_b - r_am, 4),
                    delta_r2_soil=round(r_c - r_b, 4) if r_c is not None else None,
                ))
    df = pd.DataFrame(rows)
    df.to_csv(R.REV / "ablation_matched_by_algorithm.csv", index=False)
    print(f"[save] ablation_matched_by_algorithm.csv rows={len(df)}", flush=True)

    summ = (df.groupby(["crop", "cv"])
            .agg(mean_delta_ndvi=("delta_r2_ndvi", "mean"),
                 median_delta_ndvi=("delta_r2_ndvi", "median"),
                 mean_delta_soil=("delta_r2_soil", "mean"),
                 median_delta_soil=("delta_r2_soil", "median"),
                 n_algos=("algorithm", "nunique")).round(4).reset_index())
    summ.to_csv(R.REV / "ablation_by_algorithm_summary.csv", index=False)
    print("\n=== per-algorithm NDVI ΔR² (mean across algorithms) ===", flush=True)
    print(summ.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
