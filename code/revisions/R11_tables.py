"""R11 — Regenerate manuscript tables (#15) from the revision artefacts. CSV only; no hand-typing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402

REV = R.REV
TAB = R.RTAB


def main():
    summ = pd.read_csv(REV / "master_ledger_summary.csv")
    clu = pd.read_csv(REV / "clustered_inference.csv")

    # --- Table 2: temporal (LOYO) performance, best ML per crop×tier + matched baseline ---
    loyo = summ[summ.cv == "LOYO"].copy()
    rows = []
    for crop in loyo.crop.unique():
        for tier in ("A", "B", "C"):
            sl = loyo[(loyo.crop == crop) & (loyo.tier == tier)]
            if sl.empty:
                continue
            best = sl.loc[sl.skill_score.idxmax()]
            cl = clu[(clu.crop == crop) & (clu.tier == tier) & (clu.model == best.model) & (clu.cv == "LOYO")]
            yp = float(cl.p_value.iloc[0]) if not cl.empty else np.nan
            rows.append(dict(
                crop=crop, tier=tier, best_model=best.model, n=int(best.n),
                baseline_rmse_matched=best.baseline_rmse_matched, ml_rmse=best.model_rmse,
                baseline_r2=best.baseline_r2_matched, ml_r2=best.model_r2,
                skill_score=best.skill_score,
                ss_clustered_ci=f"[{best.ss_ci_low_clustered:+.3f}, {best.ss_ci_high_clustered:+.3f}]",
                ss_iid_ci=f"[{best.ss_ci_low_iid:+.3f}, {best.ss_ci_high_iid:+.3f}]",
                year_level_p=round(yp, 4) if yp == yp else np.nan))
    pd.DataFrame(rows).to_csv(TAB / "table2_temporal_performance.csv", index=False)

    # --- Table 3: same-model spatial vs temporal gap (from Phase-1 generalization_gap.csv) ---
    gap = pd.read_csv(R.ANALYSIS / "generalization_gap.csv")
    t3 = gap[["crop", "layer", "fixed_model", "r2_loyo_fixed", "r2_loilo_fixed", "dR2_fixed",
              "dR2_fixed_ci_lo", "dR2_fixed_ci_hi", "wilcoxon_p", "rank_biserial"]].rename(
        columns={"layer": "tier", "fixed_model": "model", "r2_loyo_fixed": "r2_LOYO_temporal",
                 "r2_loilo_fixed": "r2_LOILO_spatial", "dR2_fixed": "gap_dR2_spatial_minus_temporal"})
    t3.to_csv(TAB / "table3_generalization_gap.csv", index=False)

    # --- Table 4: per-algorithm matched ablation ---
    ab = pd.read_csv(REV / "ablation_matched_by_algorithm.csv")
    ab[ab.cv == "LOYO"].to_csv(TAB / "table4_ablation.csv", index=False)

    # --- Table 5: parcel per-stage (model + persistence) ---
    pd.read_csv(REV / "parcel_per_stage.csv").to_csv(TAB / "table5_parcel.csv", index=False)

    # --- Table 6: fold-wise importance (top 8 per crop×tier) ---
    imp = pd.read_csv(REV / "permutation_importance_foldwise.csv")
    imp6 = (imp.sort_values(["crop", "tier", "imp_mean"], ascending=[True, True, False])
            .groupby(["crop", "tier"]).head(8))
    imp6.to_csv(TAB / "table6_importance.csv", index=False)

    # --- Rolling-origin table ---
    pd.read_csv(REV / "rolling_origin_summary.csv").to_csv(TAB / "table_rolling_origin.csv", index=False)

    # --- Staged-forecast table (best model per stage) ---
    sf = pd.read_csv(REV / "staged_forecast_results.csv")
    order = {s: i for i, s in enumerate(["pre_season", "vegetative", "flowering", "grain_fill", "end_of_season"])}
    best_sf = (sf.loc[sf.groupby(["crop", "issuance_stage"])["model_r2"].idxmax()]
               .sort_values(["crop", "issuance_stage"], key=lambda s: s.map(order) if s.name == "issuance_stage" else s))
    best_sf.to_csv(TAB / "table_staged_forecast.csv", index=False)

    print("[save] tables: " + ", ".join(p.name for p in sorted(TAB.glob("*.csv"))), flush=True)
    print("\n=== Table 2 (temporal, matched baseline) ===", flush=True)
    print(pd.DataFrame(rows)[["crop", "tier", "best_model", "baseline_rmse_matched", "ml_rmse",
                              "skill_score", "ss_clustered_ci"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
