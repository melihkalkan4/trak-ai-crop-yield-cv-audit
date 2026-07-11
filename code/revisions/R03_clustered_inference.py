"""R03 — Cluster-aware inference (#5, #6): model vs matched climatology baseline.

Primary = clustered tests (respect year / ilce / block dependence). iid Wilcoxon → supplementary.
Per crop×tier×model×cv: cluster-level mean |error| difference (model − baseline) with
block-bootstrap CI + Wilcoxon on cluster-mean errors. Negative diff ⇒ model better.

#6: the sunflower skill-score year-clustered CI (from master_ledger_summary) is surfaced
explicitly; if it includes zero we report that, we do not hide it.

Outputs: revisions/clustered_inference.csv, revisions/iid_inference_supplementary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402
rc = R.rc


def main():
    led = pd.read_csv(R.REV / "master_ledger.csv")
    led["cluster"] = led["fold_id"]  # fold_id already encodes year/ilce/block per cv

    clustered, iid = [], []
    for (crop, tier, model, cv), g in led.groupby(["crop", "tier", "model", "cv"]):
        # clustered MAE-difference test (model vs baseline)
        res = R.cluster_mae_diff_test(g, "model_pred", "baseline_pred", "cluster")
        clustered.append(dict(comparison=f"{crop}|tier{tier}|{model}|{cv}|model_vs_climatology",
                              crop=crop, tier=tier, model=model, cv=cv, **res,
                              method="cluster block-bootstrap + signed-rank on cluster means"))
        # iid (observation-level) Wilcoxon — supplementary
        ae_m = (g["y_true"] - g["model_pred"]).abs().values
        ae_b = (g["y_true"] - g["baseline_pred"]).abs().values
        nz = (ae_m - ae_b) != 0
        try:
            W, p = wilcoxon(ae_m[nz], ae_b[nz], alternative="two-sided")
            W, p = float(W), float(p)
        except Exception:
            W, p = np.nan, np.nan
        iid.append(dict(comparison=f"{crop}|tier{tier}|{model}|{cv}|model_vs_climatology",
                        crop=crop, tier=tier, model=model, cv=cv, n_obs=len(g),
                        median_abs_err_model=round(float(np.median(ae_m)), 3),
                        median_abs_err_baseline=round(float(np.median(ae_b)), 3),
                        wilcoxon_W=W, p_value=p, method="iid observation-level (NOT primary)"))

    cdf = pd.DataFrame(clustered)
    cdf.to_csv(R.REV / "clustered_inference.csv", index=False)
    pd.DataFrame(iid).to_csv(R.REV / "iid_inference_supplementary.csv", index=False)
    print(f"[save] clustered_inference.csv rows={len(cdf)} ; iid_inference_supplementary.csv", flush=True)

    # #6 — surface sunflower skill-score year-clustered CI
    summ = pd.read_csv(R.REV / "master_ledger_summary.csv")
    sun = summ[(summ.crop == "aycicegi") & (summ.cv == "LOYO")].sort_values("skill_score", ascending=False)
    print("\n=== #6 Sunflower LOYO skill score — year-clustered CI (primary) ===", flush=True)
    incl_zero = []
    for _, r in sun.iterrows():
        z = r.ss_ci_low_clustered <= 0 <= r.ss_ci_high_clustered
        if z:
            incl_zero.append(f"tier{r.tier}/{r.model}")
        print(f"  tier{r.tier} {r.model:14s} SS={r.skill_score:+.3f} "
              f"clustered CI[{r.ss_ci_low_clustered:+.3f},{r.ss_ci_high_clustered:+.3f}] "
              f"iid CI[{r.ss_ci_low_iid:+.3f},{r.ss_ci_high_iid:+.3f}]"
              f"{'  <-- CI INCLUDES ZERO' if z else ''}", flush=True)
    note = ("Sunflower skill-score year-clustered confidence intervals that INCLUDE ZERO "
            f"(substantial across-year uncertainty): {', '.join(incl_zero) if incl_zero else 'none'}. "
            "This is reported, not hidden (#6).")
    (R.REV / "sunflower_clustered_ci_note.txt").write_text(note, encoding="utf-8")
    print("\n" + note, flush=True)


if __name__ == "__main__":
    main()
