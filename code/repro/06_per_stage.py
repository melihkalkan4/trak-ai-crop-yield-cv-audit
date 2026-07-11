"""06 — Per-stage NDVI tahmin skili (FLOV) + senescence çöküşü.

Mevcut FLOV artefaktlarını konsolide eder (yeniden hesap YOK; metrikler FLOV
pipeline'ı tarafından üretilmiştir — sağlayıcı: src/prospective_validation).

Girdi (read-only):
* reports/prospective/EVR_01_{2025,2026}_validation_per_stage.csv
* reports/prospective/EVR_01_{2025,2026}_validation_summary.json
Çıktı:
* tables/T3_per_stage.csv          — evre × yıl R²/MAE/RMSE/MAPE
* analysis/prospective_overall.csv — genel model vs naive persistence + Wilcoxon

DÜRÜST BULGU: NDVI t+7 tahmini flowering'e kadar iyi (R²≈0.74–0.79), ama
grain_fill/maturity'de (senescence) ÇÖKÜYOR (R²<0); ve genelde naive
persistence'ı GEÇEMİYOR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

PROSP = rc.PROJECT_ROOT / "reports" / "prospective"
SITE = "EVR_01"
YEARS = [2025, 2026]


def main() -> None:
    # ---- per-stage ----
    frames = []
    for yr in YEARS:
        f = PROSP / f"{SITE}_{yr}_validation_per_stage.csv"
        if not f.exists():
            print(f"  missing {f}", flush=True)
            continue
        d = pd.read_csv(f)
        d.insert(0, "site", SITE)
        d.insert(1, "year", yr)
        frames.append(d)
    ps = pd.concat(frames, ignore_index=True)
    ps = ps.rename(columns={"R2": "r2", "MAE": "mae", "RMSE": "rmse",
                            "MAPE_pct": "mape_pct"})
    ps.to_csv(rc.TABLES_DIR / "T3_per_stage.csv", index=False)
    print(f"[save] T3_per_stage.csv rows={len(ps)}", flush=True)

    # ---- overall model vs persistence ----
    ov = []
    for yr in YEARS:
        f = PROSP / f"{SITE}_{yr}_validation_summary.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        m = j["overall_model"]
        p = j["overall_naive_persistence"]
        w = j["wilcoxon_model_vs_naive"]
        ov.append(dict(
            site=SITE, year=yr, source=j.get("source"),
            n_matched=j.get("n_matched"), coverage_pct=round(j.get("coverage_pct", float("nan")), 1),
            model_R2=round(m["R2"], 4), model_MAE=round(m["MAE"], 4), model_RMSE=round(m["RMSE"], 4),
            persistence_R2=round(p["R2"], 4), persistence_MAE=round(p["MAE"], 4),
            persistence_RMSE=round(p["RMSE"], 4),
            model_beats_persistence=bool(m["R2"] > p["R2"] and m["MAE"] < p["MAE"]),
            wilcoxon_p=w["wilcoxon_p_value"],
            median_abserr_model=round(w["median_abs_err_model"], 4),
            median_abserr_persistence=round(w["median_abs_err_naive"], 4),
        ))
    ovdf = pd.DataFrame(ov)
    ovdf.to_csv(rc.ANALYSIS_DIR / "prospective_overall.csv", index=False)

    print("\n=== Per-stage R² (senescence collapse) ===", flush=True)
    for _, r in ps.iterrows():
        flag = "  <-- COLLAPSE" if r["r2"] < 0 else ""
        print(f"  {r['year']} {r['stage']:12s} n={int(r['n']):3d} R²={r['r2']:+.3f}{flag}", flush=True)
    print("\n=== Overall model vs naive persistence ===", flush=True)
    for _, r in ovdf.iterrows():
        print(f"  {r['year']}: model R²={r['model_R2']:+.3f} vs persistence R²={r['persistence_R2']:+.3f} "
              f"| beats={r['model_beats_persistence']} | Wilcoxon p={r['wilcoxon_p']:.3f}", flush=True)


if __name__ == "__main__":
    main()
