"""T6 — Assemble tiers A–D per crop (matched NDVI-era sample n=213/209).

Same observations across tiers (district-years with yield + indices); tiers differ only by feature
GROUP (clean ladder):
  A = climate (14, existing, READ-ONLY reuse)
  B = A + index means (8 indices × 3 windows)
  C = B + soil (0–30 cm means + AWC)
  D = C + index distribution metrics (median/std/P10/P90/CV/range) + topography
D is a wide CANDIDATE pool; T7 prunes per crop (collinearity + importance + count cap).
Output: outputs/tier_{A,B,C,D}_<crop>.csv + outputs/feature_groups.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

LAYERC = E.PROJECT_ROOT / "data" / "processed" / "calibration_features_layerC.csv"
CROP_FULL = {"bugday": "bugday", "aycicegi": "aycicegi_yaglik"}
CLIMATE = ["gdd_cum_season", "gdd_flowering", "vernalization_days", "tp_season_sum",
           "tp_winter_sum", "tp_flowering", "tp_grain_fill", "aridity_index", "heat_stress_days",
           "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean", "ssr_flowering_sum",
           "ssr_season_sum"]
KEYS = ["ilce_id", "ilce", "il", "year"]
DIST_SUFFIX = ("_median", "_stdDev", "_p10", "_p90", "_cv", "_range")
TOPO = ["elevation_m", "slope_deg", "northness", "eastness", "twi"]


def main():
    panel = pd.read_csv(LAYERC)
    soil = pd.read_csv(E.OUT / "soil_features.csv") if (E.OUT / "soil_features.csv").exists() else None
    topo = pd.read_csv(E.OUT / "topo_features.csv") if (E.OUT / "topo_features.csv").exists() else None
    if soil is None or topo is None:
        print("STOP: soil/topo outputs missing — run T3/T4 first."); return 2

    groups = {}
    for crop in ("bugday", "aycicegi"):
        idx = pd.read_csv(E.OUT / f"indices_{crop}.csv")
        ph = panel[panel["crop"] == CROP_FULL[crop]][KEYS + CLIMATE + ["verim_kg_da"]].copy()
        # base = climate panel ∩ indices (matched NDVI-era sample)
        base = ph.merge(idx, on=KEYS, how="inner")
        base = base.merge(soil.drop(columns=["ilce", "il"]), on="ilce_id", how="left")
        base = base.merge(topo.drop(columns=["ilce", "il"]), on="ilce_id", how="left")

        idx_mean = [c for c in idx.columns if c.endswith("_mean")]
        idx_dist = [c for c in idx.columns if c.endswith(DIST_SUFFIX)]
        soil_feat = [c for c in soil.columns if c.endswith("_0_30_mean") or c == "awc_0_30"]

        tiers = {
            "A": CLIMATE,
            "B": CLIMATE + idx_mean,
            "C": CLIMATE + idx_mean + soil_feat,
            "D": CLIMATE + idx_mean + soil_feat + idx_dist + TOPO,
        }
        for t, feats in tiers.items():
            feats = [f for f in feats if f in base.columns]
            cols = KEYS + ["verim_kg_da"] + feats
            base[cols].to_csv(E.OUT / f"tier_{t}_{crop}.csv", index=False)
        n = len(base)
        print(f"[t6] {crop}: n={n} | A={len(tiers['A'])} B={len(tiers['B'])} "
              f"C={len(tiers['C'])} D={len(tiers['D'])} features", flush=True)
        groups[crop] = {"n": int(n), "climate": CLIMATE, "index_mean": idx_mean,
                        "soil": soil_feat, "index_dist": idx_dist, "topo": TOPO,
                        "tier_sizes": {t: len([f for f in v if f in base.columns]) for t, v in tiers.items()}}
    (E.OUT / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print("[t6] saved tier matrices + feature_groups.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
