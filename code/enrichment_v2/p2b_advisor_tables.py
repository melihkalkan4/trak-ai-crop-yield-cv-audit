"""P2b — Advisor documentation tables (#2 RS inventory, #3 masking/windows/tiers).

Pure documentation from verified config/source (no GEE). Produces the tables the advisor asked to
include in the manuscript so masking/variables/tiers are explicit and reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

# 1) RS variable inventory: thesis-original + newly added, with formula + rationale
RS_INVENTORY = [
    # thesis original (cp25 FEATURES_NDVI) — single-value district NDVI descriptors
    dict(group="thesis_original", variable="ndvi_max", formula="max NDVI over season", rationale="peak greenness (thesis)"),
    dict(group="thesis_original", variable="ndvi_mean_season", formula="mean NDVI over season", rationale="seasonal vigour (thesis)"),
    dict(group="thesis_original", variable="ndvi_integral", formula="∑ NDVI over season", rationale="cumulative biomass (thesis)"),
    dict(group="thesis_original", variable="ndvi_flowering", formula="NDVI in flowering window", rationale="reproductive-stage canopy (thesis)"),
    dict(group="thesis_original", variable="ndvi_grain_fill", formula="NDVI in grain-fill window", rationale="grain-fill canopy (thesis)"),
    dict(group="thesis_original", variable="ndvi_spring_slope", formula="green-up slope", rationale="emergence vigour (thesis)"),
    dict(group="thesis_original", variable="greenness_days", formula="days NDVI>thr", rationale="growing-season length (thesis)"),
    # new indices (advisor #2)
    dict(group="new_index", variable="NDVI", formula="(B8-B4)/(B8+B4)", rationale="baseline greenness"),
    dict(group="new_index", variable="EVI", formula="2.5*(B8-B4)/(B8+6*B4-7.5*B2+1)", rationale="reduces NDVI saturation at high biomass (wheat dense period) — advisor"),
    dict(group="new_index", variable="EVI2", formula="2.5*(B8-B4)/(B8+2.4*B4+1)", rationale="EVI without blue band (more robust)"),
    dict(group="new_index", variable="NDRE", formula="(B8-B5)/(B8+B5)", rationale="red-edge; higher yield correlation than NDVI — advisor"),
    dict(group="new_index", variable="CIre", formula="(B7/B5)-1", rationale="red-edge chlorophyll index"),
    dict(group="new_index", variable="NDWI", formula="(B8-B11)/(B8+B11)", rationale="canopy water content / stress"),
    dict(group="new_index", variable="GNDVI", formula="(B8-B3)/(B8+B3)", rationale="green-band chlorophyll sensitivity"),
    dict(group="new_index", variable="OSAVI", formula="(B8-B4)/(B8+B4+0.16)", rationale="soil-adjusted (sparse canopy)"),
    # soil (existing + new) — context
    dict(group="soil", variable="clay/sand/silt/phh2o/soc (0-30)", formula="SoilGrids depth-weighted", rationale="thesis soil set"),
    dict(group="soil", variable="cec/bdod/nitrogen/cfvo + AWC", formula="SoilGrids + Saxton-Rawls AWC", rationale="enrichment (water/nutrient capacity)"),
]

# 2) phenological distribution metrics (advisor: replace single NDVI by spatial-component metrics)
PHENO_METRICS = [
    dict(metric="mean", definition="district cropland mean", meaning="central tendency"),
    dict(metric="median", definition="district cropland median", meaning="robust central tendency"),
    dict(metric="stdDev", definition="std across cropland pixels", meaning="intra-district heterogeneity (advisor)"),
    dict(metric="cv", definition="stdDev/mean", meaning="relative variability (advisor)"),
    dict(metric="P10", definition="10th percentile", meaning="weak-field tail (advisor)"),
    dict(metric="P90", definition="90th percentile", meaning="strong-field tail (advisor)"),
    dict(metric="range", definition="P90-P10", meaning="heterogeneity spread (advisor)"),
]

# 3) crop masking + phenological windows (advisor table)
MASK_WINDOWS = []
for crop, label in [("bugday", "winter wheat"), ("aycicegi", "sunflower")]:
    for win, dates in E.CROP_WINDOWS[crop].items():
        (sm, sd), (em, ed) = dates
        MASK_WINDOWS.append(dict(crop=label, window=win, months=f"{sm:02d}-{em:02d}",
                                 mask="crop-specific phenology classification (validated vs TÜİK ekilen_alan)"))

# 4) tier definitions (advisor-aligned)
TIERS = [
    dict(tier="A", composition="climate only (unchanged)", note="baseline"),
    dict(tier="B", composition="climate + {NDVI, NDRE, EVI} (crop-specific window means)", note="advisor B"),
    dict(tier="C", composition="B + soil", note="advisor C"),
    dict(tier="D", composition="C + phenological distribution metrics (median/std/CV/P10/P90/range)", note="advisor D"),
]


def main():
    T = E.TAB
    pd.DataFrame(RS_INVENTORY).to_csv(T / "rs_variable_inventory.csv", index=False)
    pd.DataFrame(PHENO_METRICS).to_csv(T / "phenological_metrics.csv", index=False)
    pd.DataFrame(MASK_WINDOWS).to_csv(T / "crop_masking_windows.csv", index=False)
    pd.DataFrame(TIERS).to_csv(T / "tier_definitions.csv", index=False)
    print("[p2b] saved rs_variable_inventory, phenological_metrics, crop_masking_windows, tier_definitions", flush=True)
    print(pd.DataFrame(MASK_WINDOWS).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
