"""R08 — Spatiotemporal blocking definition + membership (#11).

The cp25 'Spatiotemporal' CV assigns each observation to a block =
(year_block * n_clusters + space_cluster), with 5 year-blocks (linspace over the year range)
and 5 KMeans(lat/lon) spatial clusters → up to 25 blocks; LeaveOneGroupOut then holds out ONE
block (one year-block × space-cluster CELL) at a time.

⇒ This is **Scenario A (cell holdout) = 'spatiotemporal block interpolation'**: when a cell is
held out, OTHER years of the same regions AND other regions of the same years remain in training.
It is NOT strict spatiotemporal extrapolation (Scenario B). We name it accordingly.

Output: revisions/spatiotemporal_blocks.csv (which districts in which spatial cluster; which years
in which year-block) + revisions/spatiotemporal_scenario.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402
rc = R.rc


def main():
    rows = []
    for tier in ("A",):  # tier A spans the full panel; membership is representative
        for crop in rc.CROPS:
            sub = rc.crop_subset(rc.load_layer(tier), crop)
            years = sub["year"].astype(int).values
            yr_min, yr_max = years.min(), years.max()
            yr_bins = np.linspace(yr_min, yr_max + 1, 6)
            yr_block = np.digitize(years, yr_bins[1:-1])
            from sklearn.cluster import KMeans
            coords = pd.read_csv(rc.COORDS_PATH)[["ilce_id", "lat", "lon"]]
            dfl = sub.merge(coords, on="ilce_id", how="left")
            n_eff = min(5, dfl["ilce_id"].nunique())
            km = KMeans(n_clusters=n_eff, random_state=42, n_init=10)
            sp = km.fit_predict(dfl[["lat", "lon"]].values)
            tmp = sub[["ilce_id", "ilce", "year"]].copy()
            tmp["space_cluster"] = sp
            tmp["year_block"] = yr_block
            tmp["block_id"] = yr_block * n_eff + sp
            tmp["crop"] = crop
            tmp["tier"] = tier
            rows.append(tmp)
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(R.REV / "spatiotemporal_blocks_membership.csv", index=False)

    # compact membership tables
    sp_map = (full[full.crop == "bugday"].groupby("space_cluster")["ilce"]
              .apply(lambda s: ", ".join(sorted(set(s)))).reset_index())
    yr_map = (full[full.crop == "bugday"].groupby("year_block")["year"]
              .apply(lambda s: f"{int(min(s))}-{int(max(s))}").reset_index())
    sp_map.to_csv(R.REV / "spatiotemporal_blocks.csv", index=False)
    print("[save] spatiotemporal_blocks.csv + _membership.csv", flush=True)
    print("\n=== spatial clusters (wheat) ===", flush=True)
    print(sp_map.to_string(index=False), flush=True)
    print("\n=== year-blocks (wheat) ===", flush=True)
    print(yr_map.to_string(index=False), flush=True)

    txt = (
        "SPATIOTEMPORAL CV SCENARIO (#11)\n"
        "================================\n"
        "Implemented in src/cp25 _block_groups + LeaveOneGroupOut over block_id =\n"
        "  year_block(5, linspace over year range) * n_space_clusters + KMeans(lat/lon, k=5).\n\n"
        "SCENARIO = A  (cell holdout) = 'spatiotemporal block interpolation'.\n"
        "Holding out one (year-block x space-cluster) cell leaves, in training, both other years of\n"
        "the same regions and other regions of the same years. This is therefore NOT strict\n"
        "spatiotemporal extrapolation (Scenario B, where the test year's ALL regions and the test\n"
        "region's ALL years would be removed). The paper should name this regime 'spatiotemporal\n"
        "block interpolation' (Scenario A), distinct from LOYO (temporal) and LOILO (spatial).\n\n"
        "Scenario B (strict extrapolation) is an optional stricter variant; not run here to avoid\n"
        "changing the published metric definition — flagged in HANDOFF as available on request.\n"
    )
    (R.REV / "spatiotemporal_scenario.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)


if __name__ == "__main__":
    main()
