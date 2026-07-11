"""R06 — Global Moran's I + k-sensitivity (#12).

Reproduces the GLOBAL Moran's I on Layer A champion LOYO residuals (district-mean), KNN k=4,
row-standardised weights, 999 permutations — and adds a k = 3,4,5,6 sensitivity.
Output: revisions/morans_i_sensitivity.csv  (crop, k, morans_I, E_I, z_norm, p_norm, p_sim, n_ilce)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as R  # noqa: E402
rc = R.rc

CP25 = rc.PROJECT_ROOT / "reports" / "cp25"
CHAMP = {"bugday": "elastic_net", "aycicegi": "random_forest"}  # Layer A LOYO champions


def main():
    np.random.seed(42)
    from libpysal.weights import KNN
    from esda.moran import Moran
    coords = pd.read_csv(rc.COORDS_PATH)
    ps = pd.read_csv(R.ANALYSIS / "per_sample_predictions.csv")
    rows = []
    for crop in ("bugday", "aycicegi"):
        g = ps[(ps.layer == "A") & (ps.crop == crop) & (ps.cv == "LOYO") &
               (ps.model == CHAMP[crop])].copy()
        g["residual"] = g.y_true - g.y_pred
        agg = (g.groupby("ilce_id")["residual"].mean().reset_index()
               .merge(coords[["ilce_id", "lat", "lon"]], on="ilce_id", how="inner"))
        pts = agg[["lat", "lon"]].values
        for k in (3, 4, 5, 6):
            w = KNN.from_array(pts, k=k); w.transform = "R"
            mi = Moran(agg["residual"].values, w, permutations=999)
            rows.append(dict(crop=crop, k=k, morans_I=round(float(mi.I), 4),
                             E_I=round(float(mi.EI), 4), z_norm=round(float(mi.z_norm), 4),
                             p_norm=round(float(mi.p_norm), 4), p_sim=round(float(mi.p_sim), 4),
                             n_ilce=int(len(agg)),
                             scale="GLOBAL Moran's I (single statistic over all districts)"))
    df = pd.DataFrame(rows)
    df.to_csv(R.REV / "morans_i_sensitivity.csv", index=False)
    print(f"[save] morans_i_sensitivity.csv rows={len(df)}", flush=True)
    print(df[["crop", "k", "morans_I", "p_norm", "p_sim", "n_ilce"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
