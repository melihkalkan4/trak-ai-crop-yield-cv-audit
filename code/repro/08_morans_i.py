"""08 — Moran's I (residual mekânsal otokorelasyon) sadık reprodüksiyon.

src/cp25/11_spatial_diagnostics.py ile BİREBİR aynı yöntem:
* residual = verim_kg_da − yield_pred_loyo  (Layer A şampiyon, 05_loocv_predictions)
* ilçe başına ortalama residual
* KNN(k=4) ağırlıkları lat/lon centroid'lerden, row-standardize (transform="R")
* esda.Moran(residuals, w, permutations=999)

Gözlemlenen I deterministiktir → yayınlanan değerle eşleşmeli (wheat +0.257,
sunflower +0.117). p_sim permütasyon rastgeleliğiyle ufak oynayabilir (seed
sabitlenir). Reproduksiyon başarısızsa yayınlanan tez sonucu provenance ile
kullanılır (prompt 1.8).

Çıktı: analysis/morans_i.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

CP25 = rc.PROJECT_ROOT / "reports" / "cp25"
PUBLISHED_I = {"bugday": 0.257, "aycicegi": 0.117}


def main() -> None:
    np.random.seed(42)
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
    except Exception as e:  # noqa: BLE001
        print(f"esda/libpysal unavailable ({e}); using published thesis values.", flush=True)
        pd.DataFrame([
            dict(crop=c, I=PUBLISHED_I[c], source="published (11_spatial_diagnostics.md)",
                 reproduced=False) for c in PUBLISHED_I]).to_csv(
            rc.ANALYSIS_DIR / "morans_i.csv", index=False)
        return

    coords = pd.read_csv(rc.COORDS_PATH)
    rows = []
    for crop in ("bugday", "aycicegi"):
        p = CP25 / f"05_loocv_predictions_{crop}.csv"
        df = pd.read_csv(p)
        df["residual"] = df["verim_kg_da"] - df["yield_pred_loyo"]
        agg = df.groupby("ilce_id")["residual"].mean().reset_index()
        merged = agg.merge(coords[["ilce_id", "lat", "lon"]], on="ilce_id", how="inner")
        w = KNN.from_array(merged[["lat", "lon"]].values, k=4)
        w.transform = "R"
        mi = Moran(merged["residual"].values, w, permutations=999)
        d_pub = abs(round(float(mi.I), 3) - PUBLISHED_I[crop])
        rows.append(dict(
            crop=crop, I=round(float(mi.I), 4), expected_I=round(float(mi.EI), 4),
            z_norm=round(float(mi.z_norm), 4), p_norm=round(float(mi.p_norm), 4),
            p_sim=round(float(mi.p_sim), 4), n_ilce=int(len(merged)),
            published_I=PUBLISHED_I[crop], abs_diff_vs_published=round(d_pub, 4),
            reproduced=bool(d_pub <= 0.002), source="reproduced"))
    out = pd.DataFrame(rows)
    out.to_csv(rc.ANALYSIS_DIR / "morans_i.csv", index=False)
    print(f"[save] morans_i.csv", flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
