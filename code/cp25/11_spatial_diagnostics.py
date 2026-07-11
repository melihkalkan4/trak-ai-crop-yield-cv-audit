"""ÇP-2.5 / Görev 11 — Mekânsal Tanılama (Moran's I).

Şampiyon Layer A modelinin LOYO residuals'ı üzerinde **Moran's I** test edilir.
H5 hipotezi: ``|R²_LOILO - R²_LOYO| < 0.15`` → spatial generalization güçlü.
Moran's I < 0.3 ise residuals mekânsal bağımsız (model spatial bilgiyi
yakalamış) ✓.  Aksi halde geographic feature eksikliği var.

Yöntem
------
1. KNN (k=4) komşuluk grafiği lat/lon centroid'lerden.
2. Her ürün × CV için residuals = y − ŷ_loyo (ya da loilo).
3. ``esda.Moran(residuals, w)`` → I değeri + p-value (permutation test).

Çıktılar
--------
* ``reports/cp25/11_spatial_diagnostics.md``
* ``reports/cp25/fig_morans_i.png``
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import KNN                                    # noqa: E402
from esda.moran import Moran                                        # noqa: E402

logger = logging.getLogger("cp25.task11")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
COORDS_PATH  = PROJECT_ROOT / "data" / "external" / "tuik" / "ilce_coords.csv"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _moran_for_residuals(df: pd.DataFrame, coords: pd.DataFrame) -> dict:
    """Aggregate residuals to ilce level (mean) then compute Moran's I."""
    df = df.copy()
    df["residual"] = df["verim_kg_da"] - df["yield_pred_loyo"]
    agg = df.groupby("ilce_id")["residual"].mean().reset_index()
    merged = agg.merge(coords[["ilce_id", "lat", "lon"]], on="ilce_id", how="inner")
    if len(merged) < 5:
        return {"I": float("nan"), "p_norm": float("nan"), "n": len(merged)}
    pts = merged[["lat", "lon"]].values
    w = KNN.from_array(pts, k=4)
    w.transform = "R"
    mi = Moran(merged["residual"].values, w, permutations=999)
    return {
        "I":         float(mi.I),
        "expected_I":float(mi.EI),
        "z_norm":    float(mi.z_norm),
        "p_norm":    float(mi.p_norm),
        "p_sim":     float(mi.p_sim),
        "n":         int(len(merged)),
    }


def main() -> None:
    coords = pd.read_csv(COORDS_PATH)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, crop in zip(axes, ["bugday", "aycicegi"]):
        p = REPORT_DIR / f"05_loocv_predictions_{crop}.csv"
        if not p.exists():
            logger.warning("missing: %s", p)
            continue
        df = pd.read_csv(p)
        res = _moran_for_residuals(df, coords)
        out[crop] = res
        logger.info("[%s] Moran's I = %+.3f  E[I]=%+.3f  z=%+.2f  p_norm=%.4f "
                    "p_sim=%.4f  n_ilce=%d",
                    crop, res["I"], res["expected_I"], res["z_norm"],
                    res["p_norm"], res["p_sim"], res["n"])

        # Map: residuals colored by sign
        df["residual"] = df["verim_kg_da"] - df["yield_pred_loyo"]
        agg = (df.groupby("ilce_id")["residual"].mean().reset_index()
                 .merge(coords[["ilce_id","lat","lon","ilce"]], on="ilce_id", how="inner"))
        sc = ax.scatter(agg["lon"], agg["lat"], c=agg["residual"], cmap="RdBu_r",
                        s=80, edgecolor="black", linewidth=0.4)
        plt.colorbar(sc, ax=ax, label="residual (kg/da)")
        for _, r in agg.iterrows():
            ax.annotate(r["ilce"][:5], (r["lon"], r["lat"]), fontsize=6, alpha=0.7)
        ax.set_title(f"{crop} LOYO residuals\nMoran's I = {res['I']:+.3f}  "
                     f"(p={res['p_sim']:.3f}, n={res['n']})")
        ax.set_xlabel("Boylam"); ax.set_ylabel("Enlem")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fig_morans_i.png", dpi=130); plt.close()

    # Markdown report
    md = ["# ÇP-2.5 — Görev 11: Mekânsal Tanılama (Moran's I)", "",
          "## Yöntem", "",
          "1. Layer A şampiyon model LOYO residuals (her ilçe için ortalama).",
          "2. KNN(k=4) komşuluk grafiği lat/lon centroid'lerden.",
          "3. ``esda.Moran`` ile global Moran's I + permutation test (999 iter).",
          "",
          "## Sonuçlar",
          ""]
    md.append("| Ürün | Moran's I | E[I] | z-norm | p_norm | p_sim | n_ilçe |")
    md.append("|---|---|---|---|---|---|---|")
    for crop, r in out.items():
        md.append(f"| {crop} | {r['I']:+.3f} | {r['expected_I']:+.3f} | "
                  f"{r['z_norm']:+.2f} | {r['p_norm']:.4f} | "
                  f"{r['p_sim']:.4f} | {r['n']} |")
    md.append("")
    md.append("## H5 Yorum")
    md.append("")
    for crop, r in out.items():
        if abs(r["I"]) < 0.3 and r["p_sim"] > 0.05:
            verdict = "🟢 Residuals mekânsal bağımsız — model spatial bilgiyi yakalamış."
        elif r["p_sim"] <= 0.05 and r["I"] > 0:
            verdict = ("🟡 Pozitif spatial autocorrelation tespit edildi "
                       "(p<0.05). Komşu ilçelerde benzer hata kalıbı → "
                       "geographic feature (lat/lon, soil, micro-climate) "
                       "modele eklenmeli.")
        else:
            verdict = "Sonuç belirsiz; örneklem küçük veya gürültülü."
        md.append(f"- **{crop}**: {verdict}")
    md.append("")
    md.append("## Görsel")
    md.append("`reports/cp25/fig_morans_i.png`")
    (REPORT_DIR / "11_spatial_diagnostics.md").write_text("\n".join(md), encoding="utf-8")

    print("=== ÖZET ===")
    for crop, r in out.items():
        print(f"{crop:9s}: I={r['I']:+.3f}, p_sim={r['p_sim']:.4f}, n={r['n']}")


if __name__ == "__main__":
    main()
