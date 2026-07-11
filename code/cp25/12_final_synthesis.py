"""ÇP-2.5 / Görev 12 — Final Sentez.

Tüm görev raporlarını birleştiren master karşılaştırma tablosu,
şampiyon model seçimi, hipotez sonuçları, inference entegrasyonu.

Çıktılar
--------
* ``reports/cp25/12_final_synthesis.md``
* ``reports/cp25/12_master_comparison.csv``
* ``models/cp25/champion_{crop}.pkl``  (en iyi katman/model bundle)
* ``models/cp25/champion_metadata.json``
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("cp25.task12")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
MODELS_DIR   = PROJECT_ROOT / "models" / "cp25"


def _load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:                                                # noqa: BLE001
        return pd.DataFrame()


def _best_loyo(df: pd.DataFrame, crop: str) -> dict | None:
    if df.empty or "crop" not in df.columns or "cv" not in df.columns:
        return None
    sub = df[(df["crop"] == crop) & (df["cv"] == "LOYO")]
    if sub.empty:
        return None
    best = sub.sort_values("rmse_kg_da").iloc[0]
    return best.to_dict()


def main() -> None:
    # ---- Baselines ----
    bl = _load_csv(REPORT_DIR / "02_baselines.csv")
    la = _load_csv(REPORT_DIR / "05_layer_a_results.csv")
    lb = _load_csv(REPORT_DIR / "06_layer_b_results.csv")
    lc = _load_csv(REPORT_DIR / "07_layer_c_results.csv")

    # ---- Master Comparison Table ----
    rows = []
    for crop in ("bugday", "aycicegi"):
        # B0 climatology
        if not bl.empty:
            sub = bl[(bl["crop"] == crop) & (bl["model"] == "B0_Climatology")]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append({"layer": "B0", "model": "Climatology",
                             "crop": crop, "n": int(r.get("n", 0)),
                             "r2_loyo": float(r.get("r2", 0)),
                             "rmse_loyo": float(r.get("rmse_kg_da", 0)),
                             "ss_loyo": 0.0})
        # B3 climate-mean naive
        if not bl.empty:
            sub = bl[(bl["crop"] == crop) & (bl["model"] == "B3_ClimateProxy")]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append({"layer": "B3", "model": "ClimateProxy",
                             "crop": crop, "n": int(r.get("n", 0)),
                             "r2_loyo": float(r.get("r2", 0)),
                             "rmse_loyo": float(r.get("rmse_kg_da", 0)),
                             "ss_loyo": float(r.get("skill_score_vs_B0", 0))})
        for layer_name, layer_df in (("A", la), ("B", lb), ("C", lc)):
            best = _best_loyo(layer_df, crop)
            if not best: continue
            rows.append({"layer": layer_name, "model": best["model"],
                         "crop": crop, "n": int(best.get("n", 0)),
                         "r2_loyo": float(best["r2"]),
                         "rmse_loyo": float(best["rmse_kg_da"]),
                         "ss_loyo": float(best.get("ss_vs_b0", 0) or 0)})
    master = pd.DataFrame(rows)
    master.to_csv(REPORT_DIR / "12_master_comparison.csv", index=False)

    # ---- Champion selection: highest SS_LOYO (positive only) ----
    champions = {}
    for crop in ("bugday", "aycicegi"):
        sub = master[(master["crop"] == crop) & (master["ss_loyo"] > 0)]
        if sub.empty:
            sub = master[master["crop"] == crop]
        champ = sub.sort_values("ss_loyo", ascending=False).iloc[0]
        champions[crop] = champ.to_dict()

    # Copy champion bundle (if Layer artifact exists)
    for crop, champ in champions.items():
        layer = champ["layer"]
        if layer in ("A", "B", "C"):
            src = MODELS_DIR / f"layer_{layer.lower()}_{crop}.pkl"
            dst = MODELS_DIR / f"champion_{crop}.pkl"
            if src.exists():
                shutil.copy2(src, dst)
                logger.info("champion %s ← Layer %s (%s → %s)",
                            crop, layer, src.name, dst.name)

    # ---- Hypothesis results ----
    hypotheses = {
        "H1": {
            "claim": "İlçe-bazlı (n=1165) il-bazlıyı (n=132) outperform eder",
            "delta_r2_loyo_bugday": _delta_h1("bugday", la),
            "delta_r2_loyo_aycicegi": _delta_h1("aycicegi", la),
            "verdict_loyo": "LOYO için RED, LOILO için PASS",
        },
        "H2": {
            "claim": "NDVI ekleme climate-only baseline'ı outperform eder",
            "delta_r2_loyo_bugday": _delta_layer("bugday", la, lb, "LOYO"),
            "delta_r2_loyo_aycicegi": _delta_layer("aycicegi", la, lb, "LOYO"),
            "verdict": _h_verdict_two_crop(la, lb, 0.10, "LOYO"),
        },
        "H3": {
            "claim": "Multimodal füzyon (Layer C) Layer B'yi outperform eder",
            "delta_r2_loyo_bugday": _delta_layer("bugday", lb, lc, "LOYO"),
            "delta_r2_loyo_aycicegi": _delta_layer("aycicegi", lb, lc, "LOYO"),
            "verdict": _h_verdict_two_crop(lb, lc, 0.05, "LOYO"),
        },
        "H4": {"claim": "Anomali yıllarda SS > 0.30",
               "verdict": "Görev 9 raporlarından oku"},
        "H5": {"claim": "|LOILO - LOYO| < 0.15",
               "delta_loyo_loilo_bugday": _delta_cv("bugday", la, "LOYO", "LOILO"),
               "delta_loyo_loilo_aycicegi": _delta_cv("aycicegi", la, "LOYO", "LOILO"),
               "verdict": "Layer A için reddedildi (fark 0.5+)"},
    }

    # ---- Metadata ----
    meta = {
        "version":    "cp25-v2.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "champions":  champions,
        "hypotheses": hypotheses,
        "source_data": {
            "yields":  "TÜİK ilçe-bazlı (data/external/tuik/tuik_ilce_yields_clean.csv)",
            "climate": "NASA POWER MERRA-2 (data/processed/openmeteo_ilce/)",
            "ndvi":    "Sentinel-2 (data/processed/ndvi_ilce/)",
            "soil":    "ISRIC SoilGrids (data/processed/soil_ilce.csv)",
        },
    }
    with (MODELS_DIR / "champion_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("champion metadata → %s", MODELS_DIR / "champion_metadata.json")

    # ---- Markdown synthesis ----
    md = ["# ÇP-2.5 — Görev 12: Final Sentez", "",
          f"Üretim tarihi (UTC): {meta['generated_utc']}", "",
          "## Master Karşılaştırma Tablosu (LOYO)", "",
          "| Layer | Model | Ürün | n | R² | RMSE | Skill Score |",
          "|---|---|---|---|---|---|---|"]
    for _, r in master.iterrows():
        md.append(f"| {r['layer']} | {r['model']} | {r['crop']} | {r['n']} | "
                  f"{r['r2_loyo']:+.3f} | {r['rmse_loyo']:.1f} | "
                  f"{r['ss_loyo']:+.3f} |")
    md.append("")
    md.append("## Şampiyon Modeller")
    md.append("")
    for crop, champ in champions.items():
        md.append(f"- **{crop}**: Layer {champ['layer']} / {champ['model']}  "
                  f"(R²={champ['r2_loyo']:+.3f}, RMSE={champ['rmse_loyo']:.1f}, "
                  f"SS={champ['ss_loyo']:+.3f})")
    md.append("")
    md.append("## Hipotez Sonuçları")
    md.append("")
    for h, info in hypotheses.items():
        md.append(f"### {h}: {info['claim']}")
        md.append("")
        for k, v in info.items():
            if k == "claim": continue
            md.append(f"- {k}: `{v}`")
        md.append("")
    md.append("## Veri Kaynak Manifesti")
    md.append("")
    for k, v in meta["source_data"].items():
        md.append(f"- {k}: `{v}`")

    (REPORT_DIR / "12_final_synthesis.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("rapor → 12_final_synthesis.md")
    print("\n=== MASTER TABLE ===")
    print(master.to_string(index=False))
    print("\n=== CHAMPIONS ===")
    for crop, c in champions.items():
        print(f"{crop:10s}: Layer {c['layer']} / {c['model']:14s}  "
              f"R²={c['r2_loyo']:+.3f}  SS={c['ss_loyo']:+.3f}")


def _delta_h1(crop: str, la: pd.DataFrame) -> str:
    v1 = {"bugday": -0.085, "aycicegi": 0.646}
    best = _best_loyo(la, crop)
    if not best: return "NA"
    return f"v2={best['r2']:+.3f} vs v1={v1[crop]:+.3f} → ΔR²={best['r2']-v1[crop]:+.3f}"


def _delta_layer(crop: str, src: pd.DataFrame, dst: pd.DataFrame, cv: str) -> str:
    if src.empty or dst.empty or "crop" not in src.columns or "crop" not in dst.columns:
        return "NA (Layer eksik)"
    s = src[(src["crop"] == crop) & (src["cv"] == cv)]
    d = dst[(dst["crop"] == crop) & (dst["cv"] == cv)]
    if s.empty or d.empty: return "NA (Layer eksik)"
    r2s = s.sort_values("rmse_kg_da").iloc[0]["r2"]
    r2d = d.sort_values("rmse_kg_da").iloc[0]["r2"]
    return f"{r2s:+.3f} → {r2d:+.3f}  ΔR²={r2d-r2s:+.3f}"


def _h_verdict_two_crop(src, dst, thr, cv):
    if src.empty or dst.empty or "crop" not in src.columns or "crop" not in dst.columns:
        return "Bekleniyor"
    out = []
    for crop in ("bugday", "aycicegi"):
        s = src[(src["crop"] == crop) & (src["cv"] == cv)]
        d = dst[(dst["crop"] == crop) & (dst["cv"] == cv)]
        if s.empty or d.empty: continue
        r2s = s.sort_values("rmse_kg_da").iloc[0]["r2"]
        r2d = d.sort_values("rmse_kg_da").iloc[0]["r2"]
        out.append(f"{crop}: {'PASS' if r2d - r2s >= thr else 'FAIL'} (Δ={r2d-r2s:+.3f})")
    return "; ".join(out)


def _delta_cv(crop, df, cv_a, cv_b):
    if df.empty or "crop" not in df.columns: return "NA"
    a = df[(df["crop"] == crop) & (df["cv"] == cv_a)]
    b = df[(df["crop"] == crop) & (df["cv"] == cv_b)]
    if a.empty or b.empty: return "NA"
    r2a = a.sort_values("rmse_kg_da").iloc[0]["r2"]
    r2b = b.sort_values("rmse_kg_da").iloc[0]["r2"]
    return f"{cv_a}={r2a:+.3f}, {cv_b}={r2b:+.3f}, Δ={r2b-r2a:+.3f}"


if __name__ == "__main__":
    main()
