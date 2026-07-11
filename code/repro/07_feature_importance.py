"""07 — Permütasyon önem (XAI) konsolidasyonu.

Mevcut reports/cp25/08_perm_importance_{A,B,C}_{bugday,aycicegi}.csv dosyalarını
tek tabloya toplar (yeniden hesap YOK; permütasyon önemleri 08_xai_analysis.py
tarafından üretilmiştir). Top özellikler agronomiyle docs'ta ilişkilendirilir.

Çıktı: tables/T4_feature_importance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

CP25 = rc.PROJECT_ROOT / "reports" / "cp25"


def main() -> None:
    rows = []
    for layer in ("A", "B", "C"):
        for crop in ("bugday", "aycicegi"):
            f = CP25 / f"08_perm_importance_{layer}_{crop}.csv"
            if not f.exists():
                print(f"  missing {f}", flush=True)
                continue
            d = pd.read_csv(f).sort_values("imp_mean", ascending=False).reset_index(drop=True)
            d["rank"] = d.index + 1
            d.insert(0, "layer", layer)
            d.insert(1, "crop", crop)
            rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out = out[["layer", "crop", "rank", "feature", "imp_mean", "imp_std"]]
    out.to_csv(rc.TABLES_DIR / "T4_feature_importance.csv", index=False)
    print(f"[save] T4_feature_importance.csv rows={len(out)}", flush=True)
    print("\n=== Top-5 permutation importance per layer×crop ===", flush=True)
    for (layer, crop), g in out.groupby(["layer", "crop"]):
        top = g.head(5)
        feats = ", ".join(f"{r.feature}({r.imp_mean:.3f})" for r in top.itertuples())
        print(f"  L{layer} {crop:9s}: {feats}", flush=True)


if __name__ == "__main__":
    main()
