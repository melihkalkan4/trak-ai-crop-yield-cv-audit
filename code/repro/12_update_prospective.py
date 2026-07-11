"""12 — Gerçek-koordinat prospektif sonuçlarını konsolide et + placeholder ile kıyasla.

11_prospective_real_coords.py bittikten sonra çalıştırılır. Üretir:
* tables/T3_per_stage_real.csv          — gerçek-koordinat per-stage (raw S2 + unified)
* analysis/prospective_overall_real.csv — gerçek genel model vs persistence
* analysis/prospective_placeholder_vs_real.csv — placeholder vs gerçek karşılaştırma
* figures/F3b_per_stage_real.{pdf,png}  — gerçek-koordinat senescence figürü

Yalnızca mevcut çıktıları işler (eksik pencere/kaynağa dayanıklı). Uydurma yok.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

REAL = rc.ANALYSIS_DIR / "prospective_real"
PROSP = rc.PROJECT_ROOT / "reports" / "prospective"
STAGE_ORDER = ["pre_season", "emergence", "vegetative", "flowering",
               "grain_fill", "maturity", "post_harvest"]


def main() -> None:
    summ_path = REAL / "real_coords_validation_summary.json"
    if not summ_path.exists():
        print("real_coords_validation_summary.json not found — run step 11 first.", flush=True)
        return
    summ = json.loads(summ_path.read_text(encoding="utf-8"))
    site = summ["site"]
    print(f"[real] site area={site.get('area_ha')} ha, buffer={site.get('inward_buffer_m_used')} m", flush=True)

    # ---- per-stage real (concat available per_stage_*.csv) ----
    frames = []
    for f in sorted(REAL.glob("per_stage_*.csv")):
        d = pd.read_csv(f)
        frames.append(d)
    if frames:
        ps = pd.concat(frames, ignore_index=True)
        ps = ps.rename(columns={"R2": "r2", "MAE": "mae", "RMSE": "rmse", "MAPE_pct": "mape_pct"})
        ps.to_csv(rc.TABLES_DIR / "T3_per_stage_real.csv", index=False)
        print(f"[save] T3_per_stage_real.csv rows={len(ps)}", flush=True)
    else:
        ps = pd.DataFrame()

    # ---- overall model vs persistence (real), both sources ----
    rows = []
    for w in summ.get("windows", []):
        yl = w.get("year")
        for tag in ("raw_s2", "unified"):
            if tag not in w:
                continue
            b = w[tag]
            m, p, wl = b["overall_model"], b["overall_naive_persistence"], b["wilcoxon_model_vs_naive"]
            rows.append(dict(
                year=yl, actual_source=tag, n_matched=b.get("n_matched"),
                coverage_pct=b.get("coverage_pct"),
                model_R2=round(m.get("R2", float("nan")), 4),
                model_MAE=round(m.get("MAE", float("nan")), 4),
                persistence_R2=round(p.get("R2", float("nan")), 4),
                persistence_MAE=round(p.get("MAE", float("nan")), 4),
                model_beats_persistence=bool(m.get("R2", -9) > p.get("R2", 9)
                                             and m.get("MAE", 9) < p.get("MAE", -9)),
                wilcoxon_p=wl.get("wilcoxon_p_value"),
            ))
    ov = pd.DataFrame(rows)
    ov.to_csv(rc.ANALYSIS_DIR / "prospective_overall_real.csv", index=False)
    print("[save] prospective_overall_real.csv", flush=True)
    if not ov.empty:
        print(ov.to_string(index=False), flush=True)

    # ---- placeholder vs real comparison (overall, unified source for fairness) ----
    comp = []
    for yl in ("2025", "2026"):
        ph_path = PROSP / f"EVR_01_{yl}_validation_summary.json"
        ph = json.loads(ph_path.read_text(encoding="utf-8")) if ph_path.exists() else None
        real_w = next((w for w in summ.get("windows", []) if str(w.get("year")) == yl), None)
        row = {"year": yl}
        if ph:
            row.update(placeholder_model_R2=round(ph["overall_model"]["R2"], 4),
                       placeholder_persistence_R2=round(ph["overall_naive_persistence"]["R2"], 4),
                       placeholder_n=ph.get("n_matched"))
        if real_w and "unified" in real_w:
            b = real_w["unified"]
            row.update(real_model_R2=round(b["overall_model"]["R2"], 4),
                       real_persistence_R2=round(b["overall_naive_persistence"]["R2"], 4),
                       real_n=b.get("n_matched"))
        comp.append(row)
    pd.DataFrame(comp).to_csv(rc.ANALYSIS_DIR / "prospective_placeholder_vs_real.csv", index=False)
    print("[save] prospective_placeholder_vs_real.csv", flush=True)

    # ---- figure: real-coords per-stage (prefer raw S2; 2025 if present) ----
    if not ps.empty:
        src = "raw_s2" if (ps["actual_source"] == "raw_s2").any() else ps["actual_source"].iloc[0]
        yr = 2025 if (ps["year"] == 2025).any() else ps["year"].iloc[0]
        d = ps[(ps["actual_source"] == src) & (ps["year"] == yr)].set_index("stage").reindex(
            STAGE_ORDER).dropna(subset=["r2"])
        if not d.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(d))
            colors = ["#C44E52" if v < 0 else "#4C72B0" for v in d["r2"]]
            ax.bar(x, d["r2"], color=colors, alpha=0.9)
            for xi, (v, n) in enumerate(zip(d["r2"], d["n"])):
                ax.annotate(f"n={int(n)}", (xi, v), textcoords="offset points",
                            xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=7)
            ax.axhline(0, color="black", lw=1)
            ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=30, ha="right")
            ax.set_ylabel("$R^2$ (NDVI t+7 vs observed)")
            ax.set_title(f"Per-stage NDVI forecast skill — REAL parcel ({site.get('area_ha')} ha), "
                         f"{yr}, actuals={src}")
            fig.savefig(rc.FIGURES_DIR / "F3b_per_stage_real.pdf", bbox_inches="tight")
            fig.savefig(rc.FIGURES_DIR / "F3b_per_stage_real.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("[fig] F3b_per_stage_real.pdf/.png", flush=True)


if __name__ == "__main__":
    main()
