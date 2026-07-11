"""10 — Dergi-kalitesinde figürler (F1–F6).

Tüm figürler analysis/ ve tables/ çıktılarından üretilir (sadık reprodüksiyon).
Her figür hem .pdf (vektör) hem .png (300 dpi) olarak figures/ altına yazılır.

F1 pred-vs-actual (Layer C, LOYO, ürün başına)
F2 skill-score vs B0 (LOYO) + bootstrap CI
F3 per-stage NDVI R² (senescence çöküşü)
F4 LOYO vs LOILO vs Spatiotemporal R² (genelleme açığı)
F5 Moran's I — Layer A LOYO residual haritası
F6 permütasyon önem (Layer C)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_common as rc  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})
CROP_NAME = {"bugday": "Winter wheat", "aycicegi": "Sunflower"}
C_WHEAT, C_SUN = "#C44E52", "#4C72B0"
CROP_COLOR = {"bugday": C_WHEAT, "aycicegi": C_SUN}
FIG = rc.FIGURES_DIR


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"[fig] {name}.pdf/.png", flush=True)


def f1_pred_vs_actual(ps):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, crop in zip(axes, ("bugday", "aycicegi")):
        sub = ps[(ps.layer == "C") & (ps.crop == crop) & (ps.cv == "LOYO")]
        # best Layer C LOYO model by R2
        r2s = {m: r2_score(g.y_true, g.y_pred) for m, g in sub.groupby("model")}
        bm = max(r2s, key=r2s.get)
        g = sub[sub.model == bm]
        yt, yp = g.y_true.values, g.y_pred.values
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.scatter(yt, yp, s=22, alpha=0.5, color=CROP_COLOR[crop], edgecolor="none")
        lo = min(yt.min(), yp.min()) * 0.95
        hi = max(yt.max(), yp.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed yield (kg da$^{-1}$)")
        ax.set_ylabel("Predicted yield (kg da$^{-1}$)")
        ax.set_title(f"{CROP_NAME[crop]} — Layer C ({bm}), LOYO\n"
                     f"$R^2$={r2:+.3f}, RMSE={rmse:.1f}, n={len(g)}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Multimodal model under temporal generalization (Leave-One-Year-Out)",
                 fontsize=12, y=1.02)
    _save(fig, "F1_pred_vs_actual")


def f2_skill_score(bs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, crop in zip(axes, ("bugday", "aycicegi")):
        cc = bs[bs.crop == crop].copy()
        cc["label"] = "L" + cc.layer + "·" + cc.model
        cc = cc.sort_values("skill_score")
        y = np.arange(len(cc))
        err_lo = cc.skill_score - cc.ss_ci_lo
        err_hi = cc.ss_ci_hi - cc.skill_score
        colors = ["#55A868" if b else ("#C44E52" if w else "#999999")
                  for b, w in zip(cc.beats_b0_ci, cc.worse_than_b0_ci)]
        ax.barh(y, cc.skill_score, xerr=[err_lo, err_hi], color=colors,
                error_kw=dict(lw=0.8, capsize=2), alpha=0.9)
        ax.axvline(0, color="black", lw=1.2, label="B0 climatology")
        ax.set_yticks(y); ax.set_yticklabels(cc.label, fontsize=8)
        ax.set_xlabel("Skill score vs B0  (1 − RMSE/RMSE$_{B0}$)")
        ax.set_title(f"{CROP_NAME[crop]} — LOYO")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Skill vs climatology baseline (95% bootstrap CI; green=beats B0, red=worse)",
                 fontsize=12, y=1.02)
    _save(fig, "F2_skill_score_vs_B0")


def f3_per_stage(t3):
    order = ["pre_season", "emergence", "vegetative", "flowering",
             "grain_fill", "maturity", "post_harvest"]
    d25 = t3[t3.year == 2025].set_index("stage").reindex(order).dropna(subset=["r2"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(d25))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in d25.r2]
    ax.bar(x, d25.r2, color=colors, alpha=0.9)
    for xi, (v, n) in enumerate(zip(d25.r2, d25.n)):
        ax.annotate(f"n={int(n)}", (xi, v), textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(d25.index, rotation=30, ha="right")
    ax.set_ylabel("$R^2$ (NDVI t+7 forecast vs observed)")
    ax.set_title("Per-phenology-stage NDVI forecast skill — EVR_01, 2025 season\n"
                 "(senescence collapse at grain-fill / maturity)")
    _save(fig, "F3_per_stage_ndvi")


def f4_gap(ps):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    cvs = ["LOYO", "LOILO", "Spatiotemporal"]
    cv_lab = ["LOYO\n(temporal)", "LOILO\n(spatial)", "Spatio-\ntemporal"]
    cv_col = ["#C44E52", "#55A868", "#8172B3"]
    for ax, crop in zip(axes, ("bugday", "aycicegi")):
        layers = ["A", "B", "C"]
        x = np.arange(len(layers)); w = 0.26
        for j, cv in enumerate(cvs):
            vals = []
            for L in layers:
                g = ps[(ps.layer == L) & (ps.crop == crop) & (ps.cv == cv)]
                r2s = [r2_score(gg.y_true, gg.y_pred) for _, gg in g.groupby("model")]
                vals.append(max(r2s))
            ax.bar(x + (j - 1) * w, vals, w, label=cv_lab[j], color=cv_col[j], alpha=0.9)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["A\nclimate", "B\n+NDVI", "C\n+soil"])
        ax.set_title(CROP_NAME[crop])
        ax.set_ylabel("best $R^2$ (per regime)")
    axes[1].legend(fontsize=8, loc="lower right", title="CV regime")
    fig.suptitle("Spatial vs temporal generalization gap (best model per regime)",
                 fontsize=12, y=1.02)
    _save(fig, "F4_generalization_gap")


def f5_morans(ps, mi):
    coords = pd.read_csv(rc.COORDS_PATH)
    champ = {"bugday": "elastic_net", "aycicegi": "random_forest"}  # Layer A LOYO champions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, crop in zip(axes, ("bugday", "aycicegi")):
        g = ps[(ps.layer == "A") & (ps.crop == crop) & (ps.cv == "LOYO") &
               (ps.model == champ[crop])].copy()
        g["residual"] = g.y_true - g.y_pred
        agg = (g.groupby("ilce_id")["residual"].mean().reset_index()
               .merge(coords[["ilce_id", "lat", "lon", "ilce"]], on="ilce_id", how="inner"))
        vmax = np.abs(agg.residual).max()
        sc = ax.scatter(agg.lon, agg.lat, c=agg.residual, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, s=90, edgecolor="black", linewidth=0.4)
        plt.colorbar(sc, ax=ax, label="mean LOYO residual (kg da$^{-1}$)")
        row = mi[mi.crop == crop].iloc[0]
        ax.set_title(f"{CROP_NAME[crop]} — Moran's I = {row.I:+.3f} "
                     f"(p$_{{norm}}$={row.p_norm:.3f}, n={int(row.n_ilce)})")
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    fig.suptitle("Spatial autocorrelation of Layer A residuals (KNN k=4)",
                 fontsize=12, y=1.0)
    _save(fig, "F5_morans_i_residuals")


def f6_importance(t4):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, crop in zip(axes, ("bugday", "aycicegi")):
        d = t4[(t4.layer == "C") & (t4.crop == crop)].sort_values(
            "imp_mean", ascending=True).tail(10)
        y = np.arange(len(d))
        ax.barh(y, d.imp_mean, xerr=d.imp_std, color=CROP_COLOR[crop], alpha=0.9,
                error_kw=dict(lw=0.7, capsize=2))
        ax.set_yticks(y); ax.set_yticklabels(d.feature, fontsize=8)
        ax.set_xlabel("Permutation importance (Δ score)")
        ax.set_title(f"{CROP_NAME[crop]} — Layer C")
    fig.suptitle("Permutation feature importance (multimodal model)", fontsize=12, y=1.0)
    _save(fig, "F6_feature_importance")


def main():
    ps = pd.read_csv(rc.ANALYSIS_DIR / "per_sample_predictions.csv")
    f1_pred_vs_actual(ps)
    f4_gap(ps)
    t3 = pd.read_csv(rc.TABLES_DIR / "T3_per_stage.csv")
    f3_per_stage(t3)
    mi = pd.read_csv(rc.ANALYSIS_DIR / "morans_i.csv")
    f5_morans(ps, mi)
    t4 = pd.read_csv(rc.TABLES_DIR / "T4_feature_importance.csv")
    f6_importance(t4)
    bs_path = rc.ANALYSIS_DIR / "baseline_superiority.csv"
    if bs_path.exists():
        f2_skill_score(pd.read_csv(bs_path))
    else:
        print("[fig] F2 skipped — baseline_superiority.csv not ready yet", flush=True)
    print("figures done", flush=True)


if __name__ == "__main__":
    main()
