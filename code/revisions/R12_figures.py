"""R12 — Regenerate manuscript figures (#14) with referee fixes + English captions."""
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
import rev_common as R  # noqa: E402
rc = R.rc

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "savefig.bbox": "tight"})
FIG = R.RFIG
CN = {"bugday": "Winter wheat", "aycicegi": "Sunflower"}
COL = {"bugday": "#C44E52", "aycicegi": "#4C72B0"}
caps = []


def save(fig, name, cap):
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    caps.append(f"**{name}.** {cap}")
    print(f"[fig] {name}", flush=True)


def fig1():
    coords = pd.read_csv(rc.COORDS_PATH)
    mem = pd.read_csv(R.REV / "spatiotemporal_blocks_membership.csv")
    mem = mem[mem.crop == "bugday"][["ilce_id", "space_cluster"]].drop_duplicates()
    d = coords.merge(mem, on="ilce_id", how="left")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    sc = ax.scatter(d.lon, d.lat, c=d.space_cluster.fillna(-1), cmap="tab10", s=70,
                    edgecolor="black", linewidth=0.4)
    ax.scatter([27.8615], [41.5312], marker="*", s=420, color="gold", edgecolor="black",
               linewidth=0.8, label="0.62 ha parcel (Vize)", zorder=5)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Trakya study area — 29 districts (colour = spatial CV cluster) + parcel")
    ax.legend(loc="lower left", fontsize=8)
    ax2.axis("off")
    flow = ("DATA\n  • TÜİK district yields (2004–2025, 1165 ilçe-years)\n"
            "  • Climate: NASA POWER / MERRA-2 (season features)\n"
            "  • NDVI: Sentinel-2 L2A (ESA WorldCover cropland; 16-day median)\n"
            "  • Soil: ISRIC SoilGrids\n\n"
            "FEATURE TIERS\n  A climate (14) → B +NDVI (21) → C +soil (27)\n\n"
            "MODELS\n  PLS · ElasticNet · RF · XGBoost · GPR · Stacking\n"
            "  (fixed hyperparameters, seed 42)\n\n"
            "EVALUATION\n  • LOYO  (temporal, retrospective)\n"
            "  • LOILO (spatial)\n  • Spatiotemporal block interpolation\n"
            "  • Rolling-origin (forward-in-time) ← operational claims\n"
            "  • Baselines: climatology (matched) + persistence\n"
            "  • Cluster-aware CIs (year / district)")
    ax2.text(0.0, 1.0, flow, va="top", ha="left", fontsize=9, family="monospace")
    save(fig, "fig1_study_area_workflow",
         "Study area and modelling/evaluation workflow. Left: the 29 Trakya districts (centroids; "
         "colour = spatial cross-validation cluster) and the 0.62 ha validation parcel near Vize. "
         "Right: data sources, feature tiers, models, and cross-validation regimes.")


def fig2():
    ps = pd.read_csv(R.ANALYSIS / "per_sample_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    for i, crop in enumerate(("bugday", "aycicegi")):
        for j, cv in enumerate(("LOYO", "LOILO")):
            ax = axes[i, j]
            sub = ps[(ps.layer == "C") & (ps.crop == crop) & (ps.cv == cv)]
            r2s = {m: r2_score(g.y_true, g.y_pred) for m, g in sub.groupby("model")}
            bm = max(r2s, key=r2s.get)
            g = sub[sub.model == bm]
            yt, yp = g.y_true.values, g.y_pred.values
            ax.scatter(yt, yp, s=18, alpha=0.5, color=COL[crop], edgecolor="none")
            lo, hi = min(yt.min(), yp.min()) * .95, max(yt.max(), yp.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{CN[crop]} — {cv} ({'temporal' if cv=='LOYO' else 'spatial'})\n"
                         f"Tier C {bm}: $R^2$={r2_score(yt,yp):+.3f}, "
                         f"RMSE={np.sqrt(mean_squared_error(yt,yp)):.1f}, n={len(g)}")
            ax.set_xlabel("Observed yield (kg da$^{-1}$)")
            ax.set_ylabel("Predicted yield (kg da$^{-1}$)")
    fig.suptitle("Predicted vs observed — temporal (LOYO) and spatial (LOILO)", y=1.0, fontsize=12)
    save(fig, "fig2_pred_vs_actual_4panel",
         "Predicted versus observed district yields for the multimodal (tier C) best model under "
         "temporal (LOYO) and spatial (LOILO) cross-validation, per crop. The dashed line is 1:1.")


def fig3():
    t2 = pd.read_csv(R.RTAB / "table2_temporal_performance.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    tiers = ["A", "B", "C"]
    x = np.arange(len(tiers)); w = 0.38
    for k, crop in enumerate(("bugday", "aycicegi")):
        d = t2[t2.crop == crop].set_index("tier").reindex(tiers)
        ss = d.skill_score.values
        lo = ss - d.ss_ci_low_clustered.values if "ss_ci_low_clustered" in d else None
        # parse clustered CI string
        ci = d.ss_clustered_ci.str.strip("[]").str.split(",", expand=True).astype(float)
        err = np.vstack([ss - ci[0].values, ci[1].values - ss])
        ax.bar(x + (k - 0.5) * w, ss, w, yerr=err, capsize=3, color=COL[crop], alpha=0.9,
               label=CN[crop], error_kw=dict(lw=0.9))
    ax.axhline(0, color="black", lw=1.2, label="climatology baseline")
    ax.set_xticks(x); ax.set_xticklabels(["Tier A\n(climate)", "Tier B\n(+NDVI)", "Tier C\n(+soil)"])
    ax.set_ylabel("Skill score vs climatology  (1 − RMSE/RMSE$_{B0,matched}$)")
    ax.set_title("Temporal (LOYO) skill vs matched climatology — best model per tier\n"
                 "(error bars = 95% year-clustered bootstrap CI)")
    ax.legend(fontsize=8)
    save(fig, "fig3_skill_by_tier",
         "Leave-one-year-out skill score relative to the matched climatology baseline (SS = 1 − "
         "RMSE_model/RMSE_baseline,matched) for the best model in each feature tier. Error bars are "
         "95% year-clustered bootstrap intervals. Winter-wheat skill is negative at every tier; only "
         "sunflower tier C lies clearly above zero.")


def fig4():
    gap = pd.read_csv(R.ANALYSIS / "generalization_gap.csv")
    # 4a same-model gap
    fig, ax = plt.subplots(figsize=(8, 5))
    lab = [f"{CN[r.crop][:4]}·{r.layer}" for r in gap.itertuples()]
    x = np.arange(len(gap))
    ax.bar(x - 0.2, gap.r2_loyo_fixed, 0.4, label="LOYO (temporal)", color="#C44E52")
    ax.bar(x + 0.2, gap.r2_loilo_fixed, 0.4, label="LOILO (spatial)", color="#55A868")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=0, fontsize=8)
    ax.set_ylabel("$R^2$ (same model)"); ax.legend(fontsize=8)
    ax.set_title("Same-model spatial vs temporal generalization gap")
    save(fig, "fig4a_same_model_gap",
         "Same-model generalization gap: R² of the (fixed) LOILO-champion model under temporal "
         "(LOYO) versus spatial (LOILO) cross-validation, on identical observations. The spatial−"
         "temporal gap is large and significant for every crop and tier (see table3).")
    # 4b best-achievable per regime
    ps = pd.read_csv(R.ANALYSIS / "per_sample_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, crop in zip(axes, ("bugday", "aycicegi")):
        for j, cv in enumerate(("LOYO", "LOILO", "Spatiotemporal")):
            vals = []
            for t in ("A", "B", "C"):
                g = ps[(ps.layer == t) & (ps.crop == crop) & (ps.cv == cv)]
                vals.append(max(r2_score(gg.y_true, gg.y_pred) for _, gg in g.groupby("model")))
            ax.bar(np.arange(3) + (j - 1) * 0.26, vals, 0.26,
                   label=["LOYO", "LOILO", "Spatiotemp."][j],
                   color=["#C44E52", "#55A868", "#8172B3"][j], alpha=0.9)
        ax.axhline(0, color="black", lw=1); ax.set_xticks(range(3))
        ax.set_xticklabels(["A", "B", "C"]); ax.set_title(CN[crop]); ax.set_ylabel("best $R^2$")
    axes[1].legend(fontsize=8, title="regime")
    fig.suptitle("Best-achievable R² per regime (different model may win per cell)", y=1.0)
    save(fig, "fig4b_best_per_regime",
         "Best-achievable R² in each cross-validation regime and tier (the winning model may differ "
         "per cell). Shown separately from the same-model gap (fig4a) to avoid conflating the two.")


def fig5():
    pp = pd.read_csv(R.REV / "parcel_per_stage.csv")
    d = pp[pp.season == 2025]
    order = ["pre_season", "emergence", "vegetative", "flowering", "grain_fill", "maturity", "post_harvest"]
    d = d.set_index("stage").reindex(order).dropna(subset=["model_mae"])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.model_mae, 0.4, label="frozen model", color="#4C72B0")
    ax.bar(x + 0.2, d.persistence_mae, 0.4, label="persistence", color="#999999")
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=30, ha="right")
    ax.set_ylabel("MAE (NDVI units)")
    ax.set_title("Parcel per-stage NDVI t+7 error — model vs persistence (real parcel, 2025)\n"
                 "(MAE shown; R² unstable at low-variance stages — see table5)")
    ax.legend(fontsize=9)
    save(fig, "fig5_parcel_per_stage",
         "Per-phenological-stage NDVI t+7 forecast error (MAE) at the real parcel (2025), frozen "
         "model versus naïve persistence. Persistence is not outperformed at most stages; R²/RMSE/"
         "median errors are tabulated in table5 (R² is unstable where within-stage NDVI variance is "
         "small).")


def fig6():
    sens = pd.read_csv(R.REV / "morans_i_sensitivity.csv")
    coords = pd.read_csv(rc.COORDS_PATH)
    ps = pd.read_csv(R.ANALYSIS / "per_sample_predictions.csv")
    champ = {"bugday": "elastic_net", "aycicegi": "random_forest"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, crop in zip(axes[:2], ("bugday", "aycicegi")):
        g = ps[(ps.layer == "A") & (ps.crop == crop) & (ps.cv == "LOYO") & (ps.model == champ[crop])].copy()
        g["res"] = g.y_true - g.y_pred
        agg = g.groupby("ilce_id")["res"].mean().reset_index().merge(coords, on="ilce_id")
        I4 = sens[(sens.crop == crop) & (sens.k == 4)].iloc[0]
        vmax = np.abs(agg.res).max()
        sc = ax.scatter(agg.lon, agg.lat, c=agg.res, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        s=80, edgecolor="black", linewidth=0.4)
        plt.colorbar(sc, ax=ax, label="mean LOYO residual")
        ax.set_title(f"{CN[crop]} residuals\nglobal Moran's I (k=4) = {I4.morans_I:+.3f} "
                     f"(p$_{{norm}}$={I4.p_norm:.3f})")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax = axes[2]
    for crop in ("bugday", "aycicegi"):
        s = sens[sens.crop == crop]
        ax.plot(s.k, s.morans_I, "o-", color=COL[crop], label=CN[crop])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("KNN k"); ax.set_ylabel("global Moran's I"); ax.set_title("k-sensitivity")
    ax.set_xticks([3, 4, 5, 6]); ax.legend(fontsize=8)
    save(fig, "fig6_global_morans_i",
         "Global Moran's I of climate-tier LOYO residuals (district means; KNN row-standardised "
         "weights, 999 permutations). Maps for k=4 (left, centre) and sensitivity to k=3–6 (right). "
         "Wheat residuals show significant positive global spatial autocorrelation across all k; "
         "sunflower residuals do not.")


def fig7():
    imp = pd.read_csv(R.REV / "permutation_importance_foldwise.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, tier in enumerate(("A", "C")):
        for j, crop in enumerate(("bugday", "aycicegi")):
            ax = axes[i, j]
            d = imp[(imp.tier == tier) & (imp.crop == crop)].sort_values("imp_mean").tail(10)
            y = np.arange(len(d))
            err = np.vstack([d.imp_mean - d.imp_ci_low, d.imp_ci_high - d.imp_mean])
            ax.barh(y, d.imp_mean, xerr=err, color=COL[crop], alpha=0.9, error_kw=dict(lw=0.8, capsize=2))
            ax.set_yticks(y); ax.set_yticklabels(d.feature, fontsize=8)
            ax.set_xlabel("increase in RMSE (kg da$^{-1}$)")
            ax.set_title(f"{CN[crop]} — Tier {tier}")
    fig.suptitle("Fold-wise permutation importance (test-fold; mean ± 95% CI)", y=1.0, fontsize=12)
    save(fig, "fig7_foldwise_importance",
         "Fold-wise permutation importance (computed on held-out folds; metric = increase in RMSE), "
         "tiers A and C, both crops, mean ± 95% bootstrap CI across folds. Importance reflects "
         "predictive contribution consistent with agronomic expectations, not causation.")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    (R.REV / "figure_captions_EN.md").write_text(
        "# Figure captions (EN) — referee-revised\n\n" + "\n\n".join(caps) + "\n", encoding="utf-8")
    print("[save] figure_captions_EN.md", flush=True)


if __name__ == "__main__":
    main()
