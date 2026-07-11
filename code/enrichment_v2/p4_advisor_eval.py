"""P4 — CV evaluation on advisor-aligned tiers (crop-specific RS). Mirrors T8; copied harness.

LOYO/LOILO/Spatiotemporal × tiers A–D × models; matched climatology SS (LOYO) + year-clustered CI;
spatial-temporal gap ΔR² + year-block bootstrap + year signed-rank; ablation vs A; rolling-origin.
Outputs: advisor_master_ledger.csv, advisor_gap.csv, advisor_ablation.csv, advisor_rolling.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E
import harness as H
import harness_clusters as HC

MODELS = ["pls", "elastic_net", "random_forest", "xgboost", "gpr"]


def b0_loo(df):
    pred = np.full(len(df), np.nan)
    for _, idxs in df.groupby("ilce_id").groups.items():
        idxs = list(idxs); vals = df.loc[idxs, "verim_kg_da"].astype(float); tot, n = vals.sum(), len(vals)
        for i in idxs:
            pred[df.index.get_loc(i)] = (tot - df.loc[i, "verim_kg_da"]) / (n - 1) if n > 1 else vals.mean()
    return pred


def rolling(df, feats, min_train=4):
    sub = df.sort_values("year").reset_index(drop=True); yrs = sorted(sub.year.unique())
    ty = [y for y in yrs if y - yrs[0] >= min_train]; out = []
    for Y in ty:
        tr, te = sub[sub.year < Y], sub[sub.year == Y]
        if len(tr) < 20 or te.empty:
            continue
        Xtr = H._impute(tr[feats].astype(float)); med = Xtr.median()
        Xte = te[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        ytr, yte = tr.verim_kg_da.values, te.verim_kg_da.values
        clim = te.ilce_id.map(tr.groupby("ilce_id").verim_kg_da.mean()).fillna(ytr.mean()).values
        crmse = np.sqrt(mean_squared_error(yte, clim)); best = -9
        for m in MODELS:
            mm = H._make_model(m)
            if m in H._NEEDS_SCALER:
                sc = StandardScaler().fit(Xtr); a, b = sc.transform(Xtr), sc.transform(Xte)
            else:
                a, b = Xtr.values, Xte.values
            pred = mm.fit(a, ytr).predict(b).ravel()
            best = max(best, 1 - np.sqrt(mean_squared_error(yte, pred)) / crmse if crmse > 0 else np.nan)
        out.append(best)
    return (float(np.nanmean(out)) if out else np.nan), len(out)


def main():
    sel = json.loads((E.OUT / "advisor_selected_features.json").read_text(encoding="utf-8"))
    led, gap, roll = [], [], []
    cache = {}
    for crop in ("bugday", "aycicegi"):
        for tier in ("A", "B", "C", "D"):
            df = pd.read_csv(E.OUT / f"advisor_tier_{tier}_{crop}.csv").reset_index(drop=True)
            feats = sel[crop][tier]
            X = H._impute(df[feats].astype(float)); y = df.verim_kg_da.astype(float).values
            grp = {"LOYO": df.year.astype(int).values, "LOILO": df.ilce_id.astype(int).values,
                   "Spatiotemporal": H._block_groups(df)}
            b0 = b0_loo(df); rmse_b0 = float(np.sqrt(mean_squared_error(y, b0)))
            for m in MODELS:
                for cv in ("LOYO", "LOILO", "Spatiotemporal"):
                    p = H._cv_predict(m, X, y, grp[cv]); cache[(crop, tier, m, cv)] = p
                    row = dict(crop=crop, tier=tier, model=m, cv=cv, n=len(y),
                               r2=round(float(r2_score(y, p)), 4),
                               rmse=round(float(np.sqrt(mean_squared_error(y, p))), 3))
                    if cv == "LOYO":
                        ss = 1 - row["rmse"] / rmse_b0 if rmse_b0 > 0 else np.nan
                        lo, hi = HC.ci95(HC.boot_ss_cluster(y, p, b0, df.year.astype(int).values))
                        row.update(baseline_rmse_matched=round(rmse_b0, 3), skill_score=round(ss, 4),
                                   ss_ci_low_clustered=round(lo, 4), ss_ci_high_clustered=round(hi, 4))
                    led.append(row)
                pl, pi = cache[(crop, tier, m, "LOYO")], cache[(crop, tier, m, "LOILO")]
                dv = HC.rc.boot_paired_dr2(y, pl, pi, HC.N_BOOT, HC.SEED); glo, ghi = HC.ci95(dv)
                dd = pd.DataFrame({"year": df.year.values, "ael": np.abs(y - pl), "aei": np.abs(y - pi)}).groupby("year").mean()
                try:
                    _, pp = wilcoxon(dd.ael, dd.aei); pp = float(pp)
                except Exception:
                    pp = np.nan
                gap.append(dict(crop=crop, tier=tier, model=m, r2_LOYO=round(float(r2_score(y, pl)), 4),
                                r2_LOILO=round(float(r2_score(y, pi)), 4),
                                gap_dR2=round(float(r2_score(y, pi) - r2_score(y, pl)), 4),
                                gap_ci_low=round(glo, 4), gap_ci_high=round(ghi, 4),
                                year_signrank_p=round(pp, 4) if pp == pp else np.nan))
            ms, ny = rolling(df, feats); roll.append(dict(crop=crop, tier=tier, n_test_years=ny,
                                                          rolling_mean_skill_vs_clim=round(ms, 4) if ms == ms else np.nan))
            print(f"[p4] {crop} tier {tier} done", flush=True)
    L = pd.DataFrame(led); L.to_csv(E.OUT / "advisor_master_ledger.csv", index=False)
    pd.DataFrame(gap).to_csv(E.OUT / "advisor_gap.csv", index=False)
    pd.DataFrame(roll).to_csv(E.OUT / "advisor_rolling.csv", index=False)
    base = L[L.tier == "A"].set_index(["crop", "model", "cv"]).r2
    abl = []
    for _, r in L[L.tier != "A"].iterrows():
        try:
            a = float(base.loc[(r.crop, r.model, r.cv)]); abl.append(dict(crop=r.crop, tier=r.tier, model=r.model, cv=r.cv, delta_r2_vs_A=round(r.r2 - a, 4)))
        except KeyError:
            pass
    pd.DataFrame(abl).to_csv(E.OUT / "advisor_ablation.csv", index=False)
    print("\n=== ADVISOR tiers: LOYO skill vs matched climatology (best per tier) ===", flush=True)
    loyo = L[L.cv == "LOYO"].dropna(subset=["skill_score"])
    for crop in ("bugday", "aycicegi"):
        for t in ("A", "B", "C", "D"):
            s = loyo[(loyo.crop == crop) & (loyo.tier == t)]
            if s.empty: continue
            b = s.loc[s.skill_score.idxmax()]
            print(f"  {crop} {t}: best={b.model} SS={b.skill_score:+.3f} clustCI[{b.ss_ci_low_clustered:+.3f},{b.ss_ci_high_clustered:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
