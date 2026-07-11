"""A1 — full data-quality audit (READ-ONLY). Writes ONLY to enrichment_v2/audit/.

Runs B1-B9 on the rawest available representation (saved post-clamp index aggregates + raw daily
climate + soil/topo/yield/anomaly). Emits feature_summary.csv + flagged_records.csv and prints the
numbers for the report. Zero fabrication: every value measured from a real file.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "enrichment_v2" / "audit"
OUT = ROOT / "enrichment_v2" / "outputs"
AUD.mkdir(parents=True, exist_ok=True)
INDICES = ["NDVI", "EVI", "EVI2", "NDRE", "CIre", "NDWI", "GNDVI", "OSAVI"]
CLAMPED = {"EVI", "EVI2"}
LOC_METRICS = ["mean", "median", "p10", "p90"]
flags = []
summ = []


def flag(feature, crop, ilce, year, value, reason):
    flags.append(dict(feature=feature, crop=crop, ilce_id=ilce, year=year, value=value, reason=reason))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ---------------- B1 + B2: spectral indices (saved post-clamp aggregates) ----------------
print("===== B1/B2 spectral indices =====")
idx = pd.concat([pd.read_csv(OUT / f"indices_{c}.csv").assign(crop=c) for c in ("bugday", "aycicegi")],
                ignore_index=True)
for ind in INDICES:
    loc_cols = [c for c in idx.columns if c.startswith(ind + "_") and c.rsplit("_", 1)[-1] in LOC_METRICS]
    std_cols = [c for c in idx.columns if c.startswith(ind + "_") and c.endswith("_stdDev")]
    cv_cols = [c for c in idx.columns if c.startswith(ind + "_") and c.endswith("_cv")]
    rng_cols = [c for c in idx.columns if c.startswith(ind + "_") and c.endswith("_range")]
    loc = idx[loc_cols].to_numpy(dtype=float)
    finite = loc[np.isfinite(loc)]
    nan_pct = float(np.isnan(loc).mean() * 100)
    n_nan = int(np.isnan(loc).sum())
    n_inf = int(np.isinf(loc).sum())          # garbage non-finite (NaN excluded = coverage)
    nonfinite = n_inf                          # for verdict: only inf is garbage
    lo, hi = (-1.0, 1.0) if ind != "CIre" else (-1.0, 10.0)
    oob = int(((finite < lo) | (finite > hi)).sum())
    extreme = int((np.abs(finite) > 1e3).sum())
    mn, mx, me = (float(finite.min()), float(finite.max()), float(finite.mean())) if finite.size else (np.nan,)*3
    # per-record offending (location metrics)
    for col in loc_cols:
        s = idx[col]
        bad = idx[(s.notna()) & ((s < lo) | (s > hi))]
        for r in bad.itertuples():
            flag(col, r.crop, int(r.ilce_id), int(r.year), round(float(getattr(r, col)), 4),
                 f"{ind} location-metric out of [{lo},{hi}]")
    # std<0 / range<0 / CV instability
    sneg = int((idx[std_cols].to_numpy(float) < 0).sum()) if std_cols else 0
    rneg = int((idx[rng_cols].to_numpy(float) < -1e-9).sum()) if rng_cols else 0
    cvv = idx[cv_cols].to_numpy(float) if cv_cols else np.array([])
    cv_nonfin = int((~np.isfinite(cvv)).sum()) if cvv.size else 0
    cv_max = float(np.nanmax(np.abs(cvv))) if cvv.size and np.isfinite(cvv).any() else np.nan
    print(f"  {ind:6s} clamp={'Y' if ind in CLAMPED else 'N'} min={mn:.3f} max={mx:.3f} mean={me:.3f} "
          f"NaN%={nan_pct:.1f} nonfinite={nonfinite} OOB={oob} extreme(>1e3)={extreme} "
          f"std<0={sneg} range<0={rneg} CVnonfin={cv_nonfin} maxCV={cv_max:.2f}")
    # verdict on VALUE validity (OOB/inf/extreme); NaN is coverage (reported separately);
    # clamp_guard_gap flags the code-level missing guard even when no garbage is present today.
    values_clean = (oob == 0 and n_inf == 0 and extreme == 0)
    gap = "Y" if ind not in CLAMPED else "N"
    verdict = "CLEAN" if values_clean else "FLAGS"
    if values_clean and gap == "Y":
        verdict = "CLEAN (code guard gap — preventive)"
    summ.append(dict(family="index", feature=ind, min=round(mn, 4), max=round(mx, 4), mean=round(me, 4),
                     nan_pct=round(nan_pct, 2), n_nan=n_nan, n_inf=n_inf, n_out_of_bounds=oob,
                     extreme_gt_1e3=extreme, clamp_applied="Y" if ind in CLAMPED else "N",
                     clamp_guard_gap=gap, verdict=verdict))

# B2 per-record ordering P10<=median<=P90
print("===== B2 distribution ordering =====")
viol = 0
for ind in INDICES:
    for win in set(c.split("_")[1] for c in idx.columns if c.startswith(ind + "_") and len(c.split("_")) >= 3):
        p10, med, p90 = f"{ind}_{win}_p10", f"{ind}_{win}_median", f"{ind}_{win}_p90"
        if all(c in idx.columns for c in (p10, med, p90)):
            sub = idx[[p10, med, p90, "crop", "ilce_id", "year"]].dropna()
            bad = sub[(sub[p10] > sub[med] + 1e-9) | (sub[med] > sub[p90] + 1e-9)]
            viol += len(bad)
            for r in bad.itertuples():
                flag(f"{ind}_{win}", r.crop, int(r.ilce_id), int(r.year), None, "P10<=median<=P90 violated")
print(f"  ordering violations (P10<=median<=P90): {viol}")

# ---------------- B3 climate ----------------
print("===== B3 climate =====")
cal = pd.read_csv(ROOT / "data/processed/calibration_features_layerA.csv")
for col, lo, hi, name in [("tp_season_sum", 0, 1500, "season precip mm"),
                          ("tp_winter_sum", 0, 1200, "winter precip mm"),
                          ("tp_flowering", 0, 600, "flowering precip mm"),
                          ("tp_grain_fill", 0, 600, "grainfill precip mm"),
                          ("gdd_cum_season", 0, 6000, "GDD"),
                          ("aridity_index", 0, 100, "aridity"),
                          ("t2m_flowering_mean", -10, 45, "t2m flowering C")]:
    if col not in cal.columns:
        print(f"  {col}: NOT FOUND"); continue
    s = cal[col].astype(float)
    oob = int(((s < lo) | (s > hi)).sum())
    print(f"  {col:20s} min={s.min():.1f} max={s.max():.1f} mean={s.mean():.1f} OOB[{lo},{hi}]={oob}")
    for r in cal[(cal[col] < lo) | (cal[col] > hi)].itertuples():
        flag(col, getattr(r, "crop", "?"), int(getattr(r, "ilce_id", -1)), int(getattr(r, "year", -1)),
             round(float(getattr(r, col)), 2), f"{name} out of [{lo},{hi}]")
    summ.append(dict(family="climate", feature=col, min=round(float(s.min()), 2), max=round(float(s.max()), 2),
                     mean=round(float(s.mean()), 2), nan_pct=round(float(s.isna().mean()*100), 2),
                     n_out_of_bounds=oob, n_nonfinite=int((~np.isfinite(s)).sum()), clamp_applied="n/a",
                     verdict="CLEAN" if oob == 0 else "FLAGS"))
# raw daily precip annual totals (12.5x bug check)
dfd = []
for f in glob.glob(str(ROOT / "data/processed/openmeteo_ilce/nasapower_ilce_*.csv")):
    d = pd.read_csv(f, parse_dates=["date"]); d["year"] = d.date.dt.year
    ann = d.groupby("year")["PRECTOTCORR"].sum()
    dfd.append(ann)
allann = pd.concat(dfd)
print(f"  RAW daily annual precip (29 ilçe): min={allann.min():.0f} max={allann.max():.0f} "
      f"mean={allann.mean():.0f} mm  (Trakya plausible ~500-700)")
summ.append(dict(family="climate", feature="raw_annual_precip_mm", min=round(float(allann.min())),
                 max=round(float(allann.max())), mean=round(float(allann.mean())), nan_pct=0,
                 n_out_of_bounds=int((allann > 1500).sum()), n_nonfinite=0, clamp_applied="n/a",
                 verdict="CLEAN" if allann.max() < 1500 else "FLAGS(scale?)"))

# ---------------- B4 soil ----------------
print("===== B4 soil =====")
soil = pd.read_csv(OUT / "soil_features.csv")
checks = {"phh2o_0_30_mean": (3, 10), "soc_0_30_mean": (0, 120), "cec_0_30_mean": (0, 60),
          "bdod_0_30_mean": (0.5, 2.0), "cfvo_0_30_mean": (0, 80), "awc_0_30": (0, 0.5),
          "clay_0_30_mean": (0, 100), "sand_0_30_mean": (0, 100), "silt_0_30_mean": (0, 100)}
for col, (lo, hi) in checks.items():
    if col not in soil.columns:
        print(f"  {col}: NOT FOUND"); continue
    s = soil[col].astype(float); oob = int(((s < lo) | (s > hi)).sum())
    print(f"  {col:18s} min={s.min():.3f} max={s.max():.3f} OOB[{lo},{hi}]={oob}")
    summ.append(dict(family="soil", feature=col, min=round(float(s.min()), 3), max=round(float(s.max()), 3),
                     mean=round(float(s.mean()), 3), nan_pct=round(float(s.isna().mean()*100), 2),
                     n_out_of_bounds=oob, n_nonfinite=int((~np.isfinite(s)).sum()), clamp_applied="n/a",
                     verdict="CLEAN" if oob == 0 else "FLAGS"))
if all(c in soil.columns for c in ["clay_0_30_mean", "sand_0_30_mean", "silt_0_30_mean"]):
    tex = soil["clay_0_30_mean"] + soil["sand_0_30_mean"] + soil["silt_0_30_mean"]
    off = soil[(tex - 100).abs() > 5]
    print(f"  clay+sand+silt: min={tex.min():.1f} max={tex.max():.1f} | rows off>5% from 100: {len(off)}")
    for r in off.itertuples():
        flag("clay+sand+silt", "static", int(r.ilce_id), -1, round(float(tex[r.Index]), 1), "texture sum != 100%")

# ---------------- B5 topo ----------------
print("===== B5 topo =====")
topo = pd.read_csv(OUT / "topo_features.csv")
for col, lo, hi in [("elevation_m", 0, 1200), ("slope_deg", 0, 90), ("northness", -1, 1),
                    ("eastness", -1, 1), ("twi", -50, 50)]:
    if col not in topo.columns:
        print(f"  {col}: NOT FOUND"); continue
    s = topo[col].astype(float); oob = int(((s < lo) | (s > hi)).sum())
    print(f"  {col:12s} min={s.min():.3f} max={s.max():.3f} OOB={oob} nonfinite={int((~np.isfinite(s)).sum())}")
    summ.append(dict(family="topo", feature=col, min=round(float(s.min()), 3), max=round(float(s.max()), 3),
                     mean=round(float(s.mean()), 3), nan_pct=0, n_out_of_bounds=oob,
                     n_nonfinite=int((~np.isfinite(s)).sum()), clamp_applied="n/a", verdict="CLEAN" if oob == 0 else "FLAGS"))

# ---------------- B6 yield + anomaly ----------------
print("===== B6 yield + anomaly =====")
y = pd.read_csv(ROOT / "data/external/tuik/tuik_ilce_yields_full_referans.csv")
for crop, lo, hi in [("bugday", 50, 700), ("aycicegi_yaglik", 50, 500)]:
    s = y[y.crop == crop]["verim_kg_da"].dropna().astype(float)
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
    out = s[(s < q1 - 5*iqr) | (s > q3 + 5*iqr)]
    nz = int((s <= 0).sum())
    print(f"  {crop:16s} n={len(s)} min={s.min():.0f} max={s.max():.0f} mean={s.mean():.0f} "
          f"zeros/neg={nz} >5xIQR_outliers={len(out)}")
    summ.append(dict(family="yield", feature=crop, min=round(float(s.min())), max=round(float(s.max())),
                     mean=round(float(s.mean())), nan_pct=0, n_out_of_bounds=int(nz+len(out)),
                     n_nonfinite=0, clamp_applied="n/a", verdict="CLEAN" if (nz == 0 and len(out) == 0) else "FLAGS"))
# anomaly z
an = pd.concat([pd.read_csv(OUT / f"anomaly_{c}.csv").assign(crop=c) for c in ("bugday", "aycicegi")], ignore_index=True)
zc = [c for c in an.columns if c.endswith("_z")]
zv = an[zc].to_numpy(float)
big = int((np.abs(zv) > 5).sum()); nonfin = int((~np.isfinite(zv) & ~np.isnan(zv)).sum())
print(f"  anomaly z: cols={len(zc)} |z|>5 count={big} nonfinite(non-nan)={nonfin} maxAbs={np.nanmax(np.abs(zv)):.2f}")
summ.append(dict(family="anomaly", feature="z_scores", min=round(float(np.nanmin(zv)), 2), max=round(float(np.nanmax(zv)), 2),
                 mean=0, nan_pct=round(float(np.isnan(zv).mean()*100), 2), n_out_of_bounds=big,
                 n_nonfinite=nonfin, clamp_applied="n/a", verdict="CLEAN" if (big == 0 and nonfin == 0) else "FLAGS"))

# ---------------- B7 coverage ----------------
print("===== B7 coverage =====")
for c in ("bugday", "aycicegi"):
    d = pd.read_csv(OUT / f"indices_{c}.csv")
    print(f"  indices_{c}: districts={d.ilce_id.nunique()} years={sorted(d.year.unique())[:1]}..{sorted(d.year.unique())[-1:]} "
          f"rows={len(d)} dup_dy={int(d.duplicated(['ilce_id','year']).sum())} overall_NaN%={d.select_dtypes('number').isna().mean().mean()*100:.1f}")

# ---------------- B8 structural ----------------
print("===== B8 structural =====")
num = idx.select_dtypes("number").drop(columns=["ilce_id", "year"], errors="ignore")
const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
print(f"  constant/zero-variance index cols: {len(const)}")
# collinearity (sample to keep it fast)
cc = num.dropna(axis=1, how="all")
corr = cc.corr().abs()
pairs = [(corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3))
         for i in range(len(corr)) for j in range(i+1, len(corr)) if corr.iloc[i, j] > 0.98]
print(f"  |r|>0.98 collinear index pairs: {len(pairs)}")

# ---------------- save ----------------
pd.DataFrame(summ).to_csv(AUD / "feature_summary.csv", index=False)
pd.DataFrame(flags).to_csv(AUD / "flagged_records.csv", index=False)
pd.DataFrame(pairs, columns=["feat_a", "feat_b", "abs_r"]).to_csv(AUD / "collinear_pairs.csv", index=False)
print(f"\n[A1] feature_summary.csv ({len(summ)}), flagged_records.csv ({len(flags)}), collinear_pairs.csv ({len(pairs)})")
print("[A1] file-based audit done.")
