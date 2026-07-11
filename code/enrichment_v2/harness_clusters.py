"""rev_common — shared helpers for the referee-revision pipeline.

Imports the fidelity-verified repro_common (cp25 CV pipeline) and adds:
* fold-id derivation per CV regime (+ spatiotemporal block map)
* cluster-aware bootstrap for skill score and MAE-difference (year / ilce / block)
All deterministic (SEED reused). No fabrication; reads only existing artifacts/inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# locate repro_common (paper1_generalization/repro)
_PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as rc  # noqa: E402

REV = _PAPER / "enrichment_v2" / "outputs"
RTAB = REV / "tables"
RFIG = REV / "figures"
for d in (REV, RTAB, RFIG):
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS = _PAPER / "enrichment_v2" / "outputs"
SEED = 12345
N_BOOT = 5000


def block_map(tier: str, crop: str) -> dict:
    """(ilce_id, year) -> spatiotemporal block id, from repro_common._block_groups."""
    sub = rc.crop_subset(rc.load_layer(tier), crop)
    blocks = rc._block_groups(sub)
    return {(int(r.ilce_id), int(r.year)): int(b)
            for r, b in zip(sub.itertuples(index=False), blocks)}


def add_fold_id(df: pd.DataFrame, tier: str, crop: str) -> pd.DataFrame:
    """df has columns ilce_id, year, cv. Returns df with fold_id."""
    bm = block_map(tier, crop)
    out = df.copy()
    fid = []
    for r in out.itertuples(index=False):
        if r.cv == "LOYO":
            fid.append(f"year={int(r.year)}")
        elif r.cv == "LOILO":
            fid.append(f"ilce={int(r.ilce_id)}")
        else:
            fid.append(f"block={bm.get((int(r.ilce_id), int(r.year)), -1)}")
    out["fold_id"] = fid
    return out


def cluster_labels(df_sub: pd.DataFrame, cv: str, tier: str, crop: str) -> np.ndarray:
    """Per-observation cluster label for clustered bootstrap, given a per_sample slice."""
    if cv == "LOYO":
        return df_sub["year"].astype(int).values
    if cv == "LOILO":
        return df_sub["ilce_id"].astype(int).values
    bm = block_map(tier, crop)
    return np.array([bm.get((int(i), int(y)), -1)
                     for i, y in zip(df_sub["ilce_id"], df_sub["year"])])


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def skill_score(yt, yp_model, yp_base):
    rb = _rmse(yt, yp_base)
    return 1.0 - _rmse(yt, yp_model) / rb if rb > 0 else np.nan


def boot_ss_iid(yt, yp_model, yp_base, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(yt)
    idx = rng.integers(0, n, size=(n_boot, n))
    YT = yt[idx]
    rm = np.sqrt(((yp_model[idx] - YT) ** 2).mean(1))
    rb = np.sqrt(((yp_base[idx] - YT) ** 2).mean(1))
    return 1.0 - rm / rb


def boot_ss_cluster(yt, yp_model, yp_base, clusters, n_boot=N_BOOT, seed=SEED):
    """Cluster (group) bootstrap of skill score: resample whole clusters."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    groups = {c: np.where(clusters == c)[0] for c in uniq}
    k = len(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=k, replace=True)
        idx = np.concatenate([groups[c] for c in chosen])
        rb = _rmse(yt[idx], yp_base[idx])
        out[b] = 1.0 - _rmse(yt[idx], yp_model[idx]) / rb if rb > 0 else np.nan
    return out


def ci95(a):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def cluster_mae_diff_test(df_sub, yp_model_col, yp_base_col, cluster_col):
    """Per-cluster mean |error| difference (model - baseline); Wilcoxon signed-rank
    on cluster-level diffs + cluster block-bootstrap CI of the mean diff.
    Returns dict. Negative diff => model better than baseline."""
    from scipy.stats import wilcoxon
    g = df_sub.copy()
    g["ae_model"] = (g["y_true"] - g[yp_model_col]).abs()
    g["ae_base"] = (g["y_true"] - g[yp_base_col]).abs()
    per = g.groupby(cluster_col).agg(ae_model=("ae_model", "mean"),
                                     ae_base=("ae_base", "mean"))
    d = (per["ae_model"] - per["ae_base"]).values
    n_clusters = len(d)
    point = float(np.mean(d))
    # block bootstrap over clusters
    rng = np.random.default_rng(SEED)
    bs = np.array([np.mean(rng.choice(d, size=n_clusters, replace=True))
                   for _ in range(N_BOOT)])
    lo, hi = ci95(bs)
    nz = d[d != 0]
    if len(nz) >= 1 and not np.allclose(nz, 0):
        try:
            W, p = wilcoxon(per["ae_model"].values, per["ae_base"].values,
                            alternative="two-sided")
            W, p = float(W), float(p)
        except Exception:
            W, p = np.nan, np.nan
    else:
        W, p = np.nan, np.nan
    return dict(n_clusters=int(n_clusters), mean_mae_diff=round(point, 4),
                ci_low=round(lo, 4), ci_high=round(hi, 4),
                wilcoxon_W=W, p_value=p)


__all__ = ["rc", "REV", "RTAB", "RFIG", "ANALYSIS", "SEED", "N_BOOT",
           "block_map", "add_fold_id", "cluster_labels", "skill_score",
           "boot_ss_iid", "boot_ss_cluster", "ci95", "cluster_mae_diff_test", "_rmse"]
