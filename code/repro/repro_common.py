"""Paper 1 — sadık reprodüksiyon çekirdeği (faithful reproduction core).

Bu modül, tezin cp25 pipeline'ındaki (src/cp25/05,06,07) Layer A/B/C model
yarışını BİREBİR aynı konfigürasyonla (özellik listeleri, hiperparametreler,
SEED=42, imputasyon, CV mantığı) yeniden kurar. AMAÇ: orijinal scriptlerin
diske kaydetmeyip attığı LOILO/Spatiotemporal per-sample (out-of-fold)
tahminlerini elde etmek.

KURALLAR (bkz. logs/decisions.md D-04, D-05):
* Hiçbir orijinal dosyaya YAZILMAZ; girdiler salt-okunur.
* Yeniden EĞİTİM (retrain) değil — yayınlanmış DEĞERLENDİRMENİN birebir
  deterministik tekrarı. Sadakat, recompute edilen aggregate metriklerin
  yayınlanan tablolarla 3 ondalık eşleşmesiyle KANITLANIR (fidelity gate).
* Frozen prospektif LSTM+XGB modeline DOKUNULMAZ (o ayrı bir sözleşme).

Kaynak doğrulaması: feature listeleri ve model konfigürasyonları
src/cp25/05_layer_a_climate_only.py, 06_layer_b_climate_ndvi.py,
07_layer_c_full.py dosyalarından satır satır teyit edilmiştir.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (RandomForestRegressor, StackingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
COORDS_PATH  = PROJECT_ROOT / "data" / "external" / "tuik" / "ilce_coords.csv"
PAPER_DIR    = PROJECT_ROOT / "paper1_generalization"
ANALYSIS_DIR = PAPER_DIR / "analysis"
TABLES_DIR   = PAPER_DIR / "tables"
FIGURES_DIR  = PAPER_DIR / "figures"
for d in (ANALYSIS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Feature tiers (src/cp25/07 ile birebir) -------------------------------
FEATURES_A = [
    "gdd_cum_season", "gdd_flowering", "vernalization_days",
    "tp_season_sum", "tp_winter_sum", "tp_flowering", "tp_grain_fill",
    "aridity_index", "heat_stress_days",
    "t2m_flowering_mean", "t2m_flowering_max", "tdiff_mean",
    "ssr_flowering_sum", "ssr_season_sum",
]
FEATURES_NDVI = ["ndvi_max", "ndvi_mean_season", "ndvi_integral",
                 "ndvi_flowering", "ndvi_grain_fill", "ndvi_spring_slope",
                 "greenness_days"]
FEATURES_SOIL = ["clay_0-5cm", "sand_0-5cm", "silt_0-5cm",
                 "phh2o_0-5cm", "soc_0-5cm", "awc_0-5cm"]
FEATURES = {
    "A": FEATURES_A,
    "B": FEATURES_A + FEATURES_NDVI,
    "C": FEATURES_A + FEATURES_NDVI + FEATURES_SOIL,
}
LAYER_INPUT = {
    "A": "calibration_features_layerA.csv",
    "B": "calibration_features_layerB.csv",
    "C": "calibration_features_layerC.csv",
}
MODELS_BY_LAYER = {
    "A": ["pls", "elastic_net", "random_forest", "xgboost", "gpr"],
    "B": ["pls", "elastic_net", "random_forest", "xgboost", "gpr"],
    "C": ["pls", "elastic_net", "random_forest", "xgboost", "gpr", "stacking"],
}
_NEEDS_SCALER = {"pls", "elastic_net", "gpr", "stacking"}
CROPS = {"bugday": "bugday", "aycicegi": "aycicegi_yaglik"}
CV_SCHEMES = ["LOYO", "LOILO", "Spatiotemporal"]


# --- Replicated helpers (src/cp25/06,07 ile birebir) -----------------------
def _impute(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(0.0 if np.isnan(med) else med)
    return X


def _make_model(name: str):
    if name == "pls":
        return PLSRegression(n_components=3, scale=True)
    if name == "elastic_net":
        return ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=SEED)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_depth=5,
                                     random_state=SEED, n_jobs=-1)
    if name == "xgboost":
        return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=SEED, n_jobs=-1, verbosity=0)
    if name == "gpr":
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        random_state=SEED, alpha=1e-4)
    if name == "stacking":
        return StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=300, max_depth=5,
                                             random_state=SEED, n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=200, max_depth=4,
                                     learning_rate=0.05, random_state=SEED,
                                     n_jobs=-1, verbosity=0)),
                ("gpr", GaussianProcessRegressor(
                    kernel=Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0),
                    normalize_y=True, random_state=SEED, alpha=1e-4)),
            ],
            final_estimator=Ridge(alpha=1.0, random_state=SEED),
            n_jobs=1, cv=3, passthrough=False)
    raise ValueError(name)


def _cv_predict(model_name: str, X: pd.DataFrame, y: np.ndarray,
                groups: np.ndarray) -> np.ndarray:
    """Out-of-fold LeaveOneGroupOut tahminleri (orijinal ile birebir)."""
    preds = np.zeros_like(y, dtype=float)
    used = np.zeros_like(y, dtype=bool)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y[tr]
        if model_name in _NEEDS_SCALER:
            sc = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        else:
            X_tr_s, X_te_s = X_tr.values, X_te.values
        m = _make_model(model_name)
        m.fit(X_tr_s, y_tr)
        p = m.predict(X_te_s)
        if p.ndim > 1:
            p = p.ravel()
        preds[te] = p
        used[te] = True
    if not used.all():
        preds[~used] = float(np.mean(y))
    return preds


def _block_groups(df: pd.DataFrame, n_year_blocks: int = 5,
                  n_clusters: int = 5) -> np.ndarray:
    years = df["year"].astype(int).values
    yr_bins = np.linspace(years.min(), years.max() + 1, n_year_blocks + 1)
    yr_block = np.digitize(years, yr_bins[1:-1])
    coords = pd.read_csv(COORDS_PATH)[["ilce_id", "lat", "lon"]]
    df_loc = df.merge(coords, on="ilce_id", how="left")
    n_eff = min(n_clusters, df_loc["ilce_id"].nunique())
    km = KMeans(n_clusters=n_eff, random_state=SEED, n_init=10)
    sp = km.fit_predict(df_loc[["lat", "lon"]].values)
    return yr_block * n_eff + sp


def _b0_climatology_rmse(df: pd.DataFrame) -> float:
    preds = []
    for (_il, _crop), grp in df.groupby(["ilce_id", "crop"]):
        for idx, row in grp.iterrows():
            if (grp.index != idx).sum() == 0:
                continue
            preds.append((row["verim_kg_da"],
                          grp.loc[grp.index != idx, "verim_kg_da"].mean()))
    if not preds:
        return 1.0
    y_t, y_p = zip(*preds)
    return float(np.sqrt(mean_squared_error(np.array(y_t), np.array(y_p))))


def metrics(y_true: np.ndarray, y_pred: np.ndarray, rmse_b0: float | None = None) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_kg_da": rmse,
        "mae_kg_da": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "bias_kg_da": float(np.mean(y_pred - y_true)),
    }
    if rmse_b0 is not None and rmse_b0 > 0:
        out["ss_vs_b0"] = float(1.0 - rmse / rmse_b0)
    return out


def load_layer(layer: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / LAYER_INPUT[layer])


def crop_subset(df: pd.DataFrame, crop_short: str) -> pd.DataFrame:
    return df[df["crop"] == CROPS[crop_short]].copy().reset_index(drop=True)


def generate_per_sample(layer: str, crop_short: str) -> pd.DataFrame:
    """Bir layer × crop için TÜM model × CV out-of-fold tahminlerini üretir.

    Dönen long-format DataFrame kolonları:
    layer, crop, cv, model, ilce_id, ilce, il, year, y_true, y_pred, abs_error
    """
    df = load_layer(layer)
    sub = crop_subset(df, crop_short)
    feats = FEATURES[layer]
    X = _impute(sub[feats].astype(float))
    y = sub["verim_kg_da"].astype(float).values
    meta = sub[["ilce_id", "ilce", "il", "year"]].reset_index(drop=True)

    groups = {
        "LOYO": sub["year"].astype(int).values,
        "LOILO": sub["ilce_id"].astype(int).values,
        "Spatiotemporal": _block_groups(sub),
    }
    rows = []
    for cv in CV_SCHEMES:
        for model in MODELS_BY_LAYER[layer]:
            if model == "stacking" and cv != "LOYO":
                continue  # orijinalde stacking yalnızca LOYO
            preds = _cv_predict(model, X, y, groups[cv])
            block = meta.copy()
            block["layer"] = layer
            block["crop"] = crop_short
            block["cv"] = cv
            block["model"] = model
            block["y_true"] = y
            block["y_pred"] = preds
            block["abs_error"] = np.abs(y - preds)
            rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    return out[["layer", "crop", "cv", "model", "ilce_id", "ilce", "il",
                "year", "y_true", "y_pred", "abs_error"]]


def b0_rmse_for(layer: str, crop_short: str) -> float:
    return _b0_climatology_rmse(crop_subset(load_layer(layer), crop_short))


# ---------------------------------------------------------------------------
# Vectorized bootstrap (yüzlerce kat hızlı; sonuçlar naive döngüyle aynı)
# ---------------------------------------------------------------------------
def _r2_from(YT, err2_sum):
    mu = YT.mean(1, keepdims=True)
    ss_tot = ((YT - mu) ** 2).sum(1)
    return 1.0 - err2_sum / ss_tot


def boot_iid_metrics(yt: np.ndarray, yp: np.ndarray, n_boot: int, seed: int) -> dict:
    """Gözlem (iid case) bootstrap — tüm metrikler vektörize."""
    rng = np.random.default_rng(seed)
    n = len(yt)
    idx = rng.integers(0, n, size=(n_boot, n))
    YT = yt[idx]
    err = yp[idx] - YT
    err2 = err ** 2
    r2 = _r2_from(YT, err2.sum(1))
    return {
        "r2": r2,
        "rmse_kg_da": np.sqrt(err2.mean(1)),
        "mae_kg_da": np.abs(err).mean(1),
        "mape_pct": (np.abs(err) / np.abs(YT)).mean(1) * 100,
        "bias_kg_da": err.mean(1),
    }


def boot_cluster_metrics(yt: np.ndarray, yp: np.ndarray, groups: np.ndarray,
                         n_boot: int, seed: int) -> dict:
    """Küme (cluster) bootstrap — grupları (year/ilce) yeniden örnekler.
    Per-grup yeterli istatistiklerle tam vektörize; iid'den daha geniş/dürüst CI."""
    rng = np.random.default_rng(seed)
    err = yp - yt
    uniq = np.unique(groups)
    k = len(uniq)
    n_g = np.empty(k); sy = np.empty(k); syy = np.empty(k)
    sse = np.empty(k); sae = np.empty(k); sape = np.empty(k); serr = np.empty(k)
    for i, gv in enumerate(uniq):
        m = groups == gv
        yti = yt[m]; ei = err[m]
        n_g[i] = m.sum(); sy[i] = yti.sum(); syy[i] = (yti ** 2).sum()
        sse[i] = (ei ** 2).sum(); sae[i] = np.abs(ei).sum()
        sape[i] = (np.abs(ei) / np.abs(yti)).sum(); serr[i] = ei.sum()
    sel = rng.integers(0, k, size=(n_boot, k))
    nb = n_g[sel].sum(1); syb = sy[sel].sum(1); syyb = syy[sel].sum(1)
    sseb = sse[sel].sum(1); saeb = sae[sel].sum(1)
    sapeb = sape[sel].sum(1); serrb = serr[sel].sum(1)
    ss_tot = syyb - (syb ** 2) / nb
    return {
        "r2": 1.0 - sseb / ss_tot,
        "rmse_kg_da": np.sqrt(sseb / nb),
        "mae_kg_da": saeb / nb,
        "mape_pct": sapeb / nb * 100,
        "bias_kg_da": serrb / nb,
    }


def boot_paired_dr2(yt: np.ndarray, yp_a: np.ndarray, yp_b: np.ndarray,
                    n_boot: int, seed: int) -> np.ndarray:
    """Eşli bootstrap: aynı resample'da R²(b) − R²(a) dağılımı."""
    rng = np.random.default_rng(seed)
    n = len(yt)
    idx = rng.integers(0, n, size=(n_boot, n))
    YT = yt[idx]
    mu = YT.mean(1, keepdims=True)
    ss_tot = ((YT - mu) ** 2).sum(1)
    r2a = 1.0 - ((yp_a[idx] - YT) ** 2).sum(1) / ss_tot
    r2b = 1.0 - ((yp_b[idx] - YT) ** 2).sum(1) / ss_tot
    return r2b - r2a


def boot_skill_score(yt: np.ndarray, yp_model: np.ndarray, yp_b0: np.ndarray,
                     n_boot: int, seed: int) -> np.ndarray:
    """Eşli bootstrap: SS = 1 − RMSE_model/RMSE_B0 dağılımı."""
    rng = np.random.default_rng(seed)
    n = len(yt)
    idx = rng.integers(0, n, size=(n_boot, n))
    YT = yt[idx]
    rmse_m = np.sqrt(((yp_model[idx] - YT) ** 2).mean(1))
    rmse_b = np.sqrt(((yp_b0[idx] - YT) ** 2).mean(1))
    return 1.0 - rmse_m / rmse_b


def ci95(arr: np.ndarray) -> tuple[float, float]:
    a = arr[np.isfinite(arr)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def b0_per_sample(layer: str, crop_short: str) -> pd.DataFrame:
    """LOYO B0-climatology per-sample tahminleri (her ilçe için leave-one-out yıl
    ortalaması), bir layer×crop alt-kümesinde. Baseline-paired testler için.

    Dönüş: ilce_id, ilce, il, year, y_true, b0_pred (alt-küme satır sırası).
    """
    sub = crop_subset(load_layer(layer), crop_short)
    preds = np.full(len(sub), np.nan)
    for ilce_id, idxs in sub.groupby("ilce_id").groups.items():
        idxs = list(idxs)
        vals = sub.loc[idxs, "verim_kg_da"].astype(float)
        n = len(vals)
        if n <= 1:
            preds[[sub.index.get_loc(i) for i in idxs]] = vals.mean()
            continue
        total = vals.sum()
        for i in idxs:
            loo = (total - sub.loc[i, "verim_kg_da"]) / (n - 1)
            preds[sub.index.get_loc(i)] = loo
    out = sub[["ilce_id", "ilce", "il", "year"]].copy()
    out["y_true"] = sub["verim_kg_da"].astype(float).values
    out["b0_pred"] = preds
    return out
