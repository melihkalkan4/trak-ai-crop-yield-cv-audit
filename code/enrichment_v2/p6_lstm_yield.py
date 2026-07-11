"""P6 — Per-crop yield LSTM (deep-learning model added to the comparison).

Input = monthly climate SEQUENCE (12 months x climate features) per district-year, built from the
existing daily NASA POWER files (no new fetch). Target = TÜİK district yield (verim_kg_da,
full_referans, 2004–2025). Evaluated out-of-fold under LOYO (year groups) and LOILO (ilce groups)
— same cluster-aware CV as the paper; compared to the published cp25 Layer A (season-aggregated
climate tabular models) and to a matched climatology baseline. Wheat + sunflower.

Integrity: scaler fit on TRAIN folds only; no test-fold tuning; fixed architecture; seeds set +
oneDNN off for reproducibility. Outputs only under enrichment_v2/. Honest reporting (small n).
"""
from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev2_common as E

DAILY_DIR = E.PROJECT_ROOT / "data" / "processed" / "openmeteo_ilce"
YIELDS = E.TUIK / "tuik_ilce_yields_full_referans.csv"
CROP_FULL = {"bugday": "bugday", "aycicegi": "aycicegi_yaglik"}
MONTH_FEATS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETROOT", "RH2M"]
AGG = {"T2M": "mean", "T2M_MAX": "mean", "T2M_MIN": "mean", "PRECTOTCORR": "sum",
       "ALLSKY_SFC_SW_DWN": "sum", "GWETROOT": "mean", "RH2M": "mean"}
SEED = 42
# published cp25 Layer A (full-panel climate tabular) best R2, for comparison
CP25_LAYERA = {"bugday": {"LOYO": -0.092, "LOILO": 0.441}, "aycicegi": {"LOYO": 0.051, "LOILO": 0.504}}


def build_monthly():
    """(ilce_id, year) -> (12, nfeat) monthly climate tensor."""
    seqs = {}
    for f in sorted(DAILY_DIR.glob("nasapower_ilce_*.csv")):
        iid = int(f.stem.split("_")[-1])
        d = pd.read_csv(f, parse_dates=["date"])
        d["year"] = d.date.dt.year
        d["month"] = d.date.dt.month
        g = d.groupby(["year", "month"]).agg(AGG).reset_index()
        for year, gy in g.groupby("year"):
            gy = gy.set_index("month").reindex(range(1, 13))
            arr = gy[MONTH_FEATS].to_numpy(dtype=float)
            seqs[(iid, int(year))] = arr
    return seqs


def make_lstm(n_feat):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((12, n_feat)),
        tf.keras.layers.Masking(0.0),
        tf.keras.layers.LSTM(16, dropout=0.2),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def cv_predict(X, y, groups):
    """Out-of-fold LSTM predictions (LeaveOneGroupOut). Scaler fit on train only."""
    import tensorflow as tf
    from sklearn.model_selection import LeaveOneGroupOut
    preds = np.zeros(len(y))
    nfeat = X.shape[2]
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        # impute NaN (rare missing months) with train mean per feature/timestep
        Xtr, Xte = X[tr].copy(), X[te].copy()
        mu = np.nanmean(Xtr.reshape(-1, nfeat), axis=0)
        for a in (Xtr, Xte):
            inds = np.where(np.isnan(a))
            a[inds] = np.take(mu, inds[2])
        # standardize features (fit on train), and target
        fmean = Xtr.reshape(-1, nfeat).mean(0); fstd = Xtr.reshape(-1, nfeat).std(0) + 1e-8
        Xtr = (Xtr - fmean) / fstd; Xte = (Xte - fmean) / fstd
        ym, ys = y[tr].mean(), y[tr].std() + 1e-8
        ytr = (y[tr] - ym) / ys
        tf.keras.utils.set_random_seed(SEED)
        m = make_lstm(nfeat)
        m.fit(Xtr, ytr, validation_split=0.15, epochs=120, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
        preds[te] = (m.predict(Xte, verbose=0).ravel() * ys) + ym
        tf.keras.backend.clear_session()
    return preds


def metrics(yt, yp):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return (float(r2_score(yt, yp)), float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)))


def b0_loo(df):
    pred = np.full(len(df), np.nan)
    for _, idxs in df.groupby("ilce_id").groups.items():
        idxs = list(idxs); v = df.loc[idxs, "verim_kg_da"].astype(float); tot, n = v.sum(), len(v)
        for i in idxs:
            pred[df.index.get_loc(i)] = (tot - df.loc[i, "verim_kg_da"]) / (n - 1) if n > 1 else v.mean()
    return pred


def main():
    seqs = build_monthly()
    print(f"[p6] monthly climate sequences: {len(seqs)} district-years", flush=True)
    y = pd.read_csv(YIELDS)
    rows, persample = [], []
    for crop in ("bugday", "aycicegi"):
        yc = y[(y.crop == CROP_FULL[crop]) & (y.verim_kg_da.notna())][["ilce_id", "year", "verim_kg_da"]]
        yc = yc[[(int(r.ilce_id), int(r.year)) in seqs for r in yc.itertuples()]].reset_index(drop=True)
        X = np.stack([seqs[(int(r.ilce_id), int(r.year))] for r in yc.itertuples()])
        yv = yc.verim_kg_da.to_numpy(float)
        yr = yc.year.to_numpy(int); il = yc.ilce_id.to_numpy(int)
        b0 = b0_loo(yc); rmse_b0 = float(np.sqrt(np.mean((yv - b0) ** 2)))
        res = {}
        for cv, g in [("LOYO", yr), ("LOILO", il)]:
            p = cv_predict(X, yv, g)
            r2, rmse, mae = metrics(yv, p)
            res[cv] = r2
            ss = 1 - rmse / rmse_b0 if cv == "LOYO" and rmse_b0 > 0 else None
            rows.append(dict(crop=crop, model="LSTM_monthly_climate", cv=cv, n=len(yv),
                             r2=round(r2, 4), rmse=round(rmse, 3), mae=round(mae, 3),
                             skill_score_vs_clim=round(ss, 4) if ss is not None else None,
                             cp25_layerA_r2=CP25_LAYERA[crop][cv],
                             delta_vs_layerA=round(r2 - CP25_LAYERA[crop][cv], 4)))
            for k in range(len(yv)):
                persample.append(dict(crop=crop, cv=cv, ilce_id=int(il[k]), year=int(yr[k]),
                                      y_true=float(yv[k]), y_pred=float(p[k])))
        gap = res["LOILO"] - res["LOYO"]
        rows.append(dict(crop=crop, model="LSTM_monthly_climate", cv="GAP_LOILO_minus_LOYO",
                         n=len(yv), r2=round(gap, 4)))
        print(f"[p6] {crop}: LOYO R2={res['LOYO']:+.3f} LOILO R2={res['LOILO']:+.3f} "
              f"gap={gap:+.3f} (cp25 LayerA LOYO {CP25_LAYERA[crop]['LOYO']:+.3f})", flush=True)
    pd.DataFrame(rows).to_csv(E.OUT / "lstm_yield_results.csv", index=False)
    pd.DataFrame(persample).to_csv(E.OUT / "lstm_yield_persample.csv", index=False)
    print("[p6] saved lstm_yield_results.csv + persample", flush=True)


if __name__ == "__main__":
    main()
