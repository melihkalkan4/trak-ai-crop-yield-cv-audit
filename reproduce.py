#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline reproduction of the paper's headline invariants from the released Mendeley dataset.
No Earth Engine / no network. Usage:  python reproduce.py --data /path/to/mendeley_deposit
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    sr = np.sum((y - p) ** 2); st = np.sum((y - y.mean()) ** 2)
    return 1 - sr / st if st > 0 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the downloaded Mendeley deposit root")
    D = Path(ap.parse_args().data)
    ok = True

    print("== 1) Crop-mask validation vs TÜİK planted area (Pearson) ==")
    area = pd.read_csv(D/"02_crop_specific_layer"/"crop_classified_area_ha.csv")
    t = pd.read_csv(D/"06_tuik_reference"/"tuik_ilce_crop_yields_2004_2025.csv")
    tw = t[t.crop == "bugday"].groupby(["ilce_id","year"]).ekilen_alan_da.sum().div(10).rename("wt")
    ts = t[t.crop == "aycicegi_yaglik"].groupby(["ilce_id","year"]).ekilen_alan_da.sum().div(10).rename("st")
    m = area.merge(tw, on=["ilce_id","year"]).merge(ts, on=["ilce_id","year"])
    wr = round(m.wheat_classified_ha.corr(m.wt), 3); sr = round(m.sun_classified_ha.corr(m.st), 3)
    print(f"   wheat r={wr} (exp 0.954) | sunflower r={sr} (exp 0.615) | n={len(m)} (exp 216)")
    ok &= (wr == 0.954 and sr == 0.615 and len(m) == 216)

    print("== 2) Spatial-minus-temporal gap, climate tier (from per-sample predictions) ==")
    pp = pd.read_csv(D/"04_folds_and_predictions"/"per_sample_predictions_main.csv")
    for crop, mdl, exp in [("bugday","gpr",0.639), ("aycicegi","xgboost",0.580)]:
        s = pp[(pp.layer == "A") & (pp.crop == crop) & (pp.model == mdl)]
        g = r2(s[s.cv=="LOILO"].y_true, s[s.cv=="LOILO"].y_pred) - r2(s[s.cv=="LOYO"].y_true, s[s.cv=="LOYO"].y_pred)
        print(f"   {crop:9s} {mdl:8s} gap={g:+.3f} (exp {exp:+.3f})")
        ok &= abs(g - exp) <= 0.005

    print("\nRESULT:", "ALL PASS — dataset reproduces the published invariants." if ok else "*** FAIL ***")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
