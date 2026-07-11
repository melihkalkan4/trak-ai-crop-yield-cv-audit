# Reproduction — Paper 1

End-to-end regeneration of every table, analysis CSV, and figure from read-only
source artifacts. Nothing outside `paper1_generalization/` is written; no model is
retrained; the prospective frozen model is never touched.

## Environment
Use the repo venv (the exact versions that produced the originals — required for
bit-faithful reproduction): Python 3.13.2, numpy 2.4.3, scikit-learn 1.8.0,
xgboost 3.2.0, scipy 1.17.1, esda 2.9.0, libpysal 4.14.1, matplotlib 3.10.8.
Full freeze: `requirements_full_venv.txt`.

## Run everything
```bash
# from repo root
venv/Scripts/python.exe paper1_generalization/repro/run_all.py
```
Runtime ≈ 6–8 min (dominated by step 01 ≈ a few min and step 05 ablation ≈ 4–5 min,
both CV re-runs). Set `PYTHONIOENCODING=utf-8` (run_all does this) to avoid Windows
cp1254 console errors on non-ASCII characters.

## Steps (also runnable individually, in order)
| Step | Output |
|---|---|
| `01_regenerate_per_sample.py` | `analysis/per_sample_predictions.csv`, `recomputed_aggregate_metrics.csv`, `fidelity_check.csv` (**FIDELITY GATE**) |
| `02_bootstrap_ci.py` | `analysis/bootstrap_ci.csv` |
| `03_generalization_gap.py` | `analysis/generalization_gap.csv`, `generalization_gap_per_model.csv`, `gap_readme.md` |
| `04_baseline_superiority.py` | `analysis/baseline_superiority.csv` |
| `05_ablation.py` | `tables/T2_ablation.csv`, `analysis/ablation_per_model.csv` |
| `06_per_stage.py` | `tables/T3_per_stage.csv`, `analysis/prospective_overall.csv` |
| `07_feature_importance.py` | `tables/T4_feature_importance.csv` |
| `08_morans_i.py` | `analysis/morans_i.csv` |
| `09_tables.py` | `tables/T1_master_results.csv`, `tables/tables.md` |
| `10_figures.py` | `figures/F1–F6.{pdf,png}` |

### Optional live prospective re-run (needs GEE + CDS credentials, ~2 h)
Not part of `run_all.py` (it requires live API access and ~2 h of CDS downloads). Run separately:
| Step | Output |
|---|---|
| `11_prospective_real_coords.py [--mode smoke|full]` | `analysis/prospective_real/*` (real-coords FLOV: predictions, matched, per-stage, summary JSON). Live Sentinel-2 (GEE) + ERA5-Land (CDS); frozen LSTM inference only; real parcel polygon from 4 GPS corners. Isolation: writes only under my folder + additive untracked API cache. |
| `12_update_prospective.py` | `tables/T3_per_stage_real.csv`, `analysis/prospective_overall_real.csv`, `analysis/prospective_placeholder_vs_real.csv`, `figures/F3b_per_stage_real.{pdf,png}` |

## Fidelity gate
`repro_common.py` faithfully replicates the thesis cp25 Layer A/B/C pipeline
(`src/cp25/05,06,07`). Step 01 recomputes aggregate metrics and compares them to the
published `reports/cp25/05,06,07_*_results.csv`: **92/92 checks pass, max |Δ| = 0.0**;
per-sample champion predictions match existing CSVs to 0.0. This proves the
regenerated LOILO/Spatiotemporal per-sample predictions (used for the gap test and
CIs) are equally faithful. If the gate ever fails, step 01 stops and reports — no
fabricated numbers are emitted.

## Read-only inputs consumed
`data/processed/calibration_features_layer{A,B,C}.csv`,
`data/external/tuik/ilce_coords.csv`,
`reports/cp25/02_baselines.csv`, `05/06/07_loocv_predictions_*.csv`,
`08_perm_importance_*.csv`, `reports/prospective/EVR_01_*`.
