#!/usr/bin/env bash
# Consolidate ALL data we worked with into one self-contained bundle (by COPYING; originals untouched).
# Re-runnable (idempotent). Excludes the bulky GEE API cache and venv. Output: enrichment_v2/data_bundle/
set -e
ROOT="$(cd "$(dirname "$0")/../.." \&\& pwd)"  # repo root (was a personal absolute path; scrubbed for release)
B="$ROOT/enrichment_v2/data_bundle"
cd "$ROOT"

mkdir -p "$B"/01_inputs_used/{tuik,calibration,climate_daily_nasapower,yield,admin_shapefiles,cp25_published_results,prospective_parcel,advisor_todo}
mkdir -p "$B"/02_outputs_enrichment_v2 "$B"/03_outputs_paper1

# ---- 01 INPUTS USED (read-only sources the analysis consumed) ----
cp -f data/external/tuik/*.csv                         "$B/01_inputs_used/tuik/" 2>/dev/null || true
cp -f data/processed/calibration_features_layer*.csv   "$B/01_inputs_used/calibration/" 2>/dev/null || true
cp -f data/processed/calibration_holdout_*.csv         "$B/01_inputs_used/calibration/" 2>/dev/null || true
cp -f data/processed/calibration_train_set_*.csv       "$B/01_inputs_used/calibration/" 2>/dev/null || true
cp -f data/processed/master_feature_matrix_2017_2024.csv "$B/01_inputs_used/calibration/" 2>/dev/null || true
cp -f data/processed/soil_ilce.csv                     "$B/01_inputs_used/calibration/" 2>/dev/null || true
cp -f data/processed/openmeteo_ilce/*.csv              "$B/01_inputs_used/climate_daily_nasapower/" 2>/dev/null || true
cp -f data/yield/*.csv                                 "$B/01_inputs_used/yield/" 2>/dev/null || true
cp -f enrichment_v2/code/inputs/tur_*                  "$B/01_inputs_used/admin_shapefiles/" 2>/dev/null || true
cp -f reports/cp25/*.csv                               "$B/01_inputs_used/cp25_published_results/" 2>/dev/null || true
cp -f reports/prospective/EVR_01_*                     "$B/01_inputs_used/prospective_parcel/" 2>/dev/null || true
cp -f "$ROOT/../../Downloads/To-do.docx.pdf"           "$B/01_inputs_used/advisor_todo/" 2>/dev/null || true
cp -f paper1_generalization/logs/advisor_todo_extracted.txt "$B/01_inputs_used/advisor_todo/" 2>/dev/null || true

# ---- 02 OUTPUTS we produced (enrichment_v2) ----
cp -rf enrichment_v2/outputs                           "$B/02_outputs_enrichment_v2/" 2>/dev/null || true
cp -f  enrichment_v2/*.md enrichment_v2/*.txt          "$B/02_outputs_enrichment_v2/" 2>/dev/null || true

# ---- 03 OUTPUTS we produced (paper1 + revisions) ----
for d in analysis tables figures docs manuscript refs revisions logs; do
  if [ -d "paper1_generalization/$d" ]; then
    mkdir -p "$B/03_outputs_paper1/$d"
    cp -rf "paper1_generalization/$d/." "$B/03_outputs_paper1/$d/" 2>/dev/null || true
  fi
done

# strip any stray pyc/cache
find "$B" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$B" -name "*.pyc" -delete 2>/dev/null || true
echo "bundle built at: $B"
du -sh "$B" 2>/dev/null | awk '{print "total size:",$1}'
echo "file count: $(find "$B" -type f | wc -l)"
