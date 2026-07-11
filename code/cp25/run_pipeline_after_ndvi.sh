#!/usr/bin/env bash
# ÇP-2.5 — NDVI ETL tamamlandıktan sonra Layer B/C zinciri.
# Kullanım:
#   bash src/cp25/run_pipeline_after_ndvi.sh
#
# Beklenen NDVI dosyaları: data/processed/ndvi_ilce/ndvi_{ilce_id}.csv  × 29
# Doğrulama:
#   ls data/processed/ndvi_ilce/*.csv | wc -l   # 29 olmalı

set -euo pipefail
cd "$(dirname "$0")/../.."

N_NDVI=$(ls data/processed/ndvi_ilce/*.csv 2>/dev/null | wc -l)
echo "NDVI dosyaları: $N_NDVI / 29"
if [ "$N_NDVI" -lt 29 ]; then
    echo "⚠️  NDVI ETL henüz tamamlanmadı (n=$N_NDVI < 29). Devam ediliyor (kısmi)."
fi

source venv/Scripts/activate
export PYTHONIOENCODING=utf-8

echo "=== Adım 1: Görev 4 (Layer A/B/C feature rebuild) ==="
python src/cp25/04_seasonal_features.py

echo
echo "=== Adım 2: Görev 6 (Layer B model yarışı) ==="
python src/cp25/06_layer_b_climate_ndvi.py

echo
echo "=== Adım 3: Görev 7 (Layer C + Stacking) ==="
python src/cp25/07_layer_c_full.py

echo
echo "=== Adım 4: Görev 8 XAI Layer B + C ==="
python src/cp25/08_xai_analysis.py --layer B
python src/cp25/08_xai_analysis.py --layer C

echo
echo "=== Adım 5: Görev 9 Anomaly Validation Layer B + C ==="
python src/cp25/09_anomaly_validation.py --layer B
python src/cp25/09_anomaly_validation.py --layer C

echo
echo "=== Adım 6: Görev 10 Belirsizlik Layer B + C ==="
python src/cp25/10_uncertainty.py --layer B --n-boot 100 --skip-gpr
python src/cp25/10_uncertainty.py --layer C --n-boot 100 --skip-gpr

echo
echo "=== Adım 7: Görev 11 Spatial Diagnostics (Layer B residuals ile) ==="
python src/cp25/11_spatial_diagnostics.py

echo
echo "=== Adım 8: Görev 12 Final Synthesis ==="
python src/cp25/12_final_synthesis.py

echo
echo "✅ Pipeline tamamlandı. Aşağıdaki raporlar üretildi:"
ls -1 reports/cp25/06_*.md reports/cp25/07_*.md reports/cp25/08_xai_*.md \
       reports/cp25/09_anomaly_validation_*.md reports/cp25/10_uncertainty_*.md \
       reports/cp25/12_final_synthesis.md 2>/dev/null
