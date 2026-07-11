"""run_all — Paper 1 sonuçlarını uçtan uca yeniden üretir.

Tüm tabloları, analiz CSV'lerini ve figürleri kaynak (read-only) artefaktlardan
yeniden üretir. Sabitlenmiş venv ile çalıştırın:

    venv/Scripts/python.exe paper1_generalization/repro/run_all.py

Sıra:
  01 per-sample (+ FIDELITY GATE; gate düşerse DURUR)
  02 bootstrap CI · 03 generalization gap · 04 baseline superiority
  05 ablation (CV re-run; ~birkaç dk) · 06 per-stage · 07 importance · 08 Moran's I
  09 tables · 10 figures

Hiçbir orijinal dosyaya yazılmaz; tüm çıktılar paper1_generalization/ içine gider.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_regenerate_per_sample.py",
    "02_bootstrap_ci.py",
    "03_generalization_gap.py",
    "04_baseline_superiority.py",
    "05_ablation.py",
    "06_per_stage.py",
    "07_feature_importance.py",
    "08_morans_i.py",
    "09_tables.py",
    "10_figures.py",
]


def main() -> int:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    for step in STEPS:
        print(f"\n{'='*70}\n>>> {step}\n{'='*70}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / step)], env=env)
        if r.returncode != 0:
            print(f"!!! {step} FAILED (exit {r.returncode}); stopping.", flush=True)
            return r.returncode
    print("\nALL STEPS COMPLETE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
