"""run_all_revisions — reproduce every referee-revision artefact end-to-end (offline).

All steps read existing fidelity-verified artefacts + the read-only calibration inputs via
rev_common/repro_common; none needs network. (The live real-coords FLOV re-run is upstream:
../repro/11_prospective_real_coords.py.) Pinned env: ../repro/requirements_full_venv.txt.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [f"R{n:02d}" for n in range(1, 13)]
NAMES = {
    "R01": "R01_master_ledger", "R02": "R02_rolling_origin", "R03": "R03_clustered_inference",
    "R04": "R04_ablation_by_algorithm", "R05": "R05_staged_forecast",
    "R06": "R06_morans_sensitivity", "R07": "R07_perm_importance_foldwise",
    "R08": "R08_spatiotemporal_blocks", "R09": "R09_parcel_per_stage",
    "R10": "R10_hyperparameter_protocol", "R11": "R11_tables", "R12": "R12_figures",
}


def main() -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", TF_ENABLE_ONEDNN_OPTS="0")
    for s in STEPS:
        f = HERE / f"{NAMES[s]}.py"
        print(f"\n{'='*66}\n>>> {NAMES[s]}\n{'='*66}", flush=True)
        r = subprocess.run([sys.executable, str(f)], env=env)
        if r.returncode != 0:
            print(f"!!! {s} FAILED (exit {r.returncode}); stopping.", flush=True)
            return r.returncode
    print("\nALL REVISION STEPS COMPLETE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
