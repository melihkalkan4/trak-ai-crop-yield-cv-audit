"""ÇP-2.5 / Görev 3 — NASA POWER ile ilçe-bazlı climate ETL.

PIVOT (2026-05-23): Open-Meteo Archive sandbox'tan erişilemediği için
NASA POWER'a geçildi.  Aynı agronomik değişken seti, MERRA-2 reanalysis
tabanlı.  Akademik defansta: MERRA-2 (NASA/GMAO) ve ERA5-Land (ECMWF)
karşılaştırılabilir kalitede reanalysis ürünleridir; FAO AquaCrop +
USDA-ARS standardı NASA POWER kullanır.

API
---
GET https://power.larc.nasa.gov/api/temporal/daily/point
    ?parameters=T2M_MAX,T2M_MIN,T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,
                GWETROOT,WS10M,RH2M
    &community=AG
    &longitude=...&latitude=...
    &start=YYYYMMDD&end=YYYYMMDD
    &format=JSON

NASA POWER 10-yıl/istek limiti var → 22 yılı 3 chunk'a böleriz:
2004-2013, 2014-2023, 2024-2025.  29 ilçe × 3 chunk = ~87 istek (~1 dk).

Değişken sözlüğü
----------------
* T2M_MAX        — Günlük max sıcaklık (°C)
* T2M_MIN        — Günlük min sıcaklık (°C)
* T2M            — Günlük ort. sıcaklık (°C)
* PRECTOTCORR    — Düzeltilmiş günlük yağış (mm/gün)
* ALLSKY_SFC_SW_DWN — Tüm-gökyüzü yüzey kısa dalga radyasyon (kWh/m²/gün)
* GWETROOT       — Kök bölge nem indeksi (0-1, oransal)
* WS10M          — 10 m rüzgar hızı (m/s)
* RH2M           — Bağıl nem (%)

Çıktı
-----
``data/processed/openmeteo_ilce/nasapower_ilce_{ilce_id}.csv`` günlük 2004-2025
``reports/cp25/03_climate_fetch_log.md``                       her ilçe durum
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("cp25.task03")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUIK_DIR     = PROJECT_ROOT / "data" / "external" / "tuik"
OUT_DIR      = PROJECT_ROOT / "data" / "processed" / "openmeteo_ilce"
LOG_DIR      = PROJECT_ROOT / "logs"
REPORT_DIR   = PROJECT_ROOT / "reports" / "cp25"
for d in (OUT_DIR, LOG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = [
    "T2M_MAX", "T2M_MIN", "T2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "GWETROOT",
    "WS10M", "RH2M",
]
START_YEAR = 2004
END_YEAR   = 2025
CHUNK_YEARS = 10                       # NASA POWER ~10-yıl/istek limit
TIMEOUT_S = 60.0
AUDIT_LOG = LOG_DIR / "nasapower_audit.jsonl"


# ---------------------------------------------------------------------------
def _audit(event: str, payload: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(),
           "event": event, "source": "nasapower", **payload}
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _chunks(start_y: int, end_y: int):
    cur = start_y
    while cur <= end_y:
        last = min(cur + CHUNK_YEARS - 1, end_y)
        yield (cur, last)
        cur = last + 1


def _fetch_one_chunk(lat: float, lon: float,
                     start_y: int, end_y: int,
                     retries: int = 3) -> Optional[pd.DataFrame]:
    """Single NASA POWER request → DataFrame[date, vars...]."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",                # AGroclimatology
        "longitude": lat and lon,         # placeholder; set below
    }
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude":  round(lat, 4),
        "start":     f"{start_y}0101",
        "end":       f"{end_y}1231",
        "format":    "JSON",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            r = requests.get(POWER_URL, params=params, timeout=TIMEOUT_S)
            r.raise_for_status()
            data = r.json()
            params_block = (data.get("properties", {}) or {}).get("parameter", {}) or {}
            if not params_block:
                _audit("empty_chunk", {"lat": lat, "lon": lon,
                                       "chunk": [start_y, end_y]})
                return None
            # Each variable is {YYYYMMDD: value}; align on the date set
            all_dates = sorted(next(iter(params_block.values())).keys())
            rows = []
            for d in all_dates:
                row = {"date": pd.to_datetime(d, format="%Y%m%d")}
                for v in PARAMETERS:
                    val = params_block.get(v, {}).get(d)
                    # NASA POWER uses -999 as nodata sentinel
                    if val is not None and val <= -990:
                        val = None
                    row[v] = val
                rows.append(row)
            df = pd.DataFrame(rows)
            _audit("ok", {"lat": lat, "lon": lon,
                          "chunk": [start_y, end_y],
                          "rows": len(df),
                          "elapsed_s": round(time.time() - t0, 2),
                          "attempt": attempt})
            return df
        except Exception as exc:                                    # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("chunk %d-%d lat=%.3f lon=%.3f attempt %d failed: %s; "
                           "retry %ds", start_y, end_y, lat, lon, attempt, exc, wait)
            time.sleep(wait)
    _audit("failed", {"lat": lat, "lon": lon,
                      "chunk": [start_y, end_y],
                      "error": str(last_exc)[:200]})
    return None


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _cache_path(ilce_id: int) -> Path:
    return OUT_DIR / f"nasapower_ilce_{ilce_id}.csv"


def fetch_ilce(ilce_id: int, lat: float, lon: float,
               force: bool = False) -> Optional[pd.DataFrame]:
    """Fetch ALL chunks for one ilce and write a single CSV."""
    cache = _cache_path(ilce_id)
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["date"])

    chunks: list[pd.DataFrame] = []
    for s, e in _chunks(START_YEAR, END_YEAR):
        df = _fetch_one_chunk(lat, lon, s, e)
        if df is None or df.empty:
            logger.warning("ilce_id=%d chunk %d-%d empty", ilce_id, s, e)
            continue
        chunks.append(df)
    if not chunks:
        return None
    full = pd.concat(chunks, ignore_index=True).drop_duplicates("date")
    full = full.sort_values("date").reset_index(drop=True)
    full.to_csv(cache, index=False)
    return full


# ---------------------------------------------------------------------------
def main(dry_run_ilce_id: Optional[int] = None,
         only_trakya: bool = True) -> None:
    coords = pd.read_csv(TUIK_DIR / "ilce_coords.csv")
    if only_trakya:
        coords = coords[coords["is_trakya"]].reset_index(drop=True)
    logger.info("loaded %d ilçe (only_trakya=%s)", len(coords), only_trakya)

    if dry_run_ilce_id is not None:
        coords = coords[coords["ilce_id"] == dry_run_ilce_id].reset_index(drop=True)
        if coords.empty:
            logger.error("dry-run ilce_id=%d not in coords", dry_run_ilce_id)
            return
        logger.info("DRY RUN mode → only fetching ilce_id=%d (%s)",
                    dry_run_ilce_id, coords.iloc[0]["ilce"])

    rows = []
    t_start = time.time()
    for i, r in coords.iterrows():
        ilce_id = int(r["ilce_id"]); ilce = r["ilce"]; il = r["il"]
        out = _cache_path(ilce_id)

        if out.exists():
            df = pd.read_csv(out, parse_dates=["date"])
            rows.append({"ilce_id": ilce_id, "ilce": ilce, "il": il,
                         "lat": r["lat"], "lon": r["lon"],
                         "status": "CACHE_HIT", "n_days": len(df),
                         "first_date": str(df["date"].min().date()),
                         "last_date": str(df["date"].max().date()),
                         "sha256": _file_sha256(out)})
            logger.info("[%2d/%d] %s (id=%d): cache hit (%d days)",
                        i+1, len(coords), ilce, ilce_id, len(df))
            continue

        logger.info("[%2d/%d] %s (id=%d) lat=%.4f lon=%.4f — fetching ...",
                    i+1, len(coords), ilce, ilce_id, r["lat"], r["lon"])
        df = fetch_ilce(ilce_id, r["lat"], r["lon"])
        if df is None or df.empty:
            rows.append({"ilce_id": ilce_id, "ilce": ilce, "il": il,
                         "lat": r["lat"], "lon": r["lon"],
                         "status": "FAILED", "n_days": 0,
                         "first_date": None, "last_date": None, "sha256": None})
            continue
        rows.append({"ilce_id": ilce_id, "ilce": ilce, "il": il,
                     "lat": r["lat"], "lon": r["lon"],
                     "status": "OK", "n_days": len(df),
                     "first_date": str(df["date"].min().date()),
                     "last_date": str(df["date"].max().date()),
                     "sha256": _file_sha256(out)})

    elapsed = time.time() - t_start
    logger.info("ETL bitti: %d ilçe, %.1f saniye", len(rows), elapsed)

    # ---- Report ----
    log = pd.DataFrame(rows)
    n_ok = int((log["status"] == "OK").sum())
    n_cache = int((log["status"] == "CACHE_HIT").sum())
    n_fail = int((log["status"] == "FAILED").sum())

    md = ["# ÇP-2.5 — Görev 3: NASA POWER İlçe ETL Raporu", "",
          "## Kaynak (PIVOT)", "",
          "- **API**: NASA POWER (MERRA-2 reanalysis, NASA/GMAO)",
          "- **URL**: https://power.larc.nasa.gov/api/temporal/daily/point",
          "- **Community**: AG (Agroclimatology)",
          f"- **Değişkenler**: {', '.join(PARAMETERS)}",
          f"- **Dönem**: {START_YEAR}-01-01 → {END_YEAR}-12-31",
          "",
          "## Pivot gerekçesi",
          "",
          "Open-Meteo Archive sandbox network'ünden erişilemedi (HTTPSConnectionPool",
          "timeout).  CDS ERA5-Land erişilebilir ama wall-clock olarak ~100 saat",
          "tahmin edildi.  NASA POWER MERRA-2 reanalysis tabanlı, FAO AquaCrop ve",
          "USDA-ARS standardı.  Akademik defansta MERRA-2 ↔ ERA5-Land karşılaştırması",
          "mevcut literatürde (Reichle 2017, ECMWF 2019) eşdeğer kalitede gösterilir.",
          "",
          "## Sonuç",
          "",
          f"- Toplam ilçe: **{len(log)}**",
          f"- ✅ OK:        {n_ok}",
          f"- ⏩ CACHE_HIT: {n_cache}",
          f"- ❌ FAILED:    {n_fail}",
          f"- Wall-clock:  {elapsed:.1f} s",
          f"- Audit log:    `{AUDIT_LOG.relative_to(PROJECT_ROOT)}`",
          "",
          "## İlçe başına"]
    md.append("")
    md.append("| ilce_id | İlçe | İl | Status | Days | İlk | Son | SHA-256 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in log.iterrows():
        md.append(f"| {r['ilce_id']} | {r['ilce']} | {r['il']} | {r['status']} | "
                  f"{r['n_days']} | {r['first_date']} | {r['last_date']} | "
                  f"`{r['sha256']}` |")
    md.append("")
    md.append("## Akademik notlar")
    md.append("- MERRA-2 vs ERA5-Land farkı tezde Bölüm 4.2'de açıkça raporlanacak.")
    md.append("- ET0 türevi NASA POWER direct vermez; T+RH+rad'tan Penman-Monteith ile")
    md.append("  feature_builder katmanında türetilecek (Görev 4).")

    (REPORT_DIR / "03_climate_fetch_log.md").write_text("\n".join(md), encoding="utf-8")
    log.to_csv(REPORT_DIR / "03_climate_fetch_log.csv", index=False)
    logger.info("rapor → %s", REPORT_DIR / "03_climate_fetch_log.md")
    print("\n=== ÖZET ===")
    print(f"OK={n_ok}  CACHE={n_cache}  FAIL={n_fail}  wall-clock={elapsed:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", type=int, default=None,
                   help="Single ilce_id for dry-run (e.g. 1505 Lüleburgaz)")
    p.add_argument("--all", action="store_true",
                   help="Fetch all Trakya ilçes (default: just dry-run)")
    args = p.parse_args()
    if args.dry_run is not None:
        main(dry_run_ilce_id=args.dry_run)
    elif args.all:
        main()
    else:
        # Default: 1 ilce dry-run sanity check
        main(dry_run_ilce_id=1505)  # Lüleburgaz
