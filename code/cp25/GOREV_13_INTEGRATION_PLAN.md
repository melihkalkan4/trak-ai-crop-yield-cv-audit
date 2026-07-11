# Görev 13 — Sistem Entegrasyon Planı (NDVI ETL Sonrası)

**Strateji**: Mevcut 4 dosyaya **UPDATE**.  Yeni dosya kurma yok, breaking
change yok.  ÇP-2.5 v2 şampiyon (`models/cp25/champion_*.pkl`) artefaktları
mevcut sisteme yerleşir.

## Hedef Dosyalar (UPDATE)

| # | Dosya | Satır | İş |
|---|---|---|---|
| 1 | `src/cp2_model/inference_cp2.py` | 284 → ~345 | `predict_yield_kg_da()` yeni metod (mevcut `predict()` dokunulmaz) |
| 2 | `src/cp4_rag/demo.py` | 446 → ~490 | `get_cp2_prediction()` sonrası yield section eklenir |
| 3 | `src/dashboard_pages/_legacy_pages.py` | 1121 → ~1170 | `page_tarla()` "📊 Verim Projeksiyonu" bloğu ÇP-2.5 v2 inference ile değişir + SHAP top-3 kart |
| 4 | `src/mqtt_orchestrator.py` | 499 → ~530 | `detect_anomalies()` 5. kural: yield_dusus (<-%25 22-yıl ortalamadan) |

**Toplam**: ~170 satır ek, 4 dosya, mevcut surface area korunur.

---

## 1. `src/cp2_model/inference_cp2.py` — `predict_yield_kg_da()`

**Konum**: `predict()` fonksiyonunun **altına** ekle (line ~284, dosya sonu).
Mevcut `predict()` dokunulmaz.

```python
# ─────────────────────────────────────────────────────────────────────
# ÇP-2.5 v2 — Verim Tahmin Köprüsü (NDVI → kg/da)
# ─────────────────────────────────────────────────────────────────────
import pickle as _pickle
from datetime import datetime as _dt

_CP25_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "cp25")
_CP25_BUNDLE_CACHE: dict = {}


def _load_cp25_champion(crop_short: str):
    """Lazy load + cache champion bundle (Layer C öncelikli, B/A fallback)."""
    if crop_short in _CP25_BUNDLE_CACHE:
        return _CP25_BUNDLE_CACHE[crop_short]
    # Tier: champion → layer_c → layer_b → layer_a (en güncel hangisi varsa)
    for fname in (f"champion_{crop_short}.pkl",
                  f"layer_c_{crop_short}.pkl",
                  f"layer_b_{crop_short}.pkl",
                  f"layer_a_{crop_short}.pkl"):
        p = os.path.join(_CP25_MODELS_DIR, fname)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                bundle = _pickle.load(fh)
            _CP25_BUNDLE_CACHE[crop_short] = bundle
            return bundle
    raise FileNotFoundError(
        "ÇP-2.5 v2 bundle yok. "
        "`python src/cp25/12_final_synthesis.py` çalıştırın.")


def predict_yield_kg_da(crop_type: str, ndvi_predicted: float = None,
                         il: str = None, ilce_id: int = None,
                         feature_context: dict = None,
                         current_date: "pd.Timestamp" = None) -> dict:
    """ÇP-2 NDVI t+7 → ÇP-2.5 v2 verim tahmini (kg/dekar + PI + sapma).

    Args:
        crop_type: 'Wheat' veya 'Sunflower' (ÇP-2 contract'ı).
        ndvi_predicted: ÇP-2 predict() çıktısından NDVI t+7 (opsiyonel —
            None ise feature_context.NDVI_int kullanılır).
        il: 'Edirne'/'Kırklareli'/'Tekirdağ'.
        ilce_id: TÜİK ilçe id (1505=Lüleburgaz vs).  Per-ilçe bias correction.
        feature_context: dict ya da DataFrame.  Sezonluk climate + soil
            feature'larını içermeli (Görev 4 schema).  None → MFM proxy.
        current_date: Tahmin anı (sezon ilerleme % hesabı için).

    Returns:
        {
            'yield_kg_da'       : float,
            'yield_kg_da_lower' : float (%95 PI alt),
            'yield_kg_da_upper' : float (%95 PI üst),
            'lokal_22yil_ortalama': float | None,
            'sapma_pct'         : float | None,
            'sapma_yorum'       : str (Türkçe),
            'top_3_features'    : list[(name, shap_value)],
            'champion_model'    : str (örn. 'random_forest'),
            'layer'             : str ('A' | 'B' | 'C'),
            'model_version'     : str,
        }
    """
    crop_short = "bugday" if crop_type.lower().startswith(("w","b")) else "aycicegi"
    bundle = _load_cp25_champion(crop_short)

    # 1) Feature vector — feature_context dict ise dict kullan, yoksa MFM proxy
    feats = bundle["feature_cols"]
    if feature_context is None:
        feature_context = _build_default_context(crop_short, ndvi_predicted)
    X_row = {f: float(feature_context.get(f, 0.0)) for f in feats}

    # 2) Predict via bundle
    import pandas as _pd
    X_df = _pd.DataFrame([X_row], columns=feats)
    if bundle.get("scaler"):
        X_use = bundle["scaler"].transform(X_df)
    else:
        X_use = X_df.values
    yhat = float(bundle["model"].predict(X_use)[0])

    # 3) Per-ilçe bias correction (varsa)
    bias = bundle.get("per_il_bias_correction_kg_da", {}).get(il, 0.0)
    yhat_corr = yhat + float(bias)

    # 4) PI %95 = ±1.96 × RMSE_LOYO
    rmse = float(bundle["metrics_loyo"]["rmse_kg_da"])
    half = 1.96 * rmse
    pi_lower, pi_upper = yhat_corr - half, yhat_corr + half

    # 5) Sapma + yorum (TÜİK 22-yıl)
    stats_mean = _lookup_22yr_mean(il, crop_short)
    sapma_pct = (((yhat_corr - stats_mean) / stats_mean) * 100
                 if stats_mean else None)
    yorum = _interpret_sapma(sapma_pct or 0)

    # 6) SHAP top-3 (feature_importance_shap bundle'da varsa)
    top3 = list(
        (bundle.get("feature_importance_shap") or {}).items()
    )[:3] or _fallback_perm_imp(crop_short)

    return {
        "yield_kg_da":           round(yhat_corr, 1),
        "yield_kg_da_lower":     round(pi_lower, 1),
        "yield_kg_da_upper":     round(pi_upper, 1),
        "lokal_22yil_ortalama":  stats_mean,
        "sapma_pct":             round(sapma_pct, 1) if sapma_pct is not None else None,
        "sapma_yorum":           yorum,
        "top_3_features":        top3,
        "champion_model":        bundle["champion_name"],
        "layer":                 bundle["model_version"].split("-")[-1].replace("layer", "").upper(),
        "model_version":         bundle["model_version"],
    }


# Helpers
def _build_default_context(crop, ndvi):
    """ÇP-2.5 features'a default doldur (caller eksik bıraktıysa)."""
    return {"ndvi_max": ndvi or 0.5, "ndvi_mean_season": ndvi or 0.4,
            "gdd_cum_season": 1800, "tp_season_sum": 350,
            "tp_flowering": 30, "tp_grain_fill": 20}


def _lookup_22yr_mean(il, crop_short):
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "external",
                                       "tuik", "yield_stats_summary.csv"))
        crop_full = "bugday" if crop_short == "bugday" else "aycicegi_yaglik"
        row = df[(df["il"] == il) & (df["crop"] == crop_full)]
        return float(row["mean_kg_da"].iloc[0]) if not row.empty else None
    except Exception:
        return None


def _interpret_sapma(d):
    if d > 15:  return "Beklenenin üstünde verim — verimli sezon"
    if d > 5:   return "Hafifçe ortalamanın üstünde"
    if d > -5:  return "Ortalama yakınında"
    if d > -15: return "Hafif düşüş — risk izlenmeli"
    if d > -25: return "Belirgin düşüş — sulama/girdi gözden geçirilmeli"
    return "Kritik düşüş — kuraklık veya stres sinyali"


def _fallback_perm_imp(crop_short):
    """SHAP yoksa permutation importance CSV'sinden top-3 oku."""
    try:
        for layer in ("C", "B", "A"):
            p = os.path.join(PROJECT_ROOT, "reports", "cp25",
                              f"08_perm_importance_{layer}_{crop_short}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                return list(zip(df["feature"].iloc[:3],
                                df["imp_mean"].iloc[:3]))
    except Exception:
        return []


# Public surface
predict_yield_kg_da.__module__ = __name__
```

---

## 2. `src/cp4_rag/demo.py` — Yield section ekle

`get_cp2_prediction()` fonksiyonu sonuna **yield blokları** ekle:

```python
# get_cp2_prediction() sonunda, return cp2'den hemen önce:
try:
    from inference_cp2 import predict_yield_kg_da
    for crop_key, crop_name, il_default, ilce_default in [
        ("bugday", "Wheat", "Kırklareli", 1505),       # Lüleburgaz
        ("aycicegi", "Sunflower", "Tekirdağ", 1258),   # Çorlu
    ]:
        y = predict_yield_kg_da(
            crop_type=crop_name,
            ndvi_predicted=cp2[crop_key]["ndvi_predicted_t7"],
            il=il_default, ilce_id=ilce_default,
        )
        cp2[crop_key]["yield_kg_da"]       = y["yield_kg_da"]
        cp2[crop_key]["yield_pi"]          = (y["yield_kg_da_lower"],
                                               y["yield_kg_da_upper"])
        cp2[crop_key]["sapma_pct"]         = y["sapma_pct"]
        cp2[crop_key]["sapma_yorum"]       = y["sapma_yorum"]
        cp2[crop_key]["top_3_features"]    = y["top_3_features"]
        print(f"[CP-2.5] {crop_name:9s}: {y['yield_kg_da']:.0f} kg/da  "
              f"(PI %95: {y['yield_kg_da_lower']:.0f}-{y['yield_kg_da_upper']:.0f})  "
              f"sapma %{y['sapma_pct']:+.1f}  → {y['sapma_yorum']}")
except Exception as exc:
    print(f"[CP-2.5] uyari: yield tahmini eklenmedi ({exc})")
```

---

## 3. `src/dashboard_pages/_legacy_pages.py` — Yield card refactor

Mevcut "📊 Verim Projeksiyonu" bloğu (line 399-417) **DB'den okuyor**.  ÇP-2.5
v2 inference ile **canlı** üretsin:

```python
# Mevcut bölümü değiştir:
# ── 1B: Verim Projeksiyonu (ÇP-2.5 v2 canlı inference) ────────────────
st.subheader("📊 Verim Projeksiyonu — ÇP-2.5 v2 (ilçe-bazlı)")
try:
    from cp2_model.inference_cp2 import predict_yield_kg_da
    il_disp = tarla.get("il") or "Kırklareli"
    crop_ct = "Wheat" if crop_key == "bugday" else "Sunflower"
    y = predict_yield_kg_da(
        crop_type=crop_ct,
        ndvi_predicted=son_tahmin.get("ndvi_predicted") or 0.5,
        il=il_disp, ilce_id=tarla.get("ilce_id"),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tahmini Verim", f"{y['yield_kg_da']:.0f} kg/da",
              delta=f"%{y['sapma_pct']:+.1f}" if y['sapma_pct'] else None,
              delta_color="normal")
    c2.metric("%95 Güven Aralığı",
              f"{y['yield_kg_da_lower']:.0f}–{y['yield_kg_da_upper']:.0f}")
    c3.metric("22-yıl Lokal Ort.",
              f"{y['lokal_22yil_ortalama']:.0f} kg/da" if y['lokal_22yil_ortalama'] else "—")
    c4.metric("Model", f"Layer {y['layer']} · {y['champion_model'][:8]}")

    st.caption(f"**{y['sapma_yorum']}** · model: `{y['model_version']}`")

    if y['top_3_features']:
        st.markdown("**SHAP Top-3 Etkili Faktör:**")
        feat_df = pd.DataFrame(y['top_3_features'], columns=["Özellik", "Etki"])
        st.bar_chart(feat_df.set_index("Özellik")["Etki"], horizontal=True)
except Exception as exc:
    st.info(f"Verim tahmini hesaplanmadı: {exc}")
```

---

## 4. `src/mqtt_orchestrator.py` — Yield-based anomaly kuralı

`detect_anomalies()` 5. kural ekle (BBCH'den sonra, line ~155):

```python
# ── Kural 5: ÇP-2.5 Yield-Based Anomaly ──────────────────────────────
try:
    yield_pred  = cp2_result.get("yield_kg_da")
    yield_22yr  = cp2_result.get("lokal_22yil_ortalama")
    yield_pi_lo = cp2_result.get("yield_kg_da_lower")
    if yield_pred and yield_22yr:
        sapma = (yield_pred - yield_22yr) / yield_22yr * 100
        if sapma < -25.0:
            seviye = "KRITIK" if sapma < -40 else "YUKSEK"
            anomalies.append({
                "tip": "VERIM_DUSUS",
                "aciklama": f"Verim tahmini 22-yıl ortalamasından %{abs(sapma):.0f} "
                            f"düşük (tahmin: {yield_pred:.0f}, ort: {yield_22yr:.0f} kg/da)",
                "seviye": seviye,
            })
        elif yield_pi_lo and yield_pi_lo < yield_22yr * 0.5:
            anomalies.append({
                "tip": "VERIM_BELIRSIZ",
                "aciklama": f"Verim tahmin aralığı geniş; alt sınır "
                            f"{yield_pi_lo:.0f} kg/da (22-yıl ort %50 altı)",
                "seviye": "ORTA",
            })
except Exception:
    pass
```

---

## Görev 14 — Tez Bölüm 5.3 Revize (Layer B/C sonrası)

`thesis/bolum_5_yontem.md` Section 5.3 (Model Seti) içinde:

* Champion model isimleri (NDVI sonrası gerçekleşen şampiyonlar)
* Final hyperparams
* Layer B/C için NDVI feature listesi netleşmesi
* SHAP top-5 (Layer C'de soil eklenmesi sonrası nasıl değişiyor)

Plus `thesis/bolum_6_sonuclar.md` Section 6.2-6.3:
* Layer B model competition table
* Layer C + Stacking ensemble table
* H2/H3 hipotez sonuçları (PASS/FAIL ile niceleştirilmiş)

---

## Test Smoke (Görev 13 sonrası)

```bash
# 1. Inference köprüsü
python -c "
import sys; sys.path.insert(0, 'src/cp2_model')
from inference_cp2 import predict_yield_kg_da
print(predict_yield_kg_da('Wheat', ndvi_predicted=0.65,
                          il='Kırklareli', ilce_id=1505))
"

# 2. Demo end-to-end
python src/cp4_rag/demo.py

# 3. Dashboard preview
streamlit run src/dashboard.py
# → Tarla Durumu → "📊 Verim Projeksiyonu" → 4 kart + SHAP grafik

# 4. MQTT orchestrator dry-run
python -c "
import sys; sys.path.insert(0, 'src')
from mqtt_orchestrator import detect_anomalies
print(detect_anomalies(
    payload={'nem_1_pct':25,'nem_2_pct':22,'hastalik':None},
    cp2_result={'yield_kg_da':130,'lokal_22yil_ortalama':210,'yield_kg_da_lower':95}
))
"
# → VERIM_DUSUS anomaly bekleniyor (sapma %-38)
```

---

## Reproducibility Snapshot

Görev 13 commit sonrası:
```bash
pip freeze > requirements_freeze_$(date +%Y%m%d).txt
git add -A && git commit -m "feat(cp25-v2): integrate champion yield models into existing system (Görev 13)"
```
