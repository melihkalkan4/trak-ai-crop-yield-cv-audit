# Spatial skill is not temporal skill — reproducibility code

Code accompanying:

> Kalkan, M., & Çavdaroğlu, G. Ç. (2026). **Spatial skill is not temporal skill: a cross-validation
> audit of satellite-driven winter-wheat and sunflower yield prediction in Trakya, Türkiye.**
> *International Journal of Engineering and Geosciences (IJEG).*

**Data:** the datasets are released separately on Mendeley Data (reserved DOI
`[10.17632/XXXXXXX.1 — inserted on deposit]`). This repository holds only code.
**License:** MIT (this code). The dataset is CC-BY-4.0. **Seeds fixed** (SEED = 42; bootstrap seed 12345).

This is an **audit** of validation design, not a model-maximization. The headline finding — a large
spatial-minus-temporal generalization gap — is reproducible offline from the released per-sample
predictions (no Earth Engine account needed); the satellite feature extraction is reproducible given
a Google Earth Engine account.

## Repository layout

```
code/
  cp25/         core pipeline: data assembly, baselines, seasonal features, CV metrics, tiers A/B/C
  enrichment_v2/  crop-specific 8-index extraction (GEE), soil/AWC, topography, LSTM, plains, audit
  repro/        byte-faithful reproduction of the published CV tables + generalization-gap test
  revisions/    referee-revision analyses (rolling-origin, per-algorithm ablation, staged forecast,
                clustered inference, spatiotemporal blocks, Moran's I, fold-wise importance)
reproduce.py    offline reproduction of the headline invariants (needs the Mendeley data)
requirements.txt  pinned environment
```

## Quickstart

```bash
python -m venv venv && . venv/Scripts/activate        # (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
# 1) OFFLINE — reproduce the headline results from the released data (no Earth Engine needed):
#    download the Mendeley deposit, then:
python reproduce.py --data /path/to/mendeley_deposit
# 2) FULL EXTRACTION (optional) — requires a Google Earth Engine service account:
#    place the key at keys/<service-account>.json  (git-ignored) and set CDS/NASA-POWER creds,
#    then run the enrichment_v2 / cp25 extraction scripts (see below).
```

## How each reported result is regenerated

| Article table/figure | Script(s) |
|---|---|
| Table 2 — LOYO vs matched climatology | `code/repro/` (CV metrics + matched baseline) |
| Table 3 / Fig 4 — spatial−temporal gap (ΔR²) | `code/repro/03_generalization_gap.py` |
| Table 4 — per-algorithm matched ablation | `code/revisions/` (matched ablation) |
| Table 5 / Fig 5 — parcel forecast vs persistence | `code/enrichment_v2/` (prospective parcel) |
| Table 6 / Fig 7 — fold-wise permutation importance | `code/revisions/` (foldwise importance) |
| Table 7 — rolling-origin forward skill | `code/revisions/` (rolling-origin) |
| Table 8 — staged issuance | `code/revisions/` (staged forecast) |
| Tables 9–11 / Fig 8 — crop-specific tiers A–D, indices, gap | `code/enrichment_v2/` (crop mask + 8 indices + tiers) |
| Table 12 — monthly-climate yield-LSTM | `code/enrichment_v2/` (yield-LSTM) |
| Mask validation (r = 0.954 / 0.615) | `code/enrichment_v2/` (crop mask) + `reproduce.py` |

`reproduce.py` recomputes, from the released data, the two headline invariants:
mask validation (Pearson r = 0.954 wheat / 0.615 sunflower, n = 216) and the climate-tier
spatial−temporal gap (+0.639 wheat / +0.580 sunflower). See the Supplementary for full methods.

## Data acquisition (Earth Engine)

Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED` + `S2_CLOUD_PROBABILITY`, s2cloudless < 30%), ESA WorldCover
cropland mask, SoilGrids and SRTM are pulled on Google Earth Engine. Authentication uses a service
account key placed in `keys/` (git-ignored — never committed). Two pipelines: the main-panel NDVI
(adaptive point-buffer, 16-day median composites, 30 m) and the crop-specific layer (administrative-
polygon cropland, per-phenological-window median of eight indices). See Supplementary S8–S9.

## Security

No secrets are committed. `keys/`, `.env`, and `.cdsapirc` are git-ignored. Provide your own
Earth Engine / CDS / NASA POWER credentials locally.

## Citation

Please cite the article (above) and the dataset (Mendeley DOI). A `CITATION.cff` is included.
