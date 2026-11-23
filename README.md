# HRRR → KMDQ weather-type ML

Predicts the **observed weather type at KMDQ** (Madison County Executive Airport, AWOS) from HRRR fields at the HRRR
3 km grid point nearest **34.8108, -86.7272**, and issues an **hourly 48 h forecast** by applying the model to HRRR f01–f48.


## This repository

This project is a LightGBM model that predicts the weather type at **KMDQ (Madison County Executive Airport)** from NOAA HRRR fields at the nearest 3 km grid point (34.7981, −86.7271). Labels come from KMDQ METARs. The production model included here was refit on 22 Nov 2026.

### How to run it

Python 3 is required (`python3`). From a virtual environment:

```bash
pip install pandas numpy lightgbm scikit-learn requests openpyxl pyarrow pygrib numcodecs
```

From the repository root:

```bash
# Hourly: append the newest HRRR analyses and METAR labels
./run_hourly.sh
# or: python3 update_dataset.py --days 3

# Daily 48-hour forecast from the 12z HRRR run (f01–f48)
./run_daily.sh
# or: python3 predict_latest.py --run YYYYMMDD12

# Weekly production refit (keeps the tuned class weights)
./retrain.sh
# or: python3 train.py --prod-only

# Full train and evaluation, then skill by forecast hour
python3 train.py --test-end '2026-10-21 03:00'
python3 lead_eval.py
```

Once `models/model_KMDQ_prod.txt` is present alongside `models/model_KMDQ_prod.json`, `predict_latest.py` can be used without retraining. Raw HRRR day files and the `data/fcst_12z/` archive are not in this repository; rebuild them with `python3 fetch_hrrr.py --start 2021-01-01` and the update/forecast scripts if you need the full cache.

### Training data

The tables used to train the current production model have columns `valid_utc`, `source`, `run`, 122 model features, and `label`. They are not the raw HRRR archive; features are already joined to labels. Sample weights are applied inside `train.py` (forecast hours count double; fog, drizzle, rain, and heavy rain are upweighted; class-weight power is 0.25) and are **not** stored in these files.

| file | size | rows | contents |
|---|---|---|---|
| `data/KMDQ_training_analysis.parquet` | 18.9 MB | 50,947 | HRRR f00 analyses |
| `data/KMDQ_training_forecast12z.parquet` | 16.2 MB | 36,192 | archived 12z forecast hours (f01–f48) |

The forecast-row count is 36,192, a few short of the 36,218 recorded in the model metadata, because of a slightly different cutoff for which forecast runs are included.

Those two Parquet files and the production model weights (`models/model_KMDQ_prod.txt`, 5.5 MB) are included in this repository. The model metadata is in `models/model_KMDQ_prod.json`.

### Current benchmarks

Held-out test year (2025-10-21 → 2026-10-21), after the 14 Nov 2026 forecast-training and sky-cover changes:

- 12z f01–f48 forecasts: **65.3%** accuracy (macro-F1 **0.28**)
- f00 analyses: **71.3%** accuracy (macro-F1 **0.41**)

Cite the 12z f01–f48 figure in forecast accuracy summaries. Detail is in the 14 Nov 2026 section below.

## Point and station
* HRRR grid point (y=411, x=1227): **34.7981, -86.7271**, 1.41 km south of the target, elevation 238 m (`gridpoint.json`, `find_gridpoint.py`).
* Station **KMDQ**: 16.2 km NE. KHUA (Redstone, 15.2 km) is closer but reports only ~5 METARs/day; KHSV (19.3 km) is a full-time ASOS and is supported as an alternative (`--station KHSV` / `HRRR_ML_STATION=KHSV`).
  KMDQ is an AWOS: it reports thunder via lightning detection (TS/VCTS) but never reported freezing rain or sleet in 2021–2026. KHSV does.

## Data
* HRRR f00 analyses, hourly, 2021-01-01 → present (50,860 hours), from the `hrrrzarr` archive on AWS (same HRRR wrfsfc output as `noaa-hrrr-bdp-pds`, chunked 150×150, so one point costs about 50 KB per field).
  Fields stored as float16 before mid-2024 (PRATE/MASSDEN, which lose precision in float16, plus fields absent from zarr) are fetched by byte-range from the
  `noaa-hrrr-bdp-pds` GRIB2 via `.idx`. Spot checks match GRIB exactly. 88 fields (see `config.FIELDS`). LTNG is not available in hrrrzarr, so it is excluded.
* Raw cache: `data/hrrr_days/YYYYMMDD.parquet` (one file per UTC day, resumable).
* Labels: IEM ASOS archive (routine METARs), `data/labels_<STN>.parquet`. The routine METAR nearest the top of the hour within 30 min is used (KMDQ :55, KHSV :53).
* Merged dataset: `data/dataset_<STN>.parquet`.

## Classes (`labels.py`, first match wins)
Thunderstorm (any TS incl. VCTS) › Snow/ice (SN, SG, PL, GS, GR, FZRA, FZDZ, merged because they are rare) › Heavy rain (+RA) › Rain (RA) ›
Light rain/drizzle (-RA, any DZ, UP) › Fog/mist/haze (FG, BR, HZ, FU or vis < 3 mi) › sky cover from the highest-coverage layer: Clear (CLR/FEW), Partly cloudy (SCT), Mostly cloudy (BKN), Overcast (OVC/VV).
Note: AWOS/ASOS ceilometers only see clouds up to 12,000 ft, so "Clear" means no cloud below 12k ft.

## Model
LightGBM multiclass on 122 features (`features.py`) plus the forecast hour `fhr` (since 2026-11-14): surface, 925/850/700/500 mb, dewpoint depressions, lapse rates, winds, shear, clouds (layer cover, base/ceiling/top AGL),
reflectivity, VIL, cloud water/ice, PW, CAPE/CIN/LI/SRH, visibility, radiation, simulated satellite brightness temperatures, smoke, hour and day-of-year sin/cos, and 1 h/3 h tendencies.
Time split: train 2021-01 → 2024-10-20, validation → 2025-10-20 (used to tune the class-weight power (0.25, now fixed in `train.CLASS_WEIGHT_POWER`; `--search-power` reruns the old search), number of rounds, and the fog and precipitation emphasis), test = 2025-10-21 → 2026-10-21 (`train.py --test-end '2026-10-21 03:00'` reproduces it; default = last 12 months of data).
`models/model_<STN>_eval.*` = trained on train+val (scored on test). `models/model_<STN>_prod.*` = trained on all data (used for forecasts).

### Fog emphasis (2026-10-30)
The original model often called Fog/mist/haze hours Clear. Since 2026-10-30, Fog/mist/haze hours get **2x training weight** (`train.FOG_WEIGHT`) and the fog probability is
**multiplied by 2 and renormalized** at prediction time (`train.FOG_PROB_SCALE`, stored as `fog_prob_scale` in the model json and applied by `model_lib.predict_proba`).
Both were tuned on the **validation year** using archived 12z f01–f48 forecasts (`experiments/fog_val_fcst_grid.csv`; rule: max fog F1 with ≤1 pt forecast-accuracy loss and ≤2 pt f00-accuracy loss on validation).
`retrain.sh` / `train.py --prod-only` read the settings from `model_<STN>_eval.json`, so the weekly refit keeps them. (Since 2026-11-03 the settings live in `train.CLASS_WEIGHT_MULT` / `CLASS_PROB_SCALE`; see below.) The pre-change model, reports and code are in `models/backup_20261030/`
(restore: copy `model_KMDQ_*` back to `models/`; the old json has no `fog_prob_scale`, so no boost is applied).

Held-out test year (2025-10-21 → 2026-10-21, same split as before):

| | before (fog x1) | after (fog x2 + prob x2) |
|---|---|---|
| f00 analyses: accuracy | 73.6% | 71.0% |
| f00: macro-F1 | 0.443 | 0.430 |
| f00: fog recall / precision / F1 | 43.9% / 55.9% / 0.49 | 61.7% / 30.2% / 0.41 |
| f00: Clear precision | 84.2% | 84.6% |
| 12z f01–f48 forecasts: accuracy | 63.9% | 63.4% |
| forecasts: macro-F1 | 0.286 | 0.293 |
| forecasts: fog recall / precision / F1 | 17.5% / 37.5% / 0.24 | 30.4% / 29.9% / 0.30 |
| forecasts: Clear precision | 80.7% | 80.9% |

Many recent misses (e.g. 7 SM BR/HZ under clear skies, Oct 2026) have HRRR near-surface RH of only 30–85% and visibility > 7 mi, so the model gives them ~1–2% fog; no weighting catches those.

### Precipitation emphasis (2026-11-03)
The fog-weighted model called most rain Overcast (Oct 31 12z run: all 6 Rain hours and the 1 Heavy rain hour). Since 2026-11-03, on top of the fog settings:
**training weight** Light rain/drizzle x1.5, Rain x2, Heavy rain x2; **prediction-time probability** Light rain/drizzle x1.5, Rain x3, Heavy rain x3 (renormalized).
Thunderstorm stays x1 (TS weight/boost gave no validation gain). All settings are in `train.CLASS_WEIGHT_MULT` / `train.CLASS_PROB_SCALE`, saved as
`class_weight_mult` / `class_prob_scale` in the model json and applied by `model_lib.predict_proba` (`class_boost`); `retrain.sh` keeps them; jsons without these keys fall back to `fog_prob_scale`.
Tuned on the **validation year** only (`experiments/rain_tune.py`, `rain_tune_seeds.py` → `rain_val_grid_seeds.csv`: 7 training-weight configs × 140 boost combos × 3 seeds,
scored on archived 12z f01–f48 forecasts and f00). Rule: best precip F1 (any-precip F1 + mean drizzle/rain/heavy F1, forecasts) with ≤1 pt forecast-accuracy loss, ≤2 pt f00 loss, ≤2 pt fog-recall loss.
Test year scored once (`experiments/rain_final_test.py` → `rain_final_test.json`). "Any precip" = observed drizzle/rain/heavy rain vs forecast in that group.

| held-out test year | fog only (2026-10-30) | fog + precip (2026-11-03) |
|---|---|---|
| f00: accuracy / macro-F1 | 71.0% / 0.430 | 70.4% / 0.430 |
| f00: any-precip recall / precision | 63.4% / 66.5% | 78.4% / 58.3% |
| f00: Rain recall / precision | 23.8% / 39.5% | 41.3% / 27.1% |
| f00: fog recall | 61.7% | 62.6% |
| 12z f01–f48: accuracy / macro-F1 | 63.4% / 0.293 | 63.0% / 0.295 |
| forecasts: day-1 / day-2 accuracy | 65.1% / 61.7% | 64.7% / 61.2% |
| forecasts: any-precip recall / precision / F1 | 48.0% / 48.8% / 0.48 | 59.9% / 39.7% / 0.48 |
| forecasts: drizzle recall / precision | 41.7% / 38.8% | 48.2% / 30.9% |
| forecasts: Rain recall / precision | 14.3% / 19.6% | 24.6% / 16.0% |
| forecasts: Heavy rain recall | 0% | 0% |
| forecasts: Thunderstorm recall / precision | 15.2% / 15.8% | 11.8% / 19.7% |
| forecasts: precip hours called Overcast | 252 | 182 |
| forecasts: fog recall | 30.4% | 29.6% |
| forecasts: Clear / Overcast precision | 80.9% / 52.4% | 81.1% / 52.2% |

Trade-off: many more wet hours caught, at the cost of more drizzle false alarms (precision down), so any-precip F1 is flat on test (val: +0.015).
Recent out-of-sample runs (2026-10-20..11-02 12z, models trained before 2026-10-20 12z; `experiments/rain_recent_hours.csv`): wet hours forecast as some precip type 4 of 22 → 8 of 22;
none of the 14 Rain/Heavy rain hours was called Rain by either model (best: drizzle). On Nov 1 the HRRR itself had ~0 precip and low reflectivity during most of the observed rain (timing error), which no weighting can fix.
Backup of the fog-only model, reports, code and experiments: `models/backup_20261103/` (restore: copy `model_KMDQ_*` and the `.py` files back; its json has only `fog_prob_scale`, which `predict_proba` still honors).

### Fog weight trim (2026-11-11)
The fog x2 model had a run of early-morning fog false alarms (Nov 9 7 AM and Nov 10 7–8 AM CST; KMDQ was clear). Since 2026-11-11 the fog **training weight and prediction-time boost are both x1.5** (was x2); drizzle / Rain / Heavy rain weights are unchanged. Nearby modest boosts (x1.25 and x1.75) were checked on the same trees; x1.5 was kept as a small change that cuts false alarms without a meaningful accuracy loss. Comparison: `experiments/fog_reduce_20261111.py`. `retrain.sh` / `train.py --prod-only` keep the new settings from `model_<STN>_eval.json`. The previous (fog x2 + precip) model, reports and code are in `models/backup_20261111/` (restore: copy `model_KMDQ_*` and `train.py` back).

Held-out test year (2025-10-21 → 2026-10-21, same split as before):

| | fog x2 + precip (2026-11-03) | fog x1.5 + precip (2026-11-11) |
|---|---|---|
| f00: accuracy / macro-F1 | 70.4% / 0.430 | 72.5% / 0.445 |
| f00: fog recall / precision / F1 | 62.6% / 29.6% / 0.40 | 55.5% / 41.4% / 0.47 |
| f00: fog false alarms (of 326 fog hours, 8,709 total) | 486 | 256 |
| f00: any-precip recall / precision | 78.4% / 58.3% | 77.1% / 58.2% |
| 12z f01–f48: accuracy / macro-F1 | 63.0% / 0.295 | 63.0% / 0.286 |
| forecasts: day-1 / day-2 accuracy | 64.7% / 61.2% | 64.9% / 61.0% |
| forecasts: fog recall / precision / F1 | 29.6% / 29.3% / 0.29 | 22.7% / 32.5% / 0.27 |
| forecasts: fog false alarms (of 652 fog hours, 17,272 total) | 466 | 308 |
| forecasts: any-precip recall / precision / F1 | 59.9% / 39.7% / 0.48 | 59.1% / 40.5% / 0.48 |
| forecasts: Clear / Overcast precision | 81.1% / 52.2% | 80.6% / 52.1% |

Trade-off: about half as many f00 fog false alarms and higher fog precision, at the cost of some fog recall (especially on 12z forecasts, 29.6% → 22.7%). Overall forecast accuracy is unchanged; f00 accuracy and macro-F1 improved because many previous fog false alarms were actually clear or cloudy hours. Precipitation skill is essentially flat.

On the Nov 9–10 early-morning false alarms (4 forecast hours, all observed Clear at KMDQ), a prod-style model trained with the new weights on data through Nov 8 13 UTC flips 2 of 4 to Clear (Nov 9 7 AM CST from the Nov 9 12z run, fog 58% → 36%; Nov 10 8 AM CST from the Nov 10 12z run, fog 60% → 36%). The other two stay Fog (Nov 10 7 AM from the Nov 10 12z run, and Nov 10 8 AM from the Nov 9 12z run) because those hours remain high-confidence fog. Reducing only the inference boost, without retraining, flipped none of them. A missed real fog hour on Nov 8 at 9 PM CST (called Clear at 1–2% fog) is still missed.

## Scripts
| script | purpose |
|---|---|
| `update_dataset.py` / `run_hourly.sh` | hourly, idempotent: fetch missing f00 hours (last 3 days) and new METARs, then rebuild the dataset |
| `predict_latest.py [--run YYYYMMDDHH]` | 48 h hourly forecast from an extended (00/06/12/18z) run → `outputs/forecast_<STN>_<run>.csv/.xlsx` (UTC + Central time, type, confidence, all class probabilities, key HRRR fields), appended to `logs/forecast_log.parquet` |
| `verify.py` | scores logged forecasts against obs → `logs/verification.parquet`, `logs/verification_summary.txt/.json` (latest run, per-class hits/misses/false alarms, rolling 7/30-day day-1 and day-2 accuracy, rule-based comparison, test-year reference) |
| `run_daily.sh` | 9 AM CDT job: update, verify, forecast from today's 12z run (retries for up to about 45 min), then print the verification and forecast summary |
| `train.py [--station] [--prod-only] [--test-end]` / `retrain.sh` | full train and evaluation (use `--test-end '2026-10-21 03:00'` to reproduce the current split) / weekly production refit (keeps fog and precip settings) |
| `lead_eval.py` | test-year skill by lead time, using archived 12z f01–f48 forecasts (cached in `data/fcst_12z/`) |
| `fetch_hrrr.py --start --end` | historical backfill (resumable; newest days first) |

Reports: `reports/metrics_<STN>.json`, `reports/test_predictions_<STN>.csv`, `reports/lead_time_KMDQ.{json,csv}`.

## Extending / notes
* Older history: `python3 fetch_hrrr.py --start 2020-12-03` (HRRRv4 start). Earlier HRRR versions differ, so be careful with those.
* The 12z extended run's zarr forecast is published around 14:00 UTC (9:00 AM CDT). If it isn't up yet, `predict_latest.py` falls back to GRIB (about 1–3 min) and `run_daily.sh` retries.
* Forecast-hour skill is lower than analysis skill (see report). The model learns HRRR-analysis → obs, so HRRR forecast error adds to it.

### Sky cover: clear vs cloudy (2026-11-14)
Asked to separate Clear from Overcast/Mostly cloudy better. Previous model, reports and code: `models/backup_20261114/` (restore: copy `model_KMDQ_*`, `train.py`, `model_lib.py`, `predict_latest.py` and the `reports/` files back).

**Diagnosis** (`experiments/sky_diag.py` → `sky_diag_20261114.json`):
* Labels are not the problem. When KMDQ reported OVC/BKN and the model said Clear, the HRRR itself had almost no cloud (analysis median total cloud 7%, low cloud 3%); the observed decks were mostly 4,000–9,000 ft. Clear hours under high cloud (above the 12,000 ft ceilometer limit) are already handled: only 79 Clear hours of 4,700 were called Overcast at f00.
* Recent misses are the same: e.g. Nov 13 3–8 PM CST OVC029–033 while the 12z HRRR forecast had 0–9% cloud; Nov 6 evening and Nov 12 morning HRRR forecast a 100% stratus deck at ~450–800 m that never formed (KMDQ CLR). These are HRRR forecast errors at the point.
* On forecasts the model was trained only on HRRR analyses, so it trusted HRRR forecast cloud too much.

**Tried** (validation-year selection, test year scored once; `experiments/sky_*.py`):
* A: extra point features (RH at 925/850/700/500 mb, cloud below 12k ft proxy, ceiling/base < 12k ft flags, cloud depth, log cloud water/ice, solar-zenith clear-sky shortwave ratio, longwave excess, low cloud x RH) — f00 flat, forecasts flat.
* W: ±1 h/±2 h window max/mean of cloud fields — f00 +0.6 pt, forecasts flat.
* N: neighbourhood cloud stats (3x3, 7x7, 15x15 around the point, fetched for all 51k analyses and 750 forecast runs, `experiments/fetch_neigh.py`, data in `data/hrrr_neigh/`, `data/fcst_12z_neigh/`) — f00 sky F1 +0.014, forecasts flat. With forecast training (below) A+W+N added ~+0.01 sky F1 on the test year but was worse on the Oct 20–Nov 14 runs, so it is **not used** (no new data to collect).
* C: sky-class probability scales — adopted (below).
* **D: train on archived HRRR 12z forecasts (f01–f48) as well as analyses — adopted.**

**Adopted:** the model now trains on f00 analyses **plus ~36k archived 12z forecast hours** (`data/fcst_12z/`, weight x2 = `train.FCST_TRAIN_WEIGHT`), with the forecast hour `fhr` as a feature (analyses fhr=0). Prediction-time scales Partly cloudy x1.25, Mostly cloudy x1.5, Overcast x1.15; the fog boost is now x1.0 (training weight still x1.5) because the forecast-trained model calls fog more often — x1.0 matched the old validation fog false-alarm count. Iterations: early stopping on validation-year forecasts (73 → 83 rounds). `predict_latest.py` now archives each 12z run's raw point forecast to `data/fcst_12z/`, so the weekly `retrain.sh` refit keeps learning from new forecasts. `predict_latest.py --outdir DIR` writes a dry run elsewhere (use with `--no-log`).

| held-out test year | before (2026-11-11) | after (2026-11-14) |
|---|---|---|
| 12z f01–f48: accuracy / macro-F1 | 63.0% / 0.286 | **65.3%** / 0.282 |
| forecasts: day-1 / day-2 accuracy (lead_eval) | 65.8% / 62.3% | 67.6% / 64.8% |
| forecasts: F1 Clear / Partly / Mostly / Overcast | 0.828 / 0.268 / 0.153 / 0.530 | 0.839 / 0.283 / 0.211 / 0.567 |
| forecasts: Overcast→Clear / Clear→Overcast (hours) | 403 / 584 | 497 / 488 |
| forecasts: BKN or OVC→Clear / Clear→BKN or OVC | 824 / 738 | 942 / 658 |
| forecasts: fog F1 / recall / false alarms | 0.267 / 22.7% / 308 | 0.263 / 21.8% / 286 |
| forecasts: any-precip F1 / recall; Rain recall | 0.481 / 59.1%; 23.0% | 0.488 / 55.6%; 27.8% |
| f00 analyses: accuracy / macro-F1 | 72.5% / 0.445 | 71.3% / 0.412 |
| f00: fog F1 / false alarms | 0.474 / 256 | 0.445 / 229 |

Every lead bucket improved (f01-06 66.0→66.8%, f07-12 62.8→64.2%, f13-24 67.2→69.6%, f25-36 59.3→61.8%, f37-48 65.3→67.8%). The gain is mainly better Mostly cloudy/Overcast calls and less over-confidence; the total number of Clear↔Overcast mix-ups is about the same (987 → 985), with fewer false "Overcast" and more missed decks. The f00 (analysis) benchmark dropped ~1 pt because the model is now tuned for forecasts.
Recent out-of-sample check (12z runs Oct 20–Nov 14, models trained on data before Oct 20 12Z; `experiments/sky_recent*.py`): 77.5% → 77.9% (forecast-trained, base features); misses such as Nov 13 afternoon remain because the HRRR forecast itself had no cloud.
