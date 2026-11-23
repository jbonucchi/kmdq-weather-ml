# Sky-cover work, 2026-11-14 (CST)
- Decision: ADOPTED forecast-trained model (f00 + archived 12z f01-f48, fcst weight x2, fhr feature) + sky scales PC 1.25 / MC 1.5 / OVC 1.15, fog boost 1.0.
- Backup of previous model/code/reports: models/backup_20261114/
- Full train (--test-end 2026-10-21 03:00) ~13:45 CST, lead_eval, retrain.sh (prod refit OK: 50,759 analysis + 35,842 forecast hours), verify.py --no-refresh OK,
  predict dry run 2026111412 -> /tmp/dryrun_20261114 OK (posted outputs untouched), run_hourly.sh OK (no new fields needed).
- README section "Sky cover: clear vs cloudy (2026-11-14)" written.
- Not used: A/W/N extra features (experiments/features_sky.py, fetch_neigh.py; neighbourhood data in data/hrrr_neigh, data/fcst_12z_neigh).
