#!/bin/bash
# Daily ~9 AM CDT: update data, verify past forecasts, then forecast f01-f48 from today's 12z HRRR run.
# Retries up to ~45 min if the 12z extended run is not fully published yet.
cd "$(dirname "$0")" && mkdir -p logs
python3 update_dataset.py --days 3 >> logs/update.log 2>&1
python3 verify.py > logs/verify_latest.txt 2>&1
RUN=$(date -u +%Y%m%d)12
for i in $(seq 1 10); do
  if python3 predict_latest.py --run "$RUN" > logs/predict_latest.txt 2>&1; then break; fi
  sleep 270
done
python3 verify.py --no-refresh > /dev/null 2>&1
echo "=== Verification ==="; cat logs/verify_latest.txt
echo "=== Forecast ==="; cat logs/predict_latest.txt
