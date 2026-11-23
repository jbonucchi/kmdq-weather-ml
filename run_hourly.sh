#!/bin/bash
# Hourly: append newest HRRR f00 hours + METAR labels (idempotent). Log -> logs/update.log
cd "$(dirname "$0")" && mkdir -p logs && python3 update_dataset.py --days 3 >> logs/update.log 2>&1; tail -1 logs/update.log
