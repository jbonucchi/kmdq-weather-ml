#!/bin/bash
# Weekly (optional): refit the production model on all labeled data with the tuned settings.
# Full re-evaluation (new test year, metrics, eval model): python3 train.py && python3 lead_eval.py
cd "$(dirname "$0")" && python3 update_dataset.py --days 3 && python3 train.py --prod-only
