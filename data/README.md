# Training data

Place the production-model training tables here:

| file | size | rows | contents |
|---|---|---|---|
| `KMDQ_training_analysis.parquet` | 18.9 MB | 50,947 | HRRR f00 analyses |
| `KMDQ_training_forecast12z.parquet` | 16.2 MB | 36,192 | archived 12z forecast hours (f01–f48) |

Columns: `valid_utc`, `source`, `run`, 122 model features, and `label`. Sample weights are applied in `train.py` and are not stored in these files.

These two Parquet files are included in this repository.
