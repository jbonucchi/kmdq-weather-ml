#!/usr/bin/env python3
"""Incrementally update the training dataset (idempotent, resumable; safe to run hourly).
 1. Fetch any missing/incomplete HRRR f00 days (default: last 3 days) at the grid point -> data/hrrr_days/YYYYMMDD.parquet
 2. Download recent routine METARs for the station from IEM, merge into data/labels_<STN>.parquet (dedup by hour)
 3. Rebuild data/dataset_<STN>.parquet = HRRR raw values + label (inner join on valid hour)
Usage: update_dataset.py [--days 3] [--station KMDQ] [--full-start 2021-01-01]"""
import os, sys, argparse, datetime as dt, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import fetch_hrrr, labels as L

def labels_path(stn): return os.path.join(DATA, f"labels_{stn}.parquet")
def dataset_path(stn): return os.path.join(DATA, f"dataset_{stn}.parquet")

def update_labels(stn, start, end):
    raw = L.download_asos(stn, start, end)
    new = L.hourly_labels(raw)
    p = labels_path(stn)
    if os.path.exists(p):
        old = pd.read_parquet(p); new = pd.concat([old[~old.valid_utc.isin(new.valid_utc)], new])
    new = new.sort_values("valid_utc").reset_index(drop=True); new.to_parquet(p, index=False)
    return new

def labels_from_csv(stn, csv):
    lab = L.hourly_labels(pd.read_csv(csv, low_memory=False)); lab.to_parquet(labels_path(stn), index=False); return lab

def rebuild(stn):
    hr = fetch_hrrr.load_all(); lab = pd.read_parquet(labels_path(stn))
    hr["valid_utc"] = pd.to_datetime(hr.valid_utc); lab["valid_utc"] = pd.to_datetime(lab.valid_utc)
    ds = hr.merge(lab, on="valid_utc", how="left")   # keep unlabeled hours too (needed for tendency features)
    ds.to_parquet(dataset_path(stn), index=False)
    return ds

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=3); ap.add_argument("--station", default=STATION)
    ap.add_argument("--full-start"); a = ap.parse_args()
    today = dt.datetime.now(dt.timezone.utc).date()
    start = pd.Timestamp(a.full_start).date() if a.full_start else today - dt.timedelta(days=a.days)
    fetch_hrrr.run(start.isoformat(), today.isoformat(), verbose=False)
    update_labels(a.station, start, today)
    ds = rebuild(a.station)
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M} {a.station}: dataset {len(ds)} hours, {ds.label.notna().sum()} labeled, "
          f"latest HRRR {ds.valid_utc.max()} UTC, latest label {ds.loc[ds.label.notna(), 'valid_utc'].max()} UTC")
