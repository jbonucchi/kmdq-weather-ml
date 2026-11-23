#!/usr/bin/env python3
"""Skill vs forecast lead time: apply the EVAL model (trained only on data before the test year) to archived
HRRR 12z extended-run forecasts (f01-f48) for days in the test year, and score against station obs.
Caches each run's point forecast in data/fcst_12z/. Writes reports/lead_time_<STN>.json/.csv.
Usage: lead_eval.py [--station KMDQ] [--every 1] [--hour 12]"""
import os, sys, json, argparse, datetime as dt, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from labels import CLASSES
import fetch_hrrr as F, features as X, model_lib as M, update_dataset as U
FD = os.path.join(DATA, "fcst_12z")

def get(run):
    p = os.path.join(FD, f"{run:%Y%m%d%H}.parquet")
    if os.path.exists(p): return pd.read_parquet(p)
    df = F.fcst_zarr(run, workers=16)
    if df is not None: df.to_parquet(p, index=False)
    return df

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--station", default=STATION); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--hour", type=int, default=12); a = ap.parse_args(); os.makedirs(FD, exist_ok=True)
    stn = a.station; model, meta = M.load_model(stn, "eval")
    rep = json.load(open(os.path.join(ROOT, "reports", f"metrics_{stn}.json"))); t0 = pd.Timestamp(rep["split"]["test"][0])
    ds = pd.read_parquet(U.dataset_path(stn)); ds["valid_utc"] = pd.to_datetime(ds.valid_utc)
    lab = ds.dropna(subset=["label"]).set_index("valid_utc").label
    an = ds.set_index("valid_utc")
    runs = [d.to_pydatetime() + dt.timedelta(hours=a.hour) for d in pd.date_range(t0.normalize() + pd.Timedelta(days=1), ds.valid_utc.max() - pd.Timedelta(days=2), freq=f"{a.every}D")]
    with ThreadPoolExecutor(4) as ex: fcs = list(ex.map(get, runs))
    rows = []
    for run, fc in zip(runs, fcs):
        if fc is None: continue
        prev = an.loc[[run - dt.timedelta(hours=k) for k in (2, 1, 0) if run - dt.timedelta(hours=k) in an.index]].reset_index()
        prev["fhr"] = 0; seq = pd.concat([prev, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
        f = X.build_features(seq); f["fhr"] = seq.fhr.values; f = f[f.fhr >= 1]
        P = M.predict_proba(model, meta, f); fv = seq[seq.fhr >= 1]
        for i, (t, fh) in enumerate(zip(f.valid_utc, f.fhr)):
            if t in lab.index: rows.append({"run": run, "fhr": int(fh), "valid_utc": t, "obs": lab[t], "pred": CLASSES[P[i].argmax()],
                                           "rule": CLASSES[M.rule_baseline(fv.iloc[[i]])[0]]})
    r = pd.DataFrame(rows); r.to_csv(os.path.join(ROOT, "reports", f"lead_time_{stn}.csv"), index=False)
    # f00 reference on the same valid hours, from the test-set predictions
    tp = pd.read_csv(os.path.join(ROOT, "reports", f"test_predictions_{stn}.csv"), parse_dates=["valid_utc"]).set_index("valid_utc")
    yy = lambda s: s.map(CLASSES.index).values
    out = {"runs": int(r.run.nunique()), "hours": int(len(r)), "by_lead_bucket": {}, "by_fhr_accuracy": r.assign(h=r.pred == r.obs).groupby("fhr").h.mean().round(3).to_dict()}
    sub = tp.loc[tp.index.intersection(r.valid_utc.unique())]
    out["f00_same_valid_hours"] = {"accuracy": float((sub.pred == sub.obs).mean()), "macro_f1": M.metrics(yy(sub.obs), yy(sub.pred))["macro_f1"]}
    for name, lo, hi in (("f01-06", 1, 6), ("f07-12", 7, 12), ("f13-24", 13, 24), ("f25-36", 25, 36), ("f37-48", 37, 48)):
        g = r[(r.fhr >= lo) & (r.fhr <= hi)]; m = M.metrics(yy(g.obs), yy(g.pred))
        out["by_lead_bucket"][name] = {"hours": int(len(g)), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "rule_accuracy": float((g.rule == g.obs).mean()),
                                       "per_class_recall": {c: round(v["recall"], 3) for c, v in m["per_class"].items() if v["support"]}}
    json.dump(out, open(os.path.join(ROOT, "reports", f"lead_time_{stn}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("runs", "hours", "f00_same_valid_hours")}, default=str))
    for k, v in out["by_lead_bucket"].items(): print(k, v["hours"], f"acc {v['accuracy']:.3f} macroF1 {v['macro_f1']:.3f} rule {v['rule_accuracy']:.3f}")
if __name__ == "__main__": main()
