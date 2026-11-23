#!/usr/bin/env python3
"""Hourly weather-type forecast for the next 48 h at the target point, from the latest HRRR extended run.
Applies the model trained on f00 analyses to HRRR forecast hours f01..f48 (f18 for non-synoptic runs).
Default: newest 00/06/12/18z run with all 48 hours available; daily job uses --run <today>12.
Outputs outputs/forecast_<run>.csv and .xlsx (valid time UTC + America/Chicago, predicted type, class probabilities,
key HRRR fields) and appends to logs/forecast_log.parquet for verify.py.
Usage: predict_latest.py [--run YYYYMMDDHH] [--station KMDQ] [--model prod|eval] [--no-log] [--outdir DIR]"""
import os, sys, json, argparse, datetime as dt, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from labels import CLASSES
import fetch_hrrr as F, features as X, model_lib as M
CT = ZoneInfo("America/Chicago"); OUT = os.path.join(ROOT, "outputs"); LOG = os.path.join(ROOT, "logs")

def get_run(run):
    df = F.fcst_zarr(run)
    if df is None or len(df) < (48 if run.hour % 6 == 0 else 18):
        df = F.fcst_grib(run, fhrs=range(1, 49 if run.hour % 6 == 0 else 19))
    return df

def latest_run():
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    for back in range(0, 30):
        run = now - dt.timedelta(hours=back)
        if run.hour % 6: continue
        if F.sess().head(F.GRIB_F.format(d=run.strftime("%Y%m%d"), h=run.hour, f=48) + ".idx").status_code == 200: return run
    raise SystemExit("no complete extended HRRR run found")

def analysis_hours(run, n=3):
    """f00 analyses for run-n+1 .. run (for 1 h / 3 h tendency features at the first forecast hours)."""
    rows = []
    for k in range(n - 1, -1, -1):
        t = run - dt.timedelta(hours=k); p = os.path.join(HRRR_DIR, t.strftime("%Y%m%d") + ".parquet"); r = None
        if os.path.exists(p):
            d = pd.read_parquet(p); d = d[pd.to_datetime(d.valid_utc) == t]
            if len(d) and d.iloc[0].src != "missing": r = d.iloc[0].to_dict()
        if r is None:
            vals = {col: F.zarr_value(t.strftime("%Y%m%d"), t.hour, lev, var) for (lev, var), col in FIELDS.items()}
            if sum(isinstance(v, str) for v in vals.values()) > 5:
                g = F.grib_point(F.GRIB.format(d=t.strftime("%Y%m%d"), h=t.hour)); vals = g or {}
            r = {"valid_utc": t, **{k2: (np.nan if isinstance(v, str) else v) for k2, v in vals.items()}}
        rows.append(r)
    a = pd.DataFrame(rows); a["fhr"] = 0; a["run_utc"] = run; return a

def archive_run(run, fc):
    """Keep the raw 12z point forecast (f01-f48) in data/fcst_12z/ (since 2026-11-14 the model also trains on these; retrain.sh picks them up)."""
    try:
        if run.hour != 12 or len(fc) < 48: return
        d = os.path.join(DATA, "fcst_12z"); os.makedirs(d, exist_ok=True); p = os.path.join(d, f"{run:%Y%m%d%H}.parquet")
        if not os.path.exists(p): fc.to_parquet(p, index=False)
    except Exception as e: print(f"(forecast archive not written: {e})", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--station", default=STATION)
    ap.add_argument("--model", default="prod"); ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--outdir", default=OUT, help="output folder (dry runs: use a scratch folder so posted files are not overwritten)"); a = ap.parse_args()
    run = dt.datetime.strptime(a.run, "%Y%m%d%H") if a.run else latest_run()
    fc = get_run(run)
    if fc is None or not len(fc): raise SystemExit(f"HRRR {run:%Y-%m-%d %HZ} not available yet")
    archive_run(run, fc)
    an = analysis_hours(run)
    seq = pd.concat([an, fc], ignore_index=True); seq["valid_utc"] = pd.to_datetime(seq.valid_utc)
    feats = X.build_features(seq); feats["fhr"] = seq.fhr.values
    feats = feats[feats.fhr >= 1].reset_index(drop=True); fc = seq[seq.fhr >= 1].reset_index(drop=True)
    model, meta = M.load_model(a.station, a.model); P = M.predict_proba(model, meta, feats)
    out = pd.DataFrame({
        "run_utc": run.strftime("%Y-%m-%d %HZ"), "fhr": fc.fhr.astype(int),
        "valid_utc": fc.valid_utc.dt.strftime("%Y-%m-%d %H:%MZ"),
        "valid_chicago": [t.tz_localize("UTC").tz_convert(CT).strftime("%a %Y-%m-%d %I:%M %p %Z") for t in fc.valid_utc],
        "predicted_type": [CLASSES[i] for i in P.argmax(1)], "confidence_pct": (P.max(1) * 100).round(1),
        "second_choice": [CLASSES[i] for i in np.argsort(-P, 1)[:, 1]],
        **{f"prob_{c}_pct": (P[:, i] * 100).round(1) for i, c in enumerate(CLASSES)},
        "hrrr_temp_F": ((fc.t2 - 273.15) * 9 / 5 + 32).round(1), "hrrr_dewpt_F": ((fc.td2 - 273.15) * 9 / 5 + 32).round(1),
        "hrrr_cloud_pct": fc.tcdc.round(0), "hrrr_precip_in_hr": (fc.prate * 141.732).round(3), "hrrr_vis_mi": (fc.vis / 1609.34).round(1),
        "hrrr_refc_dbz": fc.refc.round(1), "hrrr_sbcape": fc.sbcape.round(0)})
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(LOG, exist_ok=True)
    base = os.path.join(a.outdir, f"forecast_{a.station}_{run:%Y%m%d%H}"); out.to_csv(base + ".csv", index=False)
    write_xlsx(out, base + ".xlsx", run, meta, a.station)
    if not a.no_log:
        lg = out[["run_utc", "fhr", "valid_utc", "predicted_type"] + [c for c in out if c.startswith("prob_")]].copy()
        lg["station"] = a.station; lg["model_kind"] = a.model; lg["model_trained_through"] = meta["trained_on"][1]; lg["logged_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        lg["rule_type"] = [CLASSES[i] for i in M.rule_baseline(fc)]
        p = os.path.join(LOG, "forecast_log.parquet")
        if os.path.exists(p):
            old = pd.read_parquet(p); old = old[~((old.run_utc == lg.run_utc.iloc[0]) & (old.station == a.station))]; lg = pd.concat([old, lg])
        lg.to_parquet(p, index=False)
    print(f"HRRR {run:%Y-%m-%d %HZ} run, {len(out)} hours -> {base}.csv / .xlsx")
    s = out.copy(); s["day"] = s.valid_chicago.str[:14]
    for d, g in s.groupby("day", sort=False):
        print(f"  {d}: " + ", ".join(f"{k} {v}h" for k, v in g.predicted_type.value_counts().items()))
    return out

def write_xlsx(out, path, run, meta, stn):
    from openpyxl.styles import Font
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Forecast", index=False)
        notes = [f"HRRR run {run:%Y-%m-%d %HZ}; point lat {TARGET_LAT:.4f}, lon {TARGET_LON:.4f} (nearest HRRR grid point).",
                 f"Model: LightGBM trained on HRRR f00 analyses vs observed weather at {stn}, data {meta['trained_on'][0]} to {meta['trained_on'][1]} UTC.",
                 f"Held-out test year (eval model): accuracy {meta.get('test_accuracy', float('nan')):.1%}, macro-F1 {meta.get('test_macro_f1', float('nan')):.3f}. Forecast hours add HRRR forecast error, so expect lower skill at longer leads.",
                 "Classes: " + ", ".join(CLASSES) + ". Probabilities in percent."]
        vs = os.path.join(LOG, "verification_summary.txt")
        if os.path.exists(vs): notes += ["", "Recent verification:"] + open(vs).read().splitlines()
        pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)
        ws = xw.sheets["Forecast"]
        for c in ws[1]: c.font = Font(bold=True); ws.column_dimensions[c.column_letter].width = max(11, min(28, len(str(c.value)) + 2))
        ws.freeze_panes = "E2"; xw.sheets["Notes"].column_dimensions["A"].width = 150
if __name__ == "__main__": main()
