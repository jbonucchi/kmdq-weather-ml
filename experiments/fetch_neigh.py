#!/usr/bin/env python3
"""Neighbourhood cloud statistics around the HRRR grid point (2026-11-14 sky experiments).
For TCDC (entire atmosphere) and LCDC (low cloud layer): mean/max over 3x3, mean/max/min over 7x7 (+/-9 km), mean and
fraction > 50 % over 15x15 (+/-21 km). All windows fit inside the 150x150 zarr chunk that holds the point (no extra requests).
Analyses -> data/hrrr_neigh/YYYYMMDD.parquet; 12z forecasts -> data/fcst_12z_neigh/YYYYMMDDHH.parquet.
Usage: fetch_neigh.py --start 2021-01-01 --end 2026-11-14 [--fcst] [--workers 48]"""
import os, sys, argparse, datetime as dt, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import fetch_hrrr as F
NFIELDS = {("entire_atmosphere", "TCDC"): "tcdc", ("low_cloud_layer", "LCDC"): "lcdc"}
ND = os.path.join(DATA, "hrrr_neigh"); NFD = os.path.join(DATA, "fcst_12z_neigh")
OY, OX = F.OY, F.OX
assert 7 <= OY < CHUNK - 7 and 7 <= OX < CHUNK - 7

def stats(a, col):
    """a: (..., 150, 150) array -> dict of neighbourhood stats arrays (leading dims kept)."""
    w = lambda r: a[..., OY - r:OY + r + 1, OX - r:OX + r + 1].reshape(a.shape[:-2] + (-1,))
    o = {}
    with np.errstate(all="ignore"):
        o[f"{col}_n3mean"] = np.nanmean(w(1), -1); o[f"{col}_n3max"] = np.nanmax(w(1), -1)
        o[f"{col}_n7mean"] = np.nanmean(w(3), -1); o[f"{col}_n7max"] = np.nanmax(w(3), -1); o[f"{col}_n7min"] = np.nanmin(w(3), -1)
        o[f"{col}_n15mean"] = np.nanmean(w(7), -1); o[f"{col}_n15gt50"] = np.nanmean(w(7) > 50, -1)
    return o

def _decode(content, lead=None):
    b = F.BLOSC.decode(content)
    for dtp, sz in (("<f4", 4), ("<f8", 8), ("<f2", 2)):
        n = len(b) // (CHUNK * CHUNK * sz)
        if len(b) == n * CHUNK * CHUNK * sz and (lead is None and n == 1 or lead is not None and n in (18, 48)):
            a = np.frombuffer(b, dtype=dtp).astype(float).reshape((n, CHUNK, CHUNK) if lead else (CHUNK, CHUNK))
            a[(a == -9999.0) | (a == -10000.0)] = np.nan; return a
    raise ValueError("bad chunk size")

def get(url, lead=None):
    for k in range(4):
        try:
            r = F.sess().get(url, timeout=60)
            if r.status_code in (403, 404): return None
            r.raise_for_status(); return _decode(r.content, lead)
        except Exception:
            time.sleep(1 + 2 * k)
    return None

def anl_hour(t):
    row = {"valid_utc": t}
    for (lev, var), col in NFIELDS.items():
        a = get(ZARR.format(d=t.strftime("%Y%m%d"), h=t.hour, lev=lev, var=var, cy=F.CY, cx=F.CX))
        if a is None: continue
        row.update({k: float(v) for k, v in stats(a, col).items()})
    return row

def run_anl(start, end, workers=48, verbose=True):
    os.makedirs(ND, exist_ok=True); now = dt.datetime.utcnow() - dt.timedelta(hours=2)
    days = pd.date_range(start, end, freq="D")[::-1]
    with ThreadPoolExecutor(workers) as ex:
        for d in days:
            p = os.path.join(ND, f"{d:%Y%m%d}.parquet")
            if os.path.exists(p):
                o = pd.read_parquet(p)
                if len(o) == 24 and o.drop(columns="valid_utc").notna().all(axis=None): continue
            hrs = [d.to_pydatetime() + dt.timedelta(hours=h) for h in range(24) if d.to_pydatetime() + dt.timedelta(hours=h) <= now]
            if not hrs: continue
            df = pd.DataFrame(list(ex.map(anl_hour, hrs))); df.to_parquet(p, index=False)
            if verbose and d.day == 1: print(f"{d:%Y-%m-%d} done", flush=True)

def fcst_run(run, workers=4):
    """Neighbourhood stats for all lead times of one run from the *_fcst.zarr; DataFrame(fhr, valid_utc, ...) or None."""
    out = {}
    def one(item):
        (lev, var), col = item
        return col, get(F.ZARR_FCST.format(d=run.strftime("%Y%m%d"), h=run.hour, lev=lev, var=var, cy=F.CY, cx=F.CX), lead=True)
    with ThreadPoolExecutor(workers) as ex: res = dict(ex.map(one, NFIELDS.items()))
    n = max((len(a) for a in res.values() if a is not None), default=0)
    if n == 0: return None
    for col, a in res.items():
        if a is not None: out.update(stats(a, col))
    df = pd.DataFrame(out); df.insert(0, "fhr", np.arange(1, n + 1)); df.insert(1, "valid_utc", [run + dt.timedelta(hours=int(h)) for h in df.fhr])
    return df

def fcst_grib(run, fhrs=range(1, 49)):
    """GRIB2 fallback (wrfsfcfXX via .idx byte ranges): same stats from the full field."""
    import pygrib, tempfile
    want = {f"{v}:{GRIB_LEVEL[lev]}": col for (lev, v), col in NFIELDS.items()}
    def one(fh):
        url = F.GRIB_F.format(d=run.strftime("%Y%m%d"), h=run.hour, f=fh); r = F.sess().get(url + ".idx", timeout=30)
        if r.status_code != 200: return None
        L = [l.split(":") for l in r.text.strip().splitlines()]; row = {"fhr": fh, "valid_utc": run + dt.timedelta(hours=fh)}
        for i, p in enumerate(L):
            k = f"{p[3]}:{p[4]}"
            if k in want and "ave" not in p[5]:
                e = int(L[i + 1][1]) - 1 if i + 1 < len(L) else ""
                g = F.sess().get(url, headers={"Range": f"bytes={p[1]}-{e}"}, timeout=60); g.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".grib2") as t:
                    t.write(g.content); t.flush(); gr = pygrib.open(t.name)
                    try: full = np.ma.filled(gr[1].values.astype(float), np.nan)
                    finally: gr.close()
                a = full[F.CY * CHUNK:(F.CY + 1) * CHUNK, F.CX * CHUNK:(F.CX + 1) * CHUNK]
                row.update({kk: float(v) for kk, v in stats(a, want[k]).items()})
        return row
    with ThreadPoolExecutor(6) as ex: rows = [r for r in ex.map(one, fhrs) if r]
    return pd.DataFrame(rows) if rows else None

def run_fcst(runs, workers=16):
    os.makedirs(NFD, exist_ok=True)
    todo = [r for r in runs if not os.path.exists(os.path.join(NFD, f"{r:%Y%m%d%H}.parquet"))]
    def one(r):
        df = fcst_run(r)
        if df is not None: df.to_parquet(os.path.join(NFD, f"{r:%Y%m%d%H}.parquet"), index=False)
        return df is not None
    with ThreadPoolExecutor(workers) as ex: ok = list(ex.map(one, todo))
    print(f"forecast runs: {sum(ok)}/{len(todo)} fetched", flush=True)

def load_anl():
    fs = sorted(os.path.join(ND, f) for f in os.listdir(ND) if f.endswith(".parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True); d["valid_utc"] = pd.to_datetime(d.valid_utc); return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2021-01-01"); ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--fcst", action="store_true", help="12z forecast runs that are cached in data/fcst_12z/")
    ap.add_argument("--workers", type=int, default=48); a = ap.parse_args()
    if a.fcst:
        runs = [dt.datetime.strptime(x[:10], "%Y%m%d%H") for x in sorted(os.listdir(os.path.join(DATA, "fcst_12z")))]
        run_fcst(runs)
    else: run_anl(a.start, a.end, a.workers)
